import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Set

from app.models.job import Job
from app.models.media import Asset, MediaAnalysis
from app.models.plan import EditPlan
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline
from app.models.timeline import Project, SubtitleCue

# Each entry upgrades the schema by one version; PRAGMA user_version records how many have been applied.
# Append new migrations to the end and never edit ones that have shipped.
_MIGRATIONS: List[List[str]] = [
    [
        "CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, data TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, version INTEGER NOT NULL, data TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, data TEXT NOT NULL)",
    ],
    [
        # Jobs are no longer tied to a project: analysis jobs belong to an asset.
        "CREATE TABLE jobs_v2 (id TEXT PRIMARY KEY, data TEXT NOT NULL)",
        "INSERT INTO jobs_v2 (id, data) SELECT id, data FROM jobs",
        "DROP TABLE jobs",
        "ALTER TABLE jobs_v2 RENAME TO jobs",
        "CREATE TABLE analyses (asset_id TEXT PRIMARY KEY, data TEXT NOT NULL)",
    ],
    [
        # The semantic timeline. Clips keep the few fields queries filter on as columns and
        # the rest as JSON; a workspace holds thousands of them at most, so plain columns and
        # a LIKE over the text are quicker than they need to be and carry no index to keep in
        # step. Full-text search was measured and rejected: SQLite's trigram tokenizer cannot
        # match a two-character Chinese word, which is most of them.
        "CREATE TABLE semantic_timelines (id TEXT PRIMARY KEY, input_hash TEXT NOT NULL UNIQUE, data TEXT NOT NULL)",
        "CREATE TABLE semantic_clips ("
        "id TEXT PRIMARY KEY, timeline_id TEXT NOT NULL, asset_id TEXT NOT NULL, level TEXT NOT NULL, "
        "kind TEXT NOT NULL, start_seconds REAL NOT NULL, end_seconds REAL NOT NULL, text TEXT NOT NULL, "
        "data TEXT NOT NULL)",
        "CREATE INDEX semantic_clips_timeline ON semantic_clips (timeline_id, level, start_seconds)",
        "CREATE INDEX semantic_clips_asset ON semantic_clips (asset_id, start_seconds)",
    ],
    [
        # Edit plans. Versioned the way projects are, so two sessions editing the same plan
        # cannot overwrite one another without noticing.
        "CREATE TABLE plans (id TEXT PRIMARY KEY, timeline_id TEXT NOT NULL, version INTEGER NOT NULL, data TEXT NOT NULL)",
        "CREATE INDEX plans_timeline ON plans (timeline_id)",
    ],
    [
        # Every version of every plan, kept. The plans table holds the one in use; this is
        # what it was before, so a round of feedback that made things worse can be undone.
        # Append-only: going back writes the old content as a new version rather than
        # rewinding, so the history is never shorter than the work that was done.
        "CREATE TABLE plan_versions (plan_id TEXT NOT NULL, version INTEGER NOT NULL, saved_at TEXT NOT NULL, "
        "note TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (plan_id, version))",
        "INSERT INTO plan_versions (plan_id, version, saved_at, note, data) "
        "SELECT id, version, json_extract(data, '$.updated_at'), '', data FROM plans",
    ],
    [
        # Captions somebody wrote or corrected, kept against the file they belong to rather
        # than the project they were made in, so the next cut of the same footage starts from
        # them instead of from the transcript again. And the words the transcriber gets wrong
        # every time, fixed once for every file.
        "CREATE TABLE reviewed_captions (asset_id TEXT PRIMARY KEY, data TEXT NOT NULL)",
        "CREATE TABLE glossary (heard TEXT PRIMARY KEY, meant TEXT NOT NULL)",
    ],
    [
        # Folders the user sorts the library and the projects into. They exist only here:
        # the files on disk never move, so no project loses a file to a tidy-up. One table
        # of folders for both kinds, and which folder each thing is in, apart from the
        # things themselves, so filing a project away does not count as editing it.
        # IF NOT EXISTS here and below: 0.1.0 and 0.1.1 wrote their own, lower schema version
        # over a newer database they opened, so a later start may find these tables already there.
        "CREATE TABLE IF NOT EXISTS folders (id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent_id TEXT, name TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS folder_items (kind TEXT NOT NULL, item_id TEXT NOT NULL, folder_id TEXT NOT NULL, "
        "PRIMARY KEY (kind, item_id))",
    ],
    [
        # What each clip is about, as the local search model put it, so a search by meaning
        # embeds only the clips that are new or whose words changed since the last one.
        # `digest` covers the model and the text the vector was made from.
        "CREATE TABLE IF NOT EXISTS clip_vectors (clip_id TEXT PRIMARY KEY, timeline_id TEXT NOT NULL, "
        "digest TEXT NOT NULL, vector BLOB NOT NULL)",
        "CREATE INDEX IF NOT EXISTS clip_vectors_timeline ON clip_vectors (timeline_id)",
    ],
]
FOLDER_KINDS = ("assets", "projects")
SCHEMA_VERSION = len(_MIGRATIONS)

class VersionConflictError(ValueError):
    """Raised when a project was modified since the caller last read it."""

class FolderNotFoundError(LookupError):
    """Raised when a folder named by its ID does not exist, or holds the other kind of thing."""

class Repository:
    """SQLite-backed store for assets, analyses, projects, and background jobs.

    Records are stored as JSON documents alongside the columns needed for
    lookups and locking. The database may be shared by several processes,
    such as multiple server sessions and job workers: projects use
    optimistic locking on their `version`, and job updates run in immediate
    transactions so concurrent writers cannot overwrite each other's changes.
    """

    def __init__(self, db_path: str):
        """Open the database, creating the file and applying pending schema migrations.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Migrate inside one immediate transaction so concurrent processes never apply a migration twice.
        with self._transaction() as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            for statements in _MIGRATIONS[current:]:
                for statement in statements:
                    conn.execute(statement)
            # Never lowered: an older copy opening a database a newer one has upgraded leaves the
            # version alone, so the newer copy does not apply its migrations a second time.
            if current < SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside an immediate write transaction.

        Yields:
            The connection to execute statements on. The transaction commits
            when the block exits normally and rolls back if it raises.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: Iterable = ()) -> List[tuple]:
        """Run a read-only query.

        Args:
            sql: SQL statement to execute.
            params: Values bound to the statement's placeholders.

        Returns:
            All result rows.
        """
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def save_asset(self, asset: Asset) -> None:
        """Insert an asset, or replace the stored asset with the same ID.

        Args:
            asset: Asset to store.
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO assets (id, path, data) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET path = excluded.path, data = excluded.data",
                (asset.id, asset.path, asset.model_dump_json()),
            )

    def find_asset_by_path(self, path: str) -> Optional[Asset]:
        """Look up the asset registered for a file path.

        Args:
            path: Absolute path of the media file.

        Returns:
            The matching asset, or `None` if the path is not registered.
        """
        rows = self._query("SELECT data FROM assets WHERE path = ?", (path,))
        return Asset.model_validate_json(rows[0][0]) if rows else None

    def get_assets(self, asset_ids: Iterable[str]) -> Dict[str, Asset]:
        """Fetch several assets by ID.

        Args:
            asset_ids: IDs of the assets to fetch. Unknown IDs are skipped.

        Returns:
            The found assets, keyed by asset ID.
        """
        ids = sorted(set(asset_ids))
        if not ids:
            return {}
        rows = self._query(f"SELECT data FROM assets WHERE id IN ({','.join('?' * len(ids))})", ids)
        return {asset.id: asset for asset in (Asset.model_validate_json(row[0]) for row in rows)}

    def list_assets(self) -> List[Asset]:
        """Return all registered assets, ordered by path.

        Returns:
            The stored assets.
        """
        return [Asset.model_validate_json(row[0]) for row in self._query("SELECT data FROM assets ORDER BY path")]

    def save_analysis(self, analysis: MediaAnalysis) -> None:
        """Store an asset's analysis, replacing any previous one.

        Args:
            analysis: Analysis to store.
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO analyses (asset_id, data) VALUES (?, ?) "
                "ON CONFLICT(asset_id) DO UPDATE SET data = excluded.data",
                (analysis.asset_id, analysis.model_dump_json()),
            )

    def get_analysis(self, asset_id: str) -> Optional[MediaAnalysis]:
        """Fetch the analysis of an asset.

        Args:
            asset_id: ID of the analyzed asset.

        Returns:
            The analysis, or `None` if the asset has not been analyzed.
        """
        rows = self._query("SELECT data FROM analyses WHERE asset_id = ?", (asset_id,))
        return MediaAnalysis.model_validate_json(rows[0][0]) if rows else None

    def remember_captions(self, cues: Iterable[SubtitleCue]) -> None:
        """Keep captions somebody wrote or corrected against the files they belong to.

        A caption replaces any kept before it over the same stretch of the same
        file, so the latest correction is the one remembered.

        Args:
            cues: The captions, each anchored to its file.
        """
        by_asset: Dict[str, List[SubtitleCue]] = {}
        for cue in cues:
            by_asset.setdefault(cue.asset_id, []).append(cue)
        with self._transaction() as conn:
            for asset_id, fresh in by_asset.items():
                rows = conn.execute("SELECT data FROM reviewed_captions WHERE asset_id = ?", (asset_id,)).fetchall()
                kept = [SubtitleCue.model_validate(item) for item in json.loads(rows[0][0])] if rows else []
                kept = [
                    old for old in kept
                    if not any(old.source_start < new.source_end and new.source_start < old.source_end for new in fresh)
                ]
                merged = sorted([*kept, *fresh], key=lambda cue: cue.source_start)
                conn.execute(
                    "INSERT INTO reviewed_captions (asset_id, data) VALUES (?, ?) "
                    "ON CONFLICT(asset_id) DO UPDATE SET data = excluded.data",
                    (asset_id, json.dumps([cue.model_dump(mode="json") for cue in merged], ensure_ascii=False)),
                )

    def reviewed_captions(self, asset_id: str) -> List[SubtitleCue]:
        """Read the captions kept for a file.

        Args:
            asset_id: The file.

        Returns:
            Its reviewed captions in source order; empty when there are none.
        """
        rows = self._query("SELECT data FROM reviewed_captions WHERE asset_id = ?", (asset_id,))
        return [SubtitleCue.model_validate(item) for item in json.loads(rows[0][0])] if rows else []

    def update_glossary(self, fixes: Mapping[str, str]) -> Dict[str, str]:
        """Add to or take from the words every transcript is corrected with.

        Args:
            fixes: What the transcriber writes, mapped to what was meant. An
                empty meaning takes the entry away.

        Returns:
            The whole glossary as it now stands.
        """
        with self._transaction() as conn:
            for heard, meant in fixes.items():
                if meant:
                    conn.execute(
                        "INSERT INTO glossary (heard, meant) VALUES (?, ?) "
                        "ON CONFLICT(heard) DO UPDATE SET meant = excluded.meant",
                        (heard, meant),
                    )
                else:
                    conn.execute("DELETE FROM glossary WHERE heard = ?", (heard,))
        return self.glossary()

    def glossary(self) -> Dict[str, str]:
        """Read the words every transcript is corrected with.

        Returns:
            What the transcriber writes, mapped to what was meant.
        """
        return dict(self._query("SELECT heard, meant FROM glossary ORDER BY heard"))

    def add_project(self, project: Project) -> None:
        """Store a new project.

        Args:
            project: Project to store.

        Raises:
            sqlite3.IntegrityError: If a project with the same ID already exists.
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO projects (id, version, data) VALUES (?, ?, ?)",
                (project.id, project.version, project.model_dump_json()),
            )

    def get_project(self, project_id: str) -> Optional[Project]:
        """Fetch a project by ID.

        Args:
            project_id: ID of the project to fetch.

        Returns:
            The project, or `None` if it does not exist.
        """
        rows = self._query("SELECT data FROM projects WHERE id = ?", (project_id,))
        return Project.model_validate_json(rows[0][0]) if rows else None

    def list_projects(self) -> List[Project]:
        """Return all projects.

        Returns:
            The stored projects.
        """
        return [Project.model_validate_json(row[0]) for row in self._query("SELECT data FROM projects ORDER BY rowid")]

    def update_project(self, project: Project, expected_version: int) -> None:
        """Replace a stored project if it is still at the expected version.

        Args:
            project: New project state, including its new `version`.
            expected_version: Version the stored project must currently have.

        Raises:
            VersionConflictError: If the stored project is missing or its
                version differs from `expected_version`.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE projects SET version = ?, data = ? WHERE id = ? AND version = ?",
                (project.version, project.model_dump_json(), project.id, expected_version),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError(f"version conflict: project {project.id} is no longer at version {expected_version}")

    def add_job(self, job: Job) -> None:
        """Store a new background job.

        Args:
            job: Job to store.
        """
        job.updated_at = datetime.now(timezone.utc)
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (id, data) VALUES (?, ?)",
                (job.job_id, job.model_dump_json()),
            )

    def get_job(self, job_id: str) -> Optional[Job]:
        """Fetch a background job by ID.

        Args:
            job_id: ID of the job to fetch.

        Returns:
            The job, or `None` if it does not exist.
        """
        rows = self._query("SELECT data FROM jobs WHERE id = ?", (job_id,))
        return Job.model_validate_json(rows[0][0]) if rows else None

    def active_job_ids(self) -> Set[str]:
        """Say which jobs are still queued or running, so nothing clears their files.

        Returns:
            Their IDs.
        """
        rows = self._query("SELECT id FROM jobs WHERE json_extract(data, '$.status') IN ('queued', 'running')")
        return {row[0] for row in rows}

    def update_active_jobs(self, change: Callable[[List[Job]], Iterable[str]]) -> List[Job]:
        """Atomically read every unfinished job, decide across them, and write back the ones chosen.

        This is how a limit that spans jobs is enforced — how many may run at
        once, and how much memory they may take together. Reading the whole
        set, deciding, and writing the decision happen in one immediate
        transaction, so two servers or two workers looking at the same
        workspace cannot both conclude that there is room for one more.

        Args:
            change: Function called with the queued and running jobs, ordered
                oldest first. It may modify them in place and returns the IDs
                of the ones whose changes should be saved. It runs inside the
                transaction and should not block.

        Returns:
            The jobs that were written, in the order `change` named them.
        """
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT data FROM jobs WHERE json_extract(data, '$.status') IN ('queued', 'running')"
            ).fetchall()
            jobs = sorted((Job.model_validate_json(row[0]) for row in rows), key=lambda job: job.created_at)
            by_id = {job.job_id: job for job in jobs}
            written = []
            for job_id in change(jobs):
                job = by_id[job_id]
                job.updated_at = datetime.now(timezone.utc)
                conn.execute("UPDATE jobs SET data = ? WHERE id = ?", (job.model_dump_json(), job.job_id))
                written.append(job)
            return written

    def update_job(self, job_id: str, change: Callable[[Job], None]) -> Optional[Job]:
        """Atomically read a job, apply a change to it, and write it back.

        The read and write happen in one immediate transaction, so a change
        made by another process in the meantime is never lost. `updated_at`
        is refreshed on every update.

        Args:
            job_id: ID of the job to update.
            change: Function that mutates the job in place. It runs inside the
                transaction and should not block.

        Returns:
            The updated job, or `None` if it does not exist.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = Job.model_validate_json(row[0])
            change(job)
            job.updated_at = datetime.now(timezone.utc)
            conn.execute("UPDATE jobs SET data = ? WHERE id = ?", (job.model_dump_json(), job_id))
            return job

    def save_semantic_timeline(self, timeline: SemanticTimeline, clips: Iterable[SemanticClip]) -> None:
        """Store a semantic timeline and its clips, replacing any earlier build of it.

        Args:
            timeline: Timeline to store.
            clips: Its clips. Everything previously stored under the same
                timeline is removed first, so a rebuild leaves nothing behind.
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO semantic_timelines (id, input_hash, data) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET input_hash = excluded.input_hash, data = excluded.data",
                (timeline.id, timeline.input_hash, timeline.model_dump_json()),
            )
            conn.execute("DELETE FROM semantic_clips WHERE timeline_id = ?", (timeline.id,))
            conn.executemany(
                "INSERT INTO semantic_clips "
                "(id, timeline_id, asset_id, level, kind, start_seconds, end_seconds, text, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        clip.id, clip.timeline_id, clip.asset_id, clip.level.value, clip.kind.value,
                        clip.source_range.start, clip.source_range.end, clip.text, clip.model_dump_json(),
                    )
                    for clip in clips
                ],
            )
            # Vectors of clips this build no longer has; the rest are checked by digest when read.
            conn.execute(
                "DELETE FROM clip_vectors WHERE timeline_id = ? "
                "AND clip_id NOT IN (SELECT id FROM semantic_clips WHERE timeline_id = ?)",
                (timeline.id, timeline.id),
            )

    def clip_vectors(self, clip_ids: Iterable[str]) -> Dict[str, tuple]:
        """Read stored search vectors.

        Args:
            clip_ids: Clips to read them for.

        Returns:
            `(digest, vector bytes)` by clip ID, for the clips that have one.
        """
        ids = list(clip_ids)
        found: Dict[str, tuple] = {}
        # SQLite caps the number of bound values in one statement.
        for first in range(0, len(ids), 500):
            chunk = ids[first:first + 500]
            rows = self._query(
                f"SELECT clip_id, digest, vector FROM clip_vectors WHERE clip_id IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            found.update({row[0]: (row[1], row[2]) for row in rows})
        return found

    def save_clip_vectors(self, rows: Iterable[tuple]) -> None:
        """Store search vectors.

        Args:
            rows: `(clip_id, timeline_id, digest, vector bytes)` each.
        """
        with self._transaction() as conn:
            conn.executemany(
                "INSERT INTO clip_vectors (clip_id, timeline_id, digest, vector) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(clip_id) DO UPDATE SET timeline_id = excluded.timeline_id, "
                "digest = excluded.digest, vector = excluded.vector",
                list(rows),
            )

    def update_semantic_clips(self, clips: Iterable[SemanticClip]) -> None:
        """Write changes to clips that already exist, in one transaction.

        Unlike storing a timeline, this leaves every other clip alone, so
        writing a description onto a handful of them does not rebuild the set.

        Args:
            clips: The changed clips. A clip that is no longer stored is
                skipped rather than inserted, because the timeline it belonged
                to has been rebuilt and it no longer means anything.
        """
        with self._transaction() as conn:
            conn.executemany(
                "UPDATE semantic_clips SET kind = ?, text = ?, data = ? WHERE id = ?",
                [(clip.kind.value, clip.text, clip.model_dump_json(), clip.id) for clip in clips],
            )

    def find_semantic_timeline(self, input_hash: str) -> Optional[SemanticTimeline]:
        """Look up the timeline built from a given input.

        Args:
            input_hash: Fingerprint of the analyses and the derivation.

        Returns:
            The matching timeline, or `None` if that input has not been built.
        """
        rows = self._query("SELECT data FROM semantic_timelines WHERE input_hash = ?", (input_hash,))
        return SemanticTimeline.model_validate_json(rows[0][0]) if rows else None

    def get_semantic_timeline(self, timeline_id: Optional[str]) -> Optional[SemanticTimeline]:
        """Fetch a semantic timeline, or the one built most recently.

        Args:
            timeline_id: ID of the timeline, or `None` for the newest build.

        Returns:
            The timeline, or `None` if it does not exist, or if none have been
            built.
        """
        if timeline_id is None:
            rows = self._query("SELECT data FROM semantic_timelines ORDER BY rowid DESC LIMIT 1")
        else:
            rows = self._query("SELECT data FROM semantic_timelines WHERE id = ?", (timeline_id,))
        return SemanticTimeline.model_validate_json(rows[0][0]) if rows else None

    def get_semantic_clip(self, clip_id: str) -> Optional[SemanticClip]:
        """Fetch one semantic clip.

        Args:
            clip_id: ID of the clip.

        Returns:
            The clip, or `None` if no clip has that ID.
        """
        rows = self._query("SELECT data FROM semantic_clips WHERE id = ?", (clip_id,))
        return SemanticClip.model_validate_json(rows[0][0]) if rows else None

    def query_semantic_clips(
        self,
        timeline_id: str,
        level: Optional[ClipLevel] = None,
        kinds: Optional[Iterable[ClipKind]] = None,
        asset_ids: Optional[Iterable[str]] = None,
        topic: Optional[str] = None,
        tag: Optional[str] = None,
        text: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_scores: Optional[Dict[str, float]] = None,
        max_scores: Optional[Dict[str, float]] = None,
        limit: int = 50,
    ) -> List[SemanticClip]:
        """Select the clips of a timeline that match a set of conditions.

        A condition left as `None` is not applied; the ones given are combined
        with and.

        Args:
            timeline_id: Timeline to search.
            level: Keep only clips at this level.
            kinds: Keep only clips of these kinds.
            asset_ids: Keep only clips from these assets.
            topic: Keep only sections labelled with this subject.
            tag: Keep only clips carrying this tag, whoever wrote it.
            text: Keep only clips whose text or description contains this.
            start: Keep only clips reaching past this second of their source.
            end: Keep only clips beginning before this second of their source.
            min_duration: Keep only clips at least this many seconds long.
            max_duration: Keep only clips at most this many seconds long.
            min_scores: Keep only clips whose named scores are at least these
                values. A clip that does not carry one of the named scores is
                dropped.
            max_scores: Keep only clips whose named scores are at most these
                values, on the same terms.
            limit: Largest number of clips to return.

        Returns:
            The matching clips, ordered by asset and then by time.

        Raises:
            ValueError: If a score name is not a plain identifier.
        """
        # Both are taken as the enum or as its plain string, so a caller that has one
        # from a tool schema and one from a literal gets the same answer either way.
        where, params = ["timeline_id = ?"], [timeline_id]
        if level is not None:
            where.append("level = ?")
            params.append(ClipLevel(level).value)
        kinds = list(kinds or [])
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params.extend(ClipKind(kind).value for kind in kinds)
        sources = list(asset_ids or [])
        if sources:
            where.append(f"asset_id IN ({','.join('?' * len(sources))})")
            params.extend(sources)
        if topic is not None:
            where.append("json_extract(data, '$.topic') = ?")
            params.append(topic)
        if tag:
            where.append("EXISTS (SELECT 1 FROM json_each(data, '$.tags') WHERE json_extract(value, '$.value') = ?)")
            params.append(tag)
        if text:
            # The wildcards belong to LIKE, so a search for a literal % or _ escapes it. The
            # description is searched too: what is on screen is as much a reason to pick a clip
            # as what is said, and a clip with no words has nothing else to be found by.
            escaped = re.sub(r"([%_\\])", r"\\\1", text)
            where.append(
                "(text LIKE '%' || ? || '%' ESCAPE '\\' "
                "OR coalesce(json_extract(data, '$.description'), '') LIKE '%' || ? || '%' ESCAPE '\\')"
            )
            params.extend([escaped, escaped])
        if start is not None:
            where.append("end_seconds > ?")
            params.append(start)
        if end is not None:
            where.append("start_seconds < ?")
            params.append(end)
        if min_duration is not None:
            where.append("end_seconds - start_seconds >= ?")
            params.append(min_duration)
        if max_duration is not None:
            where.append("end_seconds - start_seconds <= ?")
            params.append(max_duration)
        # A score name reaches the JSON path itself rather than a placeholder,
        # so only a plain identifier is let through.
        for bounds, comparison in ((min_scores, ">="), (max_scores, "<=")):
            for name, value in (bounds or {}).items():
                if not re.fullmatch(r"[a-z][a-z_]*", name):
                    raise ValueError(f"{name!r} is not a score name")
                where.append(f"json_extract(data, '$.scores.{name}') {comparison} ?")
                params.append(value)

        rows = self._query(
            f"SELECT data FROM semantic_clips WHERE {' AND '.join(where)} "
            "ORDER BY asset_id, start_seconds LIMIT ?",
            [*params, limit],
        )
        return [SemanticClip.model_validate_json(row[0]) for row in rows]

    def save_plan(self, plan: EditPlan, note: str = "") -> EditPlan:
        """Store an edit plan, raising its version if it already exists.

        A plan that is not stored yet is inserted as it stands. One that is
        must carry the version it was last read at, so two sessions working
        from the same plan cannot overwrite one another in silence.

        Args:
            plan: Plan to store.
            note: What this version is, for its history: the feedback that
                led to it, or what was changed.

        Returns:
            The stored plan, at its new version. The version is also kept in
            the plan's history.

        Raises:
            VersionConflictError: If the stored plan is at a different
                version.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT version FROM plans WHERE id = ?", (plan.id,)).fetchone()
            if row is None:
                stored = plan.model_copy(update={"version": 1, "updated_at": datetime.now(timezone.utc)})
                conn.execute(
                    "INSERT INTO plans (id, timeline_id, version, data) VALUES (?, ?, ?, ?)",
                    (stored.id, stored.timeline_id, stored.version, stored.model_dump_json()),
                )
                self._keep_version(conn, stored, note)
                return stored
            if row[0] != plan.version:
                raise VersionConflictError(
                    f"version conflict: plan {plan.id} is at version {row[0]}, not {plan.version}"
                )
            stored = plan.model_copy(update={"version": plan.version + 1, "updated_at": datetime.now(timezone.utc)})
            conn.execute(
                "UPDATE plans SET timeline_id = ?, version = ?, data = ? WHERE id = ?",
                (stored.timeline_id, stored.version, stored.model_dump_json(), stored.id),
            )
            self._keep_version(conn, stored, note)
            return stored

    @staticmethod
    def _keep_version(conn: sqlite3.Connection, plan: EditPlan, note: str) -> None:
        """Add one version of a plan to its history, inside the transaction that saved it.

        Args:
            conn: The open transaction.
            plan: The plan as stored.
            note: What this version is.
        """
        conn.execute(
            "INSERT OR REPLACE INTO plan_versions (plan_id, version, saved_at, note, data) VALUES (?, ?, ?, ?, ?)",
            (plan.id, plan.version, plan.updated_at.isoformat(), note, plan.model_dump_json()),
        )

    def get_plan_version(self, plan_id: str, version: int) -> Optional[EditPlan]:
        """Fetch one version of a plan from its history.

        Args:
            plan_id: ID of the plan.
            version: Which version.

        Returns:
            The plan as it was at that version, or None if there is no such
            version.
        """
        rows = self._query("SELECT data FROM plan_versions WHERE plan_id = ? AND version = ?", (plan_id, version))
        return EditPlan.model_validate_json(rows[0][0]) if rows else None

    def list_plan_versions(self, plan_id: str) -> List[Dict[str, object]]:
        """List a plan's history, oldest first.

        Args:
            plan_id: ID of the plan.

        Returns:
            One dictionary per version, with its `version`, `saved_at` and
            `note`.
        """
        return [
            {"version": row[0], "saved_at": row[1], "note": row[2]}
            for row in self._query(
                "SELECT version, saved_at, note FROM plan_versions WHERE plan_id = ? ORDER BY version", (plan_id,),
            )
        ]

    def get_plan(self, plan_id: Optional[str]) -> Optional[EditPlan]:
        """Fetch an edit plan, or the one saved most recently.

        Args:
            plan_id: ID of the plan, or `None` for the newest.

        Returns:
            The plan, or `None` if it does not exist or none are stored.
        """
        if plan_id is None:
            rows = self._query("SELECT data FROM plans ORDER BY rowid DESC LIMIT 1")
        else:
            rows = self._query("SELECT data FROM plans WHERE id = ?", (plan_id,))
        return EditPlan.model_validate_json(rows[0][0]) if rows else None

    def list_plans(self) -> List[EditPlan]:
        """Return every stored edit plan, newest first.

        Returns:
            The plans.
        """
        return [EditPlan.model_validate_json(row[0]) for row in self._query("SELECT data FROM plans ORDER BY rowid DESC")]

    def count_semantic_clips(self, timeline_id: str) -> Dict[str, int]:
        """Count a timeline's clips by level and kind.

        Args:
            timeline_id: Timeline to count.

        Returns:
            Counts keyed by `"<level>/<kind>"`.
        """
        rows = self._query(
            "SELECT level, kind, count(*) FROM semantic_clips WHERE timeline_id = ? GROUP BY level, kind",
            (timeline_id,),
        )
        return {f"{level}/{kind}": count for level, kind, count in rows}

    # ------------------------------------------------------------------ folders

    def list_folders(self, kind: str) -> List[Dict[str, Optional[str]]]:
        """Return the folders of one kind, by name.

        Args:
            kind: `assets` or `projects`.

        Returns:
            Each folder's `id`, `parent_id` (None at the top) and `name`.
        """
        rows = self._query("SELECT id, parent_id, name FROM folders WHERE kind = ? ORDER BY name COLLATE NOCASE", (kind,))
        return [{"id": row[0], "parent_id": row[1], "name": row[2]} for row in rows]

    @staticmethod
    def _folder_kind(conn: sqlite3.Connection, folder_id: str) -> Optional[str]:
        """The kind of a folder, or None when there is no such folder."""
        row = conn.execute("SELECT kind FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _check_name(conn: sqlite3.Connection, kind: str, parent_id: Optional[str], name: str, folder_id: str = "") -> str:
        """A folder name tidied, refused when empty, too long, or already taken beside it."""
        name = " ".join(name.split())
        if not name:
            raise ValueError("a folder needs a name")
        if len(name) > 80:
            raise ValueError("a folder name is at most 80 characters")
        taken = conn.execute(
            "SELECT 1 FROM folders WHERE kind = ? AND parent_id IS ? AND name = ? COLLATE NOCASE AND id != ?",
            (kind, parent_id, name, folder_id),
        ).fetchone()
        if taken:
            raise ValueError(f"there is already a folder called {name} here")
        return name

    def create_folder(self, folder_id: str, kind: str, name: str, parent_id: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Make a folder.

        Args:
            folder_id: Its new ID.
            kind: `assets` or `projects`.
            name: What the user called it.
            parent_id: The folder it goes in; the top when None.

        Returns:
            The folder.

        Raises:
            FolderNotFoundError: If the parent is missing or holds the other kind.
            ValueError: For an unknown kind, or a name that is empty or already taken beside it.
        """
        if kind not in FOLDER_KINDS:
            raise ValueError(f"folders hold {' or '.join(FOLDER_KINDS)}")
        with self._transaction() as conn:
            if parent_id is not None and self._folder_kind(conn, parent_id) != kind:
                raise FolderNotFoundError(f"folder {parent_id} not found")
            name = self._check_name(conn, kind, parent_id, name)
            conn.execute("INSERT INTO folders (id, kind, parent_id, name) VALUES (?, ?, ?, ?)",
                         (folder_id, kind, parent_id, name))
        return {"id": folder_id, "parent_id": parent_id, "name": name}

    def rename_folder(self, folder_id: str, name: str) -> None:
        """Rename a folder.

        Args:
            folder_id: The folder.
            name: Its new name.

        Raises:
            FolderNotFoundError: If there is no such folder.
            ValueError: If the name is empty, too long, or taken beside it.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT kind, parent_id FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if not row:
                raise FolderNotFoundError(f"folder {folder_id} not found")
            conn.execute("UPDATE folders SET name = ? WHERE id = ?",
                         (self._check_name(conn, row[0], row[1], name, folder_id), folder_id))

    def delete_folder(self, folder_id: str) -> None:
        """Remove a folder; what was in it moves up into the folder around it.

        Nothing in it is deleted, so removing a folder can never cost a file or a project.

        Args:
            folder_id: The folder.

        Raises:
            FolderNotFoundError: If there is no such folder.
            ValueError: If a folder moving up has the name of one already there.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT kind, parent_id FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if not row:
                raise FolderNotFoundError(f"folder {folder_id} not found")
            kind, parent_id = row
            for child_id, child_name in conn.execute(
                "SELECT id, name FROM folders WHERE parent_id = ?", (folder_id,)
            ).fetchall():
                self._check_name(conn, kind, parent_id, child_name, child_id)
            conn.execute("UPDATE folders SET parent_id = ? WHERE parent_id = ?", (parent_id, folder_id))
            if parent_id is None:
                conn.execute("DELETE FROM folder_items WHERE folder_id = ?", (folder_id,))
            else:
                conn.execute("UPDATE folder_items SET folder_id = ? WHERE folder_id = ?", (parent_id, folder_id))
            conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))

    def move_folder(self, folder_id: str, parent_id: Optional[str]) -> None:
        """Put a folder inside another, or at the top.

        Args:
            folder_id: The folder to move.
            parent_id: The folder to put it in; the top when None.

        Raises:
            FolderNotFoundError: If either folder is unknown, or they hold different kinds.
            ValueError: If the folder would end up inside itself, or its name is taken there.
        """
        with self._transaction() as conn:
            row = conn.execute("SELECT kind, name FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if not row:
                raise FolderNotFoundError(f"folder {folder_id} not found")
            kind, name = row
            ancestor = parent_id
            while ancestor is not None:
                if ancestor == folder_id:
                    raise ValueError("a folder cannot go inside itself")
                found = conn.execute("SELECT kind, parent_id FROM folders WHERE id = ?", (ancestor,)).fetchone()
                if not found or found[0] != kind:
                    raise FolderNotFoundError(f"folder {ancestor} not found")
                ancestor = found[1]
            self._check_name(conn, kind, parent_id, name, folder_id)
            conn.execute("UPDATE folders SET parent_id = ? WHERE id = ?", (parent_id, folder_id))

    def folder_of(self, kind: str) -> Dict[str, str]:
        """Say which folder each filed item of one kind is in.

        Args:
            kind: `assets` or `projects`.

        Returns:
            Folder ID by item ID; an item not listed is at the top.
        """
        return dict(self._query("SELECT item_id, folder_id FROM folder_items WHERE kind = ?", (kind,)))

    def file_items(self, kind: str, item_ids: Iterable[str], folder_id: Optional[str]) -> None:
        """Put assets or projects in a folder, or back at the top.

        Args:
            kind: `assets` or `projects`.
            item_ids: The assets or projects.
            folder_id: The folder; the top when None.

        Raises:
            ValueError: For an unknown kind.
            FolderNotFoundError: If the folder is unknown or holds the other kind.
        """
        if kind not in FOLDER_KINDS:
            raise ValueError(f"folders hold {' or '.join(FOLDER_KINDS)}")
        with self._transaction() as conn:
            if folder_id is not None and self._folder_kind(conn, folder_id) != kind:
                raise FolderNotFoundError(f"folder {folder_id} not found")
            for item_id in item_ids:
                if folder_id is None:
                    conn.execute("DELETE FROM folder_items WHERE kind = ? AND item_id = ?", (kind, item_id))
                else:
                    conn.execute(
                        "INSERT INTO folder_items (kind, item_id, folder_id) VALUES (?, ?, ?) "
                        "ON CONFLICT(kind, item_id) DO UPDATE SET folder_id = excluded.folder_id",
                        (kind, item_id, folder_id),
                    )

    # ------------------------------------------------------------------ taking a file out of the library

    def delete_unused_assets(self, asset_ids: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
        """Take files out of the library, unless anything still uses one of them.

        A file is in use while a project has a clip of it, the AI's current
        plan for a cut uses it (a selection, a cutaway or music), or an
        analysis of it is running. Looking and deleting happen in one write
        transaction, so nothing can start using a file between the two.

        What goes with a file: its analysis, its corrected captions, its clips
        and their search vectors, and its place in a folder. A timeline that
        covered other files too keeps them; one left with no file, and no
        plan, goes. The file on disk is not touched here.

        Args:
            asset_ids: The assets.

        Returns:
            What uses each file that is in use, as `{"projects": [names],
            "plans": [goals], "busy": ["analysis"]}` by asset ID. When this is
            not empty nothing was deleted.
        """
        wanted = set(asset_ids)
        with self._transaction() as conn:
            using: Dict[str, Dict[str, List[str]]] = {}

            def note(asset_id: str, kind: str, what: str) -> None:
                found = using.setdefault(asset_id, {"projects": [], "plans": [], "busy": []})[kind]
                if what not in found:
                    found.append(what)

            for (data,) in conn.execute("SELECT data FROM projects").fetchall():
                project = Project.model_validate_json(data)
                for asset_id in wanted & {clip.asset_id for track in project.tracks for clip in track.clips}:
                    note(asset_id, "projects", project.name or project.id)
            for (data,) in conn.execute("SELECT data FROM plans").fetchall():
                plan = json.loads(data)
                assets, clips = set(), set()
                _plan_references(plan, assets, clips)
                for first in range(0, len(clips), 500):
                    chunk = sorted(clips)[first:first + 500]
                    assets.update(row[0] for row in conn.execute(
                        f"SELECT asset_id FROM semantic_clips WHERE id IN ({','.join('?' * len(chunk))})", chunk))
                for asset_id in wanted & assets:
                    note(asset_id, "plans", plan.get("goal") or plan.get("id", ""))
            for (data,) in conn.execute("SELECT data FROM jobs").fetchall():
                job = Job.model_validate_json(data)
                if job.asset_id in wanted and job.status.is_active:
                    note(job.asset_id, "busy", "analysis")
            if using:
                return using
            for asset_id in wanted:
                self._delete_asset(conn, asset_id)
        return {}

    @staticmethod
    def _delete_asset(conn: sqlite3.Connection, asset_id: str) -> None:
        """Delete one asset and everything worked out from it, inside the caller's transaction."""
        conn.execute("DELETE FROM clip_vectors WHERE clip_id IN (SELECT id FROM semantic_clips WHERE asset_id = ?)",
                     (asset_id,))
        for statement in ("DELETE FROM assets WHERE id = ?", "DELETE FROM analyses WHERE asset_id = ?",
                          "DELETE FROM reviewed_captions WHERE asset_id = ?",
                          "DELETE FROM semantic_clips WHERE asset_id = ?"):
            conn.execute(statement, (asset_id,))
        conn.execute("DELETE FROM folder_items WHERE kind = 'assets' AND item_id = ?", (asset_id,))
        planned = {row[0] for row in conn.execute("SELECT timeline_id FROM plans")}
        for timeline_id, data in conn.execute("SELECT id, data FROM semantic_timelines").fetchall():
            timeline = SemanticTimeline.model_validate_json(data)
            if asset_id not in timeline.asset_ids:
                continue
            timeline.asset_ids = [kept for kept in timeline.asset_ids if kept != asset_id]
            if not timeline.asset_ids and timeline_id not in planned:
                conn.execute("DELETE FROM clip_vectors WHERE timeline_id = ?", (timeline_id,))
                conn.execute("DELETE FROM semantic_clips WHERE timeline_id = ?", (timeline_id,))
                conn.execute("DELETE FROM semantic_timelines WHERE id = ?", (timeline_id,))
            else:
                conn.execute("UPDATE semantic_timelines SET data = ? WHERE id = ?",
                             (timeline.model_dump_json(), timeline_id))

def _plan_references(value: object, assets: Set[str], clips: Set[str]) -> None:
    """Collect the asset IDs and semantic clip IDs anywhere in a plan's JSON.

    Read from the JSON rather than the model so that every place a plan names
    footage counts — selections, cutaways, music — including ones added later.

    Args:
        value: The plan, or a part of it.
        assets: Receives `asset_id` values.
        clips: Receives `clip_id`, `over_clip_id` and `keep_clip_ids` values.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "asset_id" and isinstance(item, str):
                assets.add(item)
            elif key in ("clip_id", "over_clip_id") and isinstance(item, str):
                clips.add(item)
            elif key == "keep_clip_ids" and isinstance(item, list):
                clips.update(entry for entry in item if isinstance(entry, str))
            else:
                _plan_references(item, assets, clips)
    elif isinstance(value, list):
        for item in value:
            _plan_references(item, assets, clips)
