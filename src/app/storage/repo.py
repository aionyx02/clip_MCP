import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from app.models.job import Job
from app.models.media import Asset, MediaAnalysis
from app.models.plan import EditPlan
from app.models.semantic import ClipKind, ClipLevel, SemanticClip, SemanticTimeline
from app.models.timeline import Project

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
]
SCHEMA_VERSION = len(_MIGRATIONS)

class VersionConflictError(ValueError):
    """Raised when a project was modified since the caller last read it."""

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
        asset_id: Optional[str] = None,
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
            asset_id: Keep only clips from this asset.
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
        if asset_id is not None:
            where.append("asset_id = ?")
            params.append(asset_id)
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

    def save_plan(self, plan: EditPlan) -> EditPlan:
        """Store an edit plan, raising its version if it already exists.

        A plan that is not stored yet is inserted as it stands. One that is
        must carry the version it was last read at, so two sessions working
        from the same plan cannot overwrite one another in silence.

        Args:
            plan: Plan to store.

        Returns:
            The stored plan, at its new version.

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
            return stored

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
