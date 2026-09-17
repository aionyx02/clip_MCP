import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from app.models.job import Job
from app.models.media import Asset, MediaAnalysis
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
