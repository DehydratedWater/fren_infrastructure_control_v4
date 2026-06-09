"""Ralf housekeeping — delete old rendered media + expired locks.

v3 parity port of ``scripts/ralf_cleanup.py`` (the cron entrypoint at
``scripts/ralf_cleanup.py`` is a thin wrapper around :func:`run`). Policies,
preserved from v3:

  - rendered media: delete files older than ``keep_days`` (v3 default: 7)
    UNLESS the media_id is referenced as a ralf "winner" (`ralf_kv` keys
    matching ``winner_*``/``final_*``/``best_*``);
  - expired locks: delete ``ralf_locks`` rows where ``expires_at < NOW()``.

SAFETY (the v4 hardening): a file is only ever unlinked when its RESOLVED path
is contained inside one of the configured media roots (the persistent
``<data_dir>/rendered`` volume dir or the repo-local ``data/rendered``) — the
same containment discipline as ``app.web.data.safe_media_path``. A DB row
whose file_path escapes the roots is left untouched and reported, never
deleted.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/app/tools/system/ralf_cleanup.py → repo root is parents[4].
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_KEEP_DAYS = 7


def allowed_media_roots() -> list[Path]:
    """The only directories the cleanup may ever delete files from."""
    roots = [PROJECT_ROOT / "data" / "rendered"]
    try:
        from app.settings import get_settings

        roots.append(Path(get_settings().data_dir) / "rendered")
    except Exception:  # noqa: BLE001 — settings unavailable in some test contexts
        pass
    return [r.resolve() for r in roots]


def resolve_media_path(file_path: str) -> Path:
    """DB ``file_path`` values are absolute (under /data) or PROJECT_ROOT-relative."""
    p = Path(file_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def is_safely_contained(path: Path, roots: list[Path]) -> bool:
    """True only when the RESOLVED path lives under one of the allowed roots.

    Mirrors ``app.web.data.safe_media_path``'s belt-and-suspenders re-check:
    resolution happens first, so neither ``..`` segments nor symlinks can
    escape a root.
    """
    resolved = path.resolve()
    for root in roots:
        if resolved == root:
            return False  # a root itself is never a deletable file
        if resolved.is_relative_to(root):
            return True
    return False


async def cleanup_rendered_media(
    keep_days: int = DEFAULT_KEEP_DAYS,
    *,
    dry_run: bool = False,
    roots: list[Path] | None = None,
) -> dict[str, int]:
    """Delete rendered files older than *keep_days* unless marked as winners."""
    from app.db.repos.ralf import RalfKVRepo
    from app.db.repos.rendered_media import RenderedMediaRepo

    roots = [r.resolve() for r in roots] if roots is not None else allowed_media_roots()
    winners = await RalfKVRepo().winner_media_ids()
    cutoff = datetime.now(UTC) - timedelta(days=keep_days)

    repo = RenderedMediaRepo()
    rows = await repo.list_older_than(cutoff)

    files_removed = 0
    rows_removed = 0
    skipped_winners = 0
    containment_violations = 0
    for row in rows:
        media_id = row.get("media_id", "")
        if media_id in winners:
            skipped_winners += 1
            continue

        path = resolve_media_path(str(row.get("file_path") or ""))
        if path.exists():
            if not is_safely_contained(path, roots):
                # NEVER delete outside the configured media dirs — keep the
                # row too so the anomaly stays visible.
                containment_violations += 1
                logger.warning("[ralf_cleanup] refusing to touch %s (outside media roots)", path)
                continue
            if dry_run:
                print(f"[dry-run] would remove {path}")
            else:
                try:
                    path.unlink()
                    files_removed += 1
                except OSError as e:
                    logger.error("[ralf_cleanup] failed to remove %s: %s", path, e)
        if not dry_run:
            await repo.delete(media_id)
            rows_removed += 1

    return {
        "files_removed": files_removed,
        "rows_removed": rows_removed,
        "skipped_winners": skipped_winners,
        "containment_violations": containment_violations,
    }


async def cleanup_expired_locks(*, dry_run: bool = False) -> int:
    """Release every expired ralf lock; returns the count."""
    from app.db.repos.ralf import RalfLocksRepo

    repo = RalfLocksRepo()
    if dry_run:
        expired = await repo.list_expired()
        for r in expired:
            print(
                f"[dry-run] would release expired lock {r.get('resource_key')} "
                f"(held by {r.get('holder_ralf_id')})"
            )
        return len(expired)
    return await repo.release_expired()


async def run(keep_days: int = DEFAULT_KEEP_DAYS, *, dry_run: bool = False) -> dict[str, int]:
    """One housekeeping pass: old rendered media + expired locks."""
    media = await cleanup_rendered_media(keep_days, dry_run=dry_run)
    locks_freed = await cleanup_expired_locks(dry_run=dry_run)
    mode = "dry-run" if dry_run else "applied"
    print(
        f"[ralf_cleanup] {mode}: removed {media['files_removed']} files, "
        f"{media['rows_removed']} media rows, {locks_freed} expired locks "
        f"(keep_days={keep_days}, winners kept={media['skipped_winners']}, "
        f"containment violations={media['containment_violations']})"
    )
    return {**media, "locks_freed": locks_freed}
