"""Techtree Manager — browse git repo, track commits, manage interests, analysis runs."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime

from src import ScriptTool
from pydantic import BaseModel, Field

TECHTREE_REPO = "/home/dw/techtree/repos/techtree"


class Input(BaseModel):
    command: str = Field(
        description=(
            "Git: git-pull|git-log|git-diff|git-show|git-branches|git-file|git-prs|git-pr-diff; "
            "Commits: ingest-new|list-commits|get-commit|commit-stats|list-unnotified|mark-notified; "
            "Interests: create-interest|get-interest|list-interests|update-interest|delete-interest|toggle-interest; "
            "Analysis: create-analysis|list-analyses|get-latest-analysis; "
            "State: get-state|set-state|list-states"
        )
    )
    # IDs
    commit_sha: str = Field(default="", description="Commit SHA")
    interest_id: str = Field(default="", description="Interest ID")
    run_id: str = Field(default="", description="Analysis run ID")
    state_key: str = Field(default="", description="State key")
    # Interest fields
    name: str = Field(default="", description="Name for interest")
    description: str = Field(default="", description="Description")
    paths: str = Field(default="", description="JSON array of file paths to watch")
    keywords: str = Field(default="", description="JSON array of keywords")
    owner: str = Field(default="", description="Committer name filter (empty=all)")
    enabled: bool = Field(default=True, description="Whether interest is enabled")
    priority: int = Field(default=50, description="Priority for ordering (lower=higher)")
    instructions: str = Field(default="", description="Custom analysis instructions")
    # Git/query fields
    since: str = Field(default="", description="Since date for git log (ISO format)")
    limit: int = Field(default=50, description="Max results")
    file_path: str = Field(default="", description="File path for git-file or git-diff")
    # Analysis fields
    run_type: str = Field(default="periodic", description="periodic|manual|briefing")
    summary: str = Field(default="", description="Analysis summary text")
    feature_suggestions: str = Field(default="", description="JSON array of suggestions")
    code_trends: str = Field(default="", description="JSON object of trends")
    commits_analyzed: str = Field(default="", description="JSON array of analyzed SHAs")
    state_value: str = Field(default="", description="State value to set")
    # Filters
    author_name: str = Field(default="", description="Filter by author name")
    area: str = Field(default="", description="Filter by area tag")
    days: int = Field(default=30, description="Days for stats")
    # Commit analysis update
    analysis: str = Field(default="", description="Analysis text for a commit")
    areas: str = Field(default="", description="JSON array of area tags for a commit")
    # Notification
    commit_shas: str = Field(default="", description="JSON array of commit SHAs to mark notified")
    # PR fields
    pr_number: int = Field(default=0, description="PR number for git-pr-diff")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    text: str = ""
    error: str = ""


def _git_env() -> dict[str, str]:
    """Environment for git subprocess — adds safe.directory without modifying global config."""
    env = os.environ.copy()
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = TECHTREE_REPO
    return env


def _git_cmd(args: list[str], *, timeout: int = 30) -> str:
    """Run a git command in the techtree repo."""
    result = subprocess.run(
        ["git", "-C", TECHTREE_REPO, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_git_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git exited with code {result.returncode}")
    return result.stdout


def _parse_numstat(stat_output: str) -> tuple[int, int, int]:
    """Parse git diff --numstat output into (files_changed, insertions, deletions)."""
    files = insertions = deletions = 0
    for line in stat_output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            files += 1
            ins = parts[0]
            dels = parts[1]
            insertions += int(ins) if ins != "-" else 0
            deletions += int(dels) if dels != "-" else 0
    return files, insertions, deletions


def _detect_areas(diff_files: list[str], interests: list[dict]) -> list[str]:
    """Match changed file paths against interest paths and keywords."""
    areas = set()
    for interest in interests:
        interest_paths = interest.get("paths", [])
        interest_keywords = interest.get("keywords", [])
        if isinstance(interest_paths, str):
            interest_paths = json.loads(interest_paths)
        if isinstance(interest_keywords, str):
            interest_keywords = json.loads(interest_keywords)

        # Check path matches
        for fpath in diff_files:
            for ip in interest_paths:
                if fpath.startswith(ip) or ip in fpath:
                    areas.add(interest["name"])
                    break

        # Check keyword matches in file paths
        for fpath in diff_files:
            fpath_lower = fpath.lower()
            for kw in interest_keywords:
                if kw.lower() in fpath_lower:
                    areas.add(interest["name"])
                    break

    return sorted(areas)


def _get_diff_files(commit_sha: str) -> list[str]:
    """Get list of files changed in a commit."""
    try:
        output = _git_cmd(["diff-tree", "--no-commit-id", "-r", "--name-only", commit_sha])
        return [f.strip() for f in output.strip().splitlines() if f.strip()]
    except RuntimeError:
        return []


class TechtreeManagerTool(ScriptTool[Input, Output]):
    name = "techtree_manager"
    description = "Browse techtree git repo, track commits, manage interests, and analyze code changes"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        # ── Git commands ──
        if cmd == "git-pull":
            try:
                out = _git_cmd(["pull", "--ff-only"], timeout=60)
                return Output(success=True, text=out.strip() or "Already up to date.")
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-log":
            try:
                args = ["log", "--format=%H|%an|%ae|%aI|%s", f"-n{inp.limit}"]
                if inp.since:
                    args.append(f"--since={inp.since}")
                if inp.author_name:
                    args.append(f"--author={inp.author_name}")
                out = _git_cmd(args)
                items = []
                for line in out.strip().splitlines():
                    parts = line.split("|", 4)
                    if len(parts) >= 5:
                        items.append(
                            {
                                "sha": parts[0],
                                "author_name": parts[1],
                                "author_email": parts[2],
                                "date": parts[3],
                                "message": parts[4],
                            }
                        )
                return Output(success=True, items=items, count=len(items))
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-diff":
            try:
                if not inp.commit_sha:
                    return Output(success=False, error="commit_sha required")
                args = ["diff", f"{inp.commit_sha}~1..{inp.commit_sha}", "--stat"]
                if inp.file_path:
                    args.extend(["--", inp.file_path])
                out = _git_cmd(args)
                # Also get the full diff (truncated)
                full_args = ["diff", f"{inp.commit_sha}~1..{inp.commit_sha}"]
                if inp.file_path:
                    full_args.extend(["--", inp.file_path])
                full_diff = _git_cmd(full_args)
                if len(full_diff) > 50000:
                    full_diff = full_diff[:50000] + "\n\n... (truncated, diff too large)"
                return Output(success=True, text=f"=== STAT ===\n{out}\n=== DIFF ===\n{full_diff}")
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-show":
            try:
                if not inp.commit_sha:
                    return Output(success=False, error="commit_sha required")
                out = _git_cmd(["show", inp.commit_sha, "--stat", "--format=%H%n%an%n%ae%n%aI%n%B"])
                if len(out) > 50000:
                    out = out[:50000] + "\n\n... (truncated)"
                return Output(success=True, text=out)
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-branches":
            try:
                out = _git_cmd(["branch", "-a", "--format=%(refname:short) %(objectname:short) %(committerdate:iso)"])
                return Output(success=True, text=out.strip())
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-file":
            try:
                if not inp.file_path:
                    return Output(success=False, error="file_path required")
                out = _git_cmd(["show", f"HEAD:{inp.file_path}"])
                if len(out) > 50000:
                    out = out[:50000] + "\n\n... (truncated)"
                return Output(success=True, text=out)
            except RuntimeError as e:
                return Output(success=False, error=str(e))

        if cmd == "git-prs":
            return self._list_prs(inp)

        if cmd == "git-pr-diff":
            return self._pr_diff(inp)

        # ── Commit tracking ──
        if cmd == "ingest-new":
            return await self._ingest_new()

        if cmd == "list-commits":
            from app.db.repos.techtree import TechtreeCommitRepo

            repo = TechtreeCommitRepo()
            if inp.area:
                items = await repo.list_by_area(inp.area, limit=inp.limit)
            elif inp.author_name:
                items = await repo.list_by_author(inp.author_name, limit=inp.limit)
            else:
                items = await repo.list_recent(limit=inp.limit)
            return Output(success=True, items=items, count=len(items))

        if cmd == "get-commit":
            from app.db.repos.techtree import TechtreeCommitRepo

            c = await TechtreeCommitRepo().get(inp.commit_sha)
            return Output(success=bool(c), item=c or {}, error="" if c else "Commit not found")

        if cmd == "commit-stats":
            from app.db.repos.techtree import TechtreeCommitRepo

            stats = await TechtreeCommitRepo().get_stats(days=inp.days)
            return Output(success=True, item=stats)

        if cmd == "list-unnotified":
            from app.db.repos.techtree import TechtreeCommitRepo

            items = await TechtreeCommitRepo().list_unnotified(limit=inp.limit)
            return Output(success=True, items=items, count=len(items))

        if cmd == "mark-notified":
            from app.db.repos.techtree import TechtreeCommitRepo

            shas = json.loads(inp.commit_shas) if inp.commit_shas else []
            count = await TechtreeCommitRepo().mark_notified(shas)
            return Output(success=True, count=count)

        if cmd == "update-commit-analysis":
            from app.db.repos.techtree import TechtreeCommitRepo

            areas_list = json.loads(inp.areas) if inp.areas else None
            c = await TechtreeCommitRepo().update_analysis(inp.commit_sha, inp.analysis, areas_list)
            return Output(success=bool(c), item=c or {}, error="" if c else "Commit not found")

        # ── Interests ──
        if cmd == "create-interest":
            from app.db.repos.techtree import TechtreeInterestRepo

            iid = f"int_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            if inp.interest_id:
                iid = inp.interest_id
            paths_list = json.loads(inp.paths) if inp.paths else []
            kw_list = json.loads(inp.keywords) if inp.keywords else []
            item = await TechtreeInterestRepo().create(
                iid,
                inp.name,
                description=inp.description,
                paths=paths_list,
                keywords=kw_list,
                owner=inp.owner,
                enabled=inp.enabled,
                priority=inp.priority,
                instructions=inp.instructions,
            )
            return Output(success=True, item=item)

        if cmd == "get-interest":
            from app.db.repos.techtree import TechtreeInterestRepo

            item = await TechtreeInterestRepo().get(inp.interest_id)
            return Output(success=bool(item), item=item or {}, error="" if item else "Interest not found")

        if cmd == "list-interests":
            from app.db.repos.techtree import TechtreeInterestRepo

            items = await TechtreeInterestRepo().list_all(limit=inp.limit)
            return Output(success=True, items=items, count=len(items))

        if cmd == "update-interest":
            from app.db.repos.techtree import TechtreeInterestRepo

            fields: dict = {}
            if inp.name:
                fields["name"] = inp.name
            if inp.description:
                fields["description"] = inp.description
            if inp.paths:
                fields["paths"] = json.loads(inp.paths)
            if inp.keywords:
                fields["keywords"] = json.loads(inp.keywords)
            if inp.owner:
                fields["owner"] = inp.owner
            if inp.instructions:
                fields["instructions"] = inp.instructions
            if inp.priority != 50:
                fields["priority"] = inp.priority
            item = await TechtreeInterestRepo().update(inp.interest_id, **fields)
            return Output(success=bool(item), item=item or {}, error="" if item else "Interest not found")

        if cmd == "delete-interest":
            from app.db.repos.techtree import TechtreeInterestRepo

            ok = await TechtreeInterestRepo().delete(inp.interest_id)
            return Output(success=ok, error="" if ok else "Interest not found")

        if cmd == "toggle-interest":
            from app.db.repos.techtree import TechtreeInterestRepo

            item = await TechtreeInterestRepo().toggle(inp.interest_id, inp.enabled)
            return Output(success=bool(item), item=item or {}, error="" if item else "Interest not found")

        # ── Analysis runs ──
        if cmd == "create-analysis":
            from app.db.repos.techtree import TechtreeAnalysisRunRepo

            rid = inp.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(inp) % 0xFFFF:04x}"
            commits_list = json.loads(inp.commits_analyzed) if inp.commits_analyzed else []
            suggestions = json.loads(inp.feature_suggestions) if inp.feature_suggestions else []
            trends = json.loads(inp.code_trends) if inp.code_trends else {}
            item = await TechtreeAnalysisRunRepo().create(
                rid,
                inp.run_type,
                commits_analyzed=commits_list,
                summary=inp.summary,
                feature_suggestions=suggestions,
                code_trends=trends,
            )
            return Output(success=True, item=item)

        if cmd == "list-analyses":
            from app.db.repos.techtree import TechtreeAnalysisRunRepo

            items = await TechtreeAnalysisRunRepo().list_recent(limit=inp.limit)
            return Output(success=True, items=items, count=len(items))

        if cmd == "get-latest-analysis":
            from app.db.repos.techtree import TechtreeAnalysisRunRepo

            item = await TechtreeAnalysisRunRepo().get_latest()
            return Output(success=bool(item), item=item or {}, error="" if item else "No analysis runs yet")

        # ── State ──
        if cmd == "get-state":
            from app.db.repos.techtree import TechtreeStateRepo

            val = await TechtreeStateRepo().get(inp.state_key)
            return Output(
                success=val is not None,
                item={"key": inp.state_key, "value": val or ""},
                error="" if val is not None else "State key not found",
            )

        if cmd == "set-state":
            from app.db.repos.techtree import TechtreeStateRepo

            item = await TechtreeStateRepo().set(inp.state_key, inp.state_value)
            return Output(success=True, item=item)

        if cmd == "list-states":
            from app.db.repos.techtree import TechtreeStateRepo

            items = await TechtreeStateRepo().list_all()
            return Output(success=True, items=items, count=len(items))

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _list_prs(self, inp: Input) -> Output:
        """List open PRs via GitHub API. Falls back to listing recent remote branches if no token."""
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            return self._list_prs_api(inp, token)
        return self._list_recent_branches(inp)

    def _list_prs_api(self, inp: Input, token: str) -> Output:
        """List open PRs using GitHub REST API."""
        import urllib.request

        try:
            url = "https://api.github.com/repos/techtree-dev/techtree/pulls?state=open&per_page=" + str(inp.limit)
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                prs = json.loads(resp.read())
            items = []
            for pr in prs:
                author_login = pr.get("user", {}).get("login", "")
                if inp.author_name and inp.author_name.lower() not in author_login.lower():
                    continue
                items.append(
                    {
                        "number": pr.get("number"),
                        "title": pr.get("title", ""),
                        "author": author_login,
                        "branch": pr.get("head", {}).get("ref", ""),
                        "created_at": pr.get("created_at", ""),
                        "updated_at": pr.get("updated_at", ""),
                        "additions": pr.get("additions", 0),
                        "deletions": pr.get("deletions", 0),
                        "changed_files": pr.get("changed_files", 0),
                        "labels": [lb.get("name", "") for lb in pr.get("labels", [])],
                        "is_draft": pr.get("draft", False),
                    }
                )
            return Output(success=True, items=items, count=len(items))
        except Exception as e:
            return Output(success=False, error=f"GitHub API failed: {e}")

    def _list_recent_branches(self, inp: Input) -> Output:
        """Fallback: list recent remote branches with commit info (no GitHub token needed)."""
        import contextlib

        with contextlib.suppress(RuntimeError):
            _git_cmd(["fetch", "--all", "--prune"], timeout=60)

        try:
            # Get remote branches sorted by most recent commit, with author info
            out = _git_cmd(
                [
                    "for-each-ref",
                    "--sort=-committerdate",
                    f"--count={inp.limit}",
                    "--format=%(refname:short)|%(authorname)|%(committerdate:iso)|%(subject)",
                    "refs/remotes/origin/",
                ]
            )
            items = []
            base_branch = "origin/staging"
            for line in out.strip().splitlines():
                parts = line.split("|", 3)
                if len(parts) < 4:
                    continue
                branch, author, date, subject = parts
                # Skip HEAD and the base branch itself
                if branch in ("origin/HEAD", base_branch):
                    continue
                # Filter by author if requested
                if inp.author_name and inp.author_name.lower() not in author.lower():
                    continue
                # Get diff stats vs staging
                try:
                    numstat = _git_cmd(["diff", "--numstat", f"{base_branch}...{branch}"])
                    files_changed, additions, deletions = _parse_numstat(numstat)
                except RuntimeError:
                    files_changed = additions = deletions = 0
                items.append(
                    {
                        "branch": branch.removeprefix("origin/"),
                        "author": author,
                        "updated_at": date.strip(),
                        "last_commit_message": subject,
                        "additions": additions,
                        "deletions": deletions,
                        "changed_files": files_changed,
                    }
                )
            return Output(success=True, items=items, count=len(items))
        except RuntimeError as e:
            return Output(success=False, error=str(e))

    def _pr_diff(self, inp: Input) -> Output:
        """Get diff details for a PR or branch. Uses GitHub API if token available, else git diff."""
        token = os.environ.get("GITHUB_TOKEN", "")
        if token and inp.pr_number:
            return self._pr_diff_api(inp, token)
        # Fallback: diff a branch against staging
        if not inp.file_path and not inp.pr_number:
            return Output(success=False, error="pr_number or file_path (branch name) required")
        branch_name = inp.file_path  # Reuse file_path field for branch name in fallback
        if inp.pr_number:
            return Output(
                success=False, error="pr_number requires GITHUB_TOKEN. Use file_path with branch name instead."
            )
        return self._branch_diff(branch_name)

    def _pr_diff_api(self, inp: Input, token: str) -> Output:
        """Get PR file list via GitHub API."""
        import urllib.request

        try:
            url = f"https://api.github.com/repos/techtree-dev/techtree/pulls/{inp.pr_number}/files?per_page=100"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                files = json.loads(resp.read())
            # Also get PR details
            pr_url = f"https://api.github.com/repos/techtree-dev/techtree/pulls/{inp.pr_number}"
            pr_req = urllib.request.Request(
                pr_url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(pr_req, timeout=30) as resp:
                pr = json.loads(resp.read())
            file_list = [
                {"path": f.get("filename", ""), "additions": f.get("additions", 0), "deletions": f.get("deletions", 0)}
                for f in files
            ]
            item = {
                "number": pr.get("number"),
                "title": pr.get("title", ""),
                "author": pr.get("user", {}).get("login", ""),
                "branch": pr.get("head", {}).get("ref", ""),
                "body": (pr.get("body", "") or "")[:5000],
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
                "commits": pr.get("commits", 0),
                "files": file_list,
            }
            return Output(success=True, item=item)
        except Exception as e:
            return Output(success=False, error=f"GitHub API failed: {e}")

    def _branch_diff(self, branch_name: str) -> Output:
        """Get diff of a remote branch vs staging using git."""
        try:
            remote_ref = f"origin/{branch_name}"
            base = "origin/staging"
            # Get file list with stats
            numstat = _git_cmd(["diff", "--numstat", f"{base}...{remote_ref}"])
            files_changed, total_add, total_del = _parse_numstat(numstat)
            # Get file names
            name_only = _git_cmd(["diff", "--name-only", f"{base}...{remote_ref}"])
            file_list = [{"path": f.strip()} for f in name_only.strip().splitlines() if f.strip()]
            # Get commit log on this branch not in staging
            log_out = _git_cmd(["log", "--format=%H|%an|%aI|%s", f"{base}..{remote_ref}", "-n50"])
            commits = []
            for line in log_out.strip().splitlines():
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    commits.append(
                        {
                            "sha": parts[0],
                            "author": parts[1],
                            "date": parts[2],
                            "message": parts[3],
                        }
                    )
            item = {
                "branch": branch_name,
                "additions": total_add,
                "deletions": total_del,
                "changed_files": files_changed,
                "commits": len(commits),
                "commit_list": commits[:20],
                "files": file_list,
            }
            return Output(success=True, item=item)
        except RuntimeError as e:
            return Output(success=False, error=str(e))

    async def _ingest_new(self) -> Output:
        """Pull latest changes and ingest new commits into the database."""
        from app.db.repos.techtree import TechtreeCommitRepo, TechtreeInterestRepo, TechtreeStateRepo

        # 1. Pull
        try:
            _git_cmd(["pull", "--ff-only"], timeout=60)
        except RuntimeError as e:
            # Fetch instead if pull fails (e.g. local modifications)
            try:
                _git_cmd(["fetch", "--all"], timeout=60)
            except RuntimeError:
                return Output(success=False, error=f"Git pull failed: {e}")

        # 2. Get last analyzed SHA
        state_repo = TechtreeStateRepo()
        last_sha = await state_repo.get("last_analyzed_sha")

        # 3. Get new commits
        log_args = ["log", "--format=%H|%an|%ae|%aI|%s", "-n500"]
        if last_sha:
            log_args.append(f"{last_sha}..HEAD")
        try:
            out = _git_cmd(log_args)
        except RuntimeError as e:
            return Output(success=False, error=f"Git log failed: {e}")

        lines = [line for line in out.strip().splitlines() if line.strip()]
        if not lines:
            return Output(success=True, text="No new commits.", count=0)

        # 4. Load enabled interests for area detection
        interests = await TechtreeInterestRepo().list_enabled()

        # 5. Process each commit
        commit_repo = TechtreeCommitRepo()
        new_count = 0
        latest_sha = None

        for line in lines:
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue

            sha, author, email, date_str, message = parts
            if await commit_repo.exists(sha):
                continue

            # Parse date
            try:
                committed_at = datetime.fromisoformat(date_str)
            except ValueError:
                committed_at = datetime.now(UTC)

            # Get diff stats
            try:
                numstat = _git_cmd(["diff", f"{sha}~1..{sha}", "--numstat"])
                files_changed, insertions, deletions = _parse_numstat(numstat)
            except RuntimeError:
                files_changed = insertions = deletions = 0

            # Detect areas
            diff_files = _get_diff_files(sha)
            areas = _detect_areas(diff_files, interests)

            await commit_repo.create(
                sha,
                author,
                email,
                committed_at,
                message,
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                areas=areas,
            )
            new_count += 1
            if latest_sha is None:
                latest_sha = sha

        # 6. Update state cursor
        if latest_sha:
            await state_repo.set("last_analyzed_sha", latest_sha)

        return Output(success=True, count=new_count, text=f"Ingested {new_count} new commits.")


if __name__ == "__main__":
    TechtreeManagerTool.run()
