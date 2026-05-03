"""
Git Auto-Detection Utilities

Utilities for automatically detecting Git context (repository, commit, branch)
for model registration.
"""

import logging
import os
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GitContext:
    """Git repository context information."""

    def __init__(
        self,
        repository_url: Optional[str] = None,
        commit_sha: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_author: Optional[str] = None,
        branch: Optional[str] = None,
        tag: Optional[str] = None,
        is_dirty: bool = False,
    ):
        """Initialize Git context."""
        self.repository_url = repository_url
        self.commit_sha = commit_sha
        self.commit_message = commit_message
        self.commit_author = commit_author
        self.branch = branch
        self.tag = tag
        self.is_dirty = is_dirty

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary for API submission."""
        return {
            "github_repository_url": self.repository_url,
            "github_commit_hash": self.commit_sha,
            "github_commit_message": self.commit_message,
            "github_commit_author": self.commit_author,
            "github_branch": self.branch,
            "github_tag": self.tag,
        }

    def __repr__(self) -> str:
        """String representation."""
        parts = []
        if self.repository_url:
            parts.append(f"repo={self.repository_url}")
        if self.commit_sha:
            parts.append(f"commit={self.commit_sha[:7]}")
        if self.branch:
            parts.append(f"branch={self.branch}")
        if self.tag:
            parts.append(f"tag={self.tag}")
        return f"GitContext({', '.join(parts)})"


def detect_git_context(path: Optional[str] = None) -> Optional[GitContext]:
    """
    Auto-detect Git context from the current directory or specified path.

    Args:
        path: Path to check for Git repository (defaults to current directory)

    Returns:
        GitContext object if Git repository found, None otherwise

    Example:
        >>> context = detect_git_context()
        >>> if context:
        ...     print(f"Repository: {context.repository_url}")
        ...     print(f"Commit: {context.commit_sha}")
        ...     print(f"Branch: {context.branch}")
    """
    try:
        import git
    except ImportError:
        logger.warning("GitPython not installed. Install with: pip install gitpython")
        return _detect_git_context_subprocess(path)

    try:
        # Find repository
        search_path = path or os.getcwd()
        repo = git.Repo(search_path, search_parent_directories=True)

        # Get repository URL
        repository_url = None
        try:
            remote = repo.remote("origin")
            repository_url = remote.url
            # Convert SSH URLs to HTTPS
            if repository_url.startswith("git@github.com:"):
                repository_url = repository_url.replace("git@github.com:", "https://github.com/")
            if repository_url.endswith(".git"):
                repository_url = repository_url[:-4]
        except Exception as e:
            logger.debug(f"Could not get remote URL: {e}")

        # Get current commit
        commit_sha = None
        commit_message = None
        commit_author = None

        try:
            head_commit = repo.head.commit
            commit_sha = head_commit.hexsha
            commit_message = head_commit.message.strip()
            commit_author = str(head_commit.author)
        except Exception as e:
            logger.debug(f"Could not get commit info: {e}")

        # Get current branch
        branch = None
        try:
            if not repo.head.is_detached:
                branch = repo.active_branch.name
        except Exception as e:
            logger.debug(f"Could not get branch: {e}")

        # Get current tag (if on a tag)
        tag = None
        try:
            tags = [tag for tag in repo.tags if tag.commit == repo.head.commit]
            if tags:
                tag = tags[0].name
        except Exception as e:
            logger.debug(f"Could not get tag: {e}")

        # Check if working directory is dirty
        is_dirty = repo.is_dirty(untracked_files=True)
        if is_dirty:
            logger.warning("Working directory has uncommitted changes")

        context = GitContext(
            repository_url=repository_url,
            commit_sha=commit_sha,
            commit_message=commit_message,
            commit_author=commit_author,
            branch=branch,
            tag=tag,
            is_dirty=is_dirty,
        )

        logger.info(f"Detected Git context: {context}")
        return context

    except git.InvalidGitRepositoryError:
        logger.debug(f"No Git repository found at {search_path}")
        return None
    except Exception as e:
        logger.error(f"Error detecting Git context: {e}")
        return None


def _detect_git_context_subprocess(path: Optional[str] = None) -> Optional[GitContext]:
    """
    Fallback Git detection using subprocess (when GitPython not available).

    Args:
        path: Path to check for Git repository

    Returns:
        GitContext object or None
    """
    try:
        cwd = path or os.getcwd()

        # Check if Git is available
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            cwd=cwd,
        )

        # Get repository URL
        repository_url = None
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            repository_url = result.stdout.strip()

            # Convert SSH to HTTPS
            if repository_url.startswith("git@github.com:"):
                repository_url = repository_url.replace("git@github.com:", "https://github.com/")
            if repository_url.endswith(".git"):
                repository_url = repository_url[:-4]
        except subprocess.CalledProcessError:
            pass

        # Get commit SHA
        commit_sha = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            commit_sha = result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Get commit message
        commit_message = None
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            commit_message = result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Get commit author
        commit_author = None
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%an <%ae>"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            commit_author = result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Get branch
        branch = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            branch_output = result.stdout.strip()
            if branch_output != "HEAD":  # Not in detached HEAD
                branch = branch_output
        except subprocess.CalledProcessError:
            pass

        # Get tag
        tag = None
        try:
            result = subprocess.run(
                ["git", "describe", "--exact-match", "--tags", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            tag = result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Check if dirty
        is_dirty = False
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
            is_dirty = bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            pass

        context = GitContext(
            repository_url=repository_url,
            commit_sha=commit_sha,
            commit_message=commit_message,
            commit_author=commit_author,
            branch=branch,
            tag=tag,
            is_dirty=is_dirty,
        )

        logger.info(f"Detected Git context (subprocess): {context}")
        return context

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Git not available or not a Git repository")
        return None
    except Exception as e:
        logger.error(f"Error detecting Git context with subprocess: {e}")
        return None


def validate_git_context(context: GitContext, require_clean: bool = False) -> bool:
    """
    Validate Git context before model registration.

    Args:
        context: GitContext object
        require_clean: Whether to require a clean working directory

    Returns:
        True if valid, False otherwise
    """
    if not context:
        logger.warning("No Git context provided")
        return False

    if not context.commit_sha:
        logger.error("Git context missing commit SHA")
        return False

    if require_clean and context.is_dirty:
        logger.error("Working directory has uncommitted changes (require_clean=True)")
        return False

    return True
