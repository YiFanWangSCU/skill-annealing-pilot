"""Download the pinned official ABCD source without exposing credentials."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from .constants import (
    EXPECTED_SOURCE_METADATA,
    SOURCE_COMMIT,
    SOURCE_FILES,
    SOURCE_REPOSITORY,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha(path: Path, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()  # noqa: S324 - Git object identity, not security.
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(
    output_dir: str | Path,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Download immutable upstream files and return a provenance manifest.

    ``gh`` is preferred because it reuses the user's configured GitHub CLI
    transport and keyring without placing a token in this process's arguments or
    output. Existing files are verified and reused; they are never overwritten.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing_manifest = destination / "source_manifest.json"
    if existing_manifest.is_file():
        try:
            return verify_sources(destination)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    selected_backend = _select_backend(backend)
    used_backends: set[str] = set()
    files: dict[str, dict[str, Any]] = {}

    for upstream_path in SOURCE_FILES:
        local_path = destination / Path(upstream_path).name
        endpoint = (
            f"repos/{SOURCE_REPOSITORY}/contents/{upstream_path}"
            f"?ref={SOURCE_COMMIT}"
        )
        if not local_path.exists():
            partial_path = local_path.with_name(local_path.name + ".part")
            if partial_path.exists():
                partial_path.unlink()
            try:
                if selected_backend == "auto":
                    if shutil.which("gh"):
                        try:
                            _download_with_gh(endpoint, partial_path)
                            used_backends.add("gh")
                        except RuntimeError:
                            if partial_path.exists():
                                partial_path.unlink()
                            _download_with_http(endpoint, partial_path)
                            used_backends.add("http")
                    else:
                        _download_with_http(endpoint, partial_path)
                        used_backends.add("http")
                elif selected_backend == "gh":
                    _download_with_gh(endpoint, partial_path)
                    used_backends.add("gh")
                else:
                    _download_with_http(endpoint, partial_path)
                    used_backends.add("http")
                _validate_expected_metadata(upstream_path, partial_path)
                partial_path.replace(local_path)
            except Exception:
                if partial_path.exists():
                    partial_path.unlink()
                raise
        _validate_expected_metadata(upstream_path, local_path)
        files[upstream_path] = {
            "local_name": local_path.name,
            "size": local_path.stat().st_size,
            "sha256": sha256_file(local_path),
            "git_blob_sha": git_blob_sha(local_path),
            "api_endpoint": f"https://api.github.com/{endpoint}",
        }

    manifest = {
        "dataset": "Action-Based Conversations Dataset (ABCD) v1.1",
        "repository": f"https://github.com/{SOURCE_REPOSITORY}",
        "commit": SOURCE_COMMIT,
        "license": "MIT",
        "paper": "https://aclanthology.org/2021.naacl-main.239/",
        "download_backend": (
            "+".join(sorted(used_backends)) if used_backends else "verified_existing"
        ),
        "files": files,
    }
    _write_json(destination / "source_manifest.json", manifest)
    return manifest


def verify_sources(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    manifest_path = directory / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("commit") != SOURCE_COMMIT:
        raise ValueError("source manifest commit does not match the frozen commit")
    if manifest.get("repository") != f"https://github.com/{SOURCE_REPOSITORY}":
        raise ValueError("source manifest repository does not match the frozen source")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or set(manifest_files) != set(SOURCE_FILES):
        raise ValueError("source manifest file set does not match the frozen source")
    for upstream_path in SOURCE_FILES:
        metadata = manifest_files[upstream_path]
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid source metadata for {upstream_path}")
        expected_local_name = Path(upstream_path).name
        if metadata.get("local_name") != expected_local_name:
            raise ValueError(f"unexpected local name for {upstream_path}")
        local_path = directory / metadata["local_name"]
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        expected = EXPECTED_SOURCE_METADATA[upstream_path]
        if metadata.get("size") != expected["size"]:
            raise ValueError(f"manifest size is not frozen for {upstream_path}")
        if metadata.get("sha256") != expected["sha256"]:
            raise ValueError(f"manifest SHA-256 is not frozen for {upstream_path}")
        if metadata.get("git_blob_sha") != expected["git_blob_sha"]:
            raise ValueError(f"manifest Git blob is not frozen for {upstream_path}")
        if local_path.stat().st_size != expected["size"]:
            raise ValueError(f"source size mismatch: {local_path}")
        if sha256_file(local_path) != expected["sha256"]:
            raise ValueError(f"source SHA-256 mismatch: {local_path}")
        if git_blob_sha(local_path) != expected["git_blob_sha"]:
            raise ValueError(f"source Git blob mismatch: {local_path}")
        _validate_expected_metadata(upstream_path, local_path)
    return manifest


def _select_backend(backend: str) -> str:
    if backend not in {"auto", "gh", "http"}:
        raise ValueError("backend must be one of: auto, gh, http")
    if backend == "auto":
        return "auto"
    if backend == "gh" and not shutil.which("gh"):
        raise RuntimeError("GitHub CLI was requested but `gh` is not available")
    return backend


def _download_with_gh(endpoint: str, destination: Path) -> None:
    with destination.open("wb") as output:
        result = subprocess.run(
            [
                "gh",
                "api",
                endpoint,
                "-H",
                "Accept: application/vnd.github.raw+json",
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh api download failed: {message}")


def _download_with_http(endpoint: str, destination: Path) -> None:
    request = urllib.request.Request(
        f"https://api.github.com/{endpoint}",
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "skill-annealing-abcd-builder",
        },
    )
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _validate_expected_metadata(upstream_path: str, local_path: Path) -> None:
    expected = EXPECTED_SOURCE_METADATA[upstream_path]
    if local_path.stat().st_size != expected["size"]:
        raise ValueError(f"unexpected upstream size for {upstream_path}")
    if git_blob_sha(local_path) != expected["git_blob_sha"]:
        raise ValueError(f"unexpected upstream Git blob for {upstream_path}")
    if expected.get("sha256") and sha256_file(local_path) != expected["sha256"]:
        raise ValueError(f"unexpected upstream SHA-256 for {upstream_path}")


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
