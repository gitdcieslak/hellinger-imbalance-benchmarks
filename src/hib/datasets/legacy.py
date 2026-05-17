"""Legacy HDDT archive extraction and manifest utilities."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


DEFAULT_RAW_DIR = Path("data/raw/legacy_hddt")
DEFAULT_EXTRACTED_DIR = Path("data/extracted/legacy_hddt")
DEFAULT_PROCESSED_DIR = Path("data/processed/legacy_hddt")


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".data":
        return "data"
    if suffix == ".arff":
        return "arff"
    if suffix == ".txt":
        return "txt"
    return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _safe_members(archive: tarfile.TarFile, output_dir: Path) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        member_path = output_dir / member.name
        if not _is_within_directory(member_path, output_dir):
            raise ValueError(f"unsafe archive member path: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"unsupported archive link member: {member.name}")
        members.append(member)
    return members


def find_legacy_archives(raw_dir: Path) -> list[Path]:
    patterns = ("*.tar", "*.tar.gz", "*.tgz")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(raw_dir.glob(pattern))
    return sorted(set(path.resolve() for path in paths))


def autodetect_single_archive(raw_dir: Path) -> Path:
    archives = find_legacy_archives(raw_dir)
    if not archives:
        raise FileNotFoundError(f"no archive found under {raw_dir}")
    if len(archives) > 1:
        raise ValueError(
            f"multiple archives found under {raw_dir}; pass --archive explicitly"
        )
    return archives[0]


def extract_archive(archive_path: Path, output_dir: Path, force: bool = False) -> list[Path]:
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths: list[Path] = []
    with tarfile.open(archive_path, "r:*") as archive:
        members = _safe_members(archive, output_dir)
        for member in members:
            if member.isdir():
                continue
            destination = output_dir / member.name
            if destination.exists() and not force:
                raise FileExistsError(
                    f"destination exists: {destination}; rerun with --force to overwrite"
                )
        archive.extractall(path=output_dir, members=members)

    for path in output_dir.rglob("*"):
        if path.is_file():
            extracted_paths.append(path)
    return sorted(extracted_paths)


def generate_manifest(extracted_dir: Path, manifest_path: Path) -> list[dict[str, object]]:
    extracted_dir = extracted_dir.resolve()
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for file_path in sorted(path for path in extracted_dir.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(extracted_dir).as_posix()
        entry = {
            "dataset_name": file_path.stem,
            "relative_path": relative_path,
            "file_size_bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
            "detected_format": detect_format(file_path),
            "n_rows": None,
            "n_columns": None,
            "notes": "",
        }
        entries.append(entry)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
    return entries
