import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from hib.datasets.legacy import (
    autodetect_single_archive,
    detect_format,
    extract_archive,
    generate_manifest,
    sha256_file,
)


def _create_tar_with_files(archive_path: Path, files: dict[str, bytes]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for member_name, content in files.items():
            info = tarfile.TarInfo(name=member_name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_safe_tar_extraction(tmp_path):
    archive = tmp_path / "raw" / "legacy.tar.gz"
    _create_tar_with_files(archive, {"foo/data.csv": b"a,b\n1,2\n"})

    output_dir = tmp_path / "extracted"
    extracted = extract_archive(archive, output_dir)

    assert (output_dir / "foo" / "data.csv").exists()
    assert any(path.name == "data.csv" for path in extracted)


def test_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / "raw" / "unsafe.tar.gz"
    _create_tar_with_files(archive, {"../escape.txt": b"bad"})

    with pytest.raises(ValueError, match="unsafe archive member path"):
        extract_archive(archive, tmp_path / "extracted")


def test_manifest_generation_includes_hash_and_format(tmp_path):
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir(parents=True)
    file_path = extracted_dir / "dataset.csv"
    file_path.write_bytes(b"x,y\n1,0\n")

    manifest_path = tmp_path / "processed" / "manifest.json"
    entries = generate_manifest(extracted_dir, manifest_path)

    assert len(entries) == 1
    assert entries[0]["dataset_name"] == "dataset"
    assert entries[0]["detected_format"] == "csv"
    assert entries[0]["n_rows"] is None
    assert entries[0]["n_columns"] is None

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert content[0]["sha256"] == hashlib.sha256(b"x,y\n1,0\n").hexdigest()


def test_format_detection_supported_types(tmp_path):
    assert detect_format(tmp_path / "a.csv") == "csv"
    assert detect_format(tmp_path / "a.data") == "data"
    assert detect_format(tmp_path / "a.arff") == "arff"
    assert detect_format(tmp_path / "a.txt") == "txt"
    assert detect_format(tmp_path / "a.bin") == "unknown"


def test_sha256_generation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_archive_autodetection(tmp_path):
    raw_dir = tmp_path / "raw" / "legacy_hddt"
    raw_dir.mkdir(parents=True)
    archive = raw_dir / "only.tar.gz"
    _create_tar_with_files(archive, {"a.txt": b"ok"})

    detected = autodetect_single_archive(raw_dir)
    assert detected == archive.resolve()


def test_archive_autodetection_errors_when_multiple(tmp_path):
    raw_dir = tmp_path / "raw" / "legacy_hddt"
    raw_dir.mkdir(parents=True)
    _create_tar_with_files(raw_dir / "a.tar.gz", {"a.txt": b"1"})
    _create_tar_with_files(raw_dir / "b.tar.gz", {"b.txt": b"2"})

    with pytest.raises(ValueError, match="multiple archives"):
        autodetect_single_archive(raw_dir)


def test_force_overwrite_handling(tmp_path):
    archive = tmp_path / "raw" / "legacy.tar.gz"
    _create_tar_with_files(archive, {"set/data.txt": b"fresh"})

    output_dir = tmp_path / "extracted"
    target = output_dir / "set" / "data.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="destination exists"):
        extract_archive(archive, output_dir, force=False)

    extract_archive(archive, output_dir, force=True)
    assert target.read_bytes() == b"fresh"
