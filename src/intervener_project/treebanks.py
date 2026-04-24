"""Download and manage configured treebanks."""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .config import LANGUAGE_BY_KEY, LANGUAGES, RAW_DIR


def ensure_treebank(language, force_download: bool = False) -> Path:
    archive_path = RAW_DIR / f"{language.corpus_id}.tgz"
    if str(language.download_url).endswith(".zip"):
        archive_path = RAW_DIR / f"{language.corpus_id}.zip"
    extract_dir = RAW_DIR / language.corpus_id

    if force_download and archive_path.exists():
        archive_path.unlink()

    if force_download and extract_dir.exists():
        for path in sorted(extract_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        extract_dir.rmdir()

    if not archive_path.exists():
        urllib.request.urlretrieve(language.download_url, archive_path)

    if not extract_dir.exists():
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(RAW_DIR)
        else:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(RAW_DIR)

    return extract_dir


def ensure_configured_treebanks(
    languages,
    force_download: bool = False,
) -> list[tuple[object, Path]]:
    available: list[tuple[object, Path]] = []

    for language in languages:
        path = ensure_treebank(language, force_download=force_download)
        available.append((language, path))

    return available


def selected_languages(language_keys: list[str] | None = None):
    if language_keys is None:
        return list(LANGUAGES)
    return [LANGUAGE_BY_KEY[key] for key in language_keys]


def ensure_selected_treebanks(
    language_keys: list[str] | None = None,
    force_download: bool = False,
):
    available = ensure_configured_treebanks(selected_languages(language_keys), force_download=force_download)
    return available, []
