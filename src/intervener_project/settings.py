"""Config loading for the production pre-ML research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class LanguageSpec:
    key: str
    label: str
    corpus_id: str
    typology: str
    family: str
    region: str
    download_base_url: str
    archive_ext: str = ".tgz"
    explicit_download_url: str | None = None

    @property
    def download_url(self) -> str:
        if self.explicit_download_url:
            return self.explicit_download_url
        return f"{self.download_base_url}/{self.corpus_id}{self.archive_ext}"


@dataclass(frozen=True)
class ComplexityConfig:
    arity_weight: float
    subtree_weight: float
    depth_weight: float
    pos_weight: float
    pos_weights: dict[str, float]


@dataclass(frozen=True)
class ProjectSettings:
    title: str
    treebank_source: str
    release: str
    output_root: Path
    final_output_root: Path
    log_root: Path
    random_seed: int
    random_runs: int
    max_sentences: int | None
    include_zero_distance_dependencies: bool
    baselines: list[str]
    generate_global_report: bool
    languages: list[LanguageSpec]
    contributor_languages: dict[str, list[str]]
    complexity: ComplexityConfig
    config_path: Path

    @property
    def language_map(self) -> dict[str, LanguageSpec]:
        return {language.key: language for language in self.languages}


def load_project_settings(
    config_path: str | Path,
    contributor: str | None = None,
    languages_override: list[str] | None = None,
    max_sentences_override: int | None = None,
) -> ProjectSettings:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = payload["project"]
    complexity = payload["complexity"]
    download_base_url = project["download_base_url"]

    languages = [
        LanguageSpec(
            key=item["key"],
            label=item["label"],
            corpus_id=item["corpus_id"],
            typology=item["typology"],
            family=item["family"],
            region=item["region"],
            download_base_url=item.get("download_base_url", download_base_url),
            archive_ext=item.get("archive_ext", ".tgz"),
            explicit_download_url=item.get("download_url"),
        )
        for item in payload["languages"]
    ]
    contributor_languages = {
        name: value["languages"] for name, value in payload.get("contributors", {}).items()
    }

    if languages_override:
        requested = set(languages_override)
        languages = [language for language in languages if language.key in requested]
    elif contributor:
        requested = set(contributor_languages.get(contributor, []))
        languages = [language for language in languages if language.key in requested]
        if not requested:
            raise ValueError(f"Contributor '{contributor}' is not defined in {config_path}")

    if not languages:
        raise ValueError("No languages selected. Check the config, contributor key, or language override.")

    settings = ProjectSettings(
        title=project["title"],
        treebank_source=project["treebank_source"],
        release=project["release"],
        output_root=PROJECT_ROOT / project["output_root"],
        final_output_root=PROJECT_ROOT / project["final_output_root"],
        log_root=PROJECT_ROOT / project["log_root"],
        random_seed=int(project["random_seed"]),
        random_runs=int(project["random_runs"]),
        max_sentences=max_sentences_override if max_sentences_override is not None else project["max_sentences"],
        include_zero_distance_dependencies=bool(project["include_zero_distance_dependencies"]),
        baselines=list(project["baselines"]),
        generate_global_report=bool(project["generate_global_report"]),
        languages=languages,
        contributor_languages=contributor_languages,
        complexity=ComplexityConfig(
            arity_weight=float(complexity["weights"]["arity"]),
            subtree_weight=float(complexity["weights"]["subtree_size"]),
            depth_weight=float(complexity["weights"]["depth"]),
            pos_weight=float(complexity["weights"]["pos_weight"]),
            pos_weights={key: float(value) for key, value in complexity["pos_weights"].items()},
        ),
        config_path=config_path,
    )
    return settings
