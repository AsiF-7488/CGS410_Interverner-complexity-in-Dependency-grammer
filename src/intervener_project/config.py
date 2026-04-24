"""Static configuration for the intervener-complexity project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SUD_RELEASE = "2.17"
SUD_BASE_URL = f"https://grew.fr/download/SUD_{SUD_RELEASE}"


@dataclass(frozen=True)
class LanguageConfig:
    key: str
    label: str
    corpus_id: str | None
    typology: str
    family: str
    region: str
    available_in_sud: bool = True
    notes: str = ""

    @property
    def download_url(self) -> str | None:
        if not self.corpus_id:
            return None
        return f"{SUD_BASE_URL}/{self.corpus_id}.tgz"


LANGUAGES = [
    LanguageConfig(
        key="marathi",
        label="Marathi",
        corpus_id="SUD_Marathi-UFAL",
        typology="SOV",
        family="Indo-Aryan",
        region="South Asia",
    ),
    LanguageConfig(
        key="catalan",
        label="Catalan",
        corpus_id="SUD_Catalan-AnCora",
        typology="SVO",
        family="Romance",
        region="Europe",
    ),
    LanguageConfig(
        key="galician",
        label="Galician",
        corpus_id="SUD_Galician-CTG",
        typology="SVO",
        family="Romance",
        region="Europe",
    ),
    LanguageConfig(
        key="persian",
        label="Persian",
        corpus_id="SUD_Persian-Seraji",
        typology="SOV",
        family="Iranian",
        region="West Asia",
    ),
    LanguageConfig(
        key="tamil",
        label="Tamil",
        corpus_id="SUD_Tamil-TTB",
        typology="SOV",
        family="Dravidian",
        region="South Asia",
    ),
    LanguageConfig(
        key="telugu",
        label="Telugu",
        corpus_id="SUD_Telugu-MTG",
        typology="SOV",
        family="Dravidian",
        region="South Asia",
    ),
    LanguageConfig(
        key="indonesian",
        label="Indonesian",
        corpus_id="SUD_Indonesian-GSD",
        typology="SVO",
        family="Austronesian",
        region="Southeast Asia",
    ),
]

LANGUAGE_BY_KEY = {language.key: language for language in LANGUAGES}

DEFAULT_RANDOM_RUNS = 5
DEFAULT_SEED = 13
