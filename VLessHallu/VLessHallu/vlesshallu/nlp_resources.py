from __future__ import annotations

import importlib.metadata
import importlib.util
from typing import Any


SPACY_MODEL = "en_core_web_lg"
SPACY_DISTRIBUTION = "en-core-web-lg"

NLTK_RESOURCES: dict[str, tuple[str, ...]] = {
    "punkt": ("tokenizers/punkt",),
    "punkt_tab": ("tokenizers/punkt_tab/english",),
    "averaged_perceptron_tagger": ("taggers/averaged_perceptron_tagger",),
    "averaged_perceptron_tagger_eng": (
        "taggers/averaged_perceptron_tagger_eng",
    ),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
}


def ensure_nlp_resources() -> dict[str, Any]:
    try:
        import nltk
        import spacy  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "evaluation dependencies are missing; install the project with .[eval]"
        ) from error

    for package, locators in NLTK_RESOURCES.items():
        if _find_nltk(locators) is None:
            ok = nltk.download(package, quiet=False, raise_on_error=True)
            if not ok or _find_nltk(locators) is None:
                raise RuntimeError(f"failed to prepare NLTK resource: {package}")

    if importlib.util.find_spec(SPACY_MODEL) is None:
        from spacy.cli import download

        download(SPACY_MODEL)
    if importlib.util.find_spec(SPACY_MODEL) is None:
        raise RuntimeError(f"failed to prepare spaCy model: {SPACY_MODEL}")
    return nlp_resource_lock()


def nlp_resource_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for package, locators in NLTK_RESOURCES.items():
        location = _find_nltk(locators)
        checks.append(
            {
                "name": f"nltk_{package}",
                "ok": location is not None,
                "required": True,
                "detail": location or ", ".join(locators),
            }
        )
    model_present = importlib.util.find_spec(SPACY_MODEL) is not None
    checks.append(
        {
            "name": SPACY_MODEL,
            "ok": model_present,
            "required": True,
            "detail": _version(SPACY_DISTRIBUTION) if model_present else "not installed",
        }
    )
    return checks


def nlp_resource_lock() -> dict[str, Any]:
    missing = [
        package
        for package, locators in NLTK_RESOURCES.items()
        if _find_nltk(locators) is None
    ]
    if missing or importlib.util.find_spec(SPACY_MODEL) is None:
        raise RuntimeError(f"evaluation NLP resources are incomplete: {missing}")
    return {
        "nltk": {
            package: _find_nltk(locators)
            for package, locators in NLTK_RESOURCES.items()
        },
        "spacy_model": {
            "name": SPACY_MODEL,
            "version": _version(SPACY_DISTRIBUTION),
        },
    }


def _find_nltk(locators: tuple[str, ...]) -> str | None:
    try:
        import nltk
    except ImportError:
        return None
    for locator in locators:
        try:
            return str(nltk.data.find(locator))
        except LookupError:
            pass
    return None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
