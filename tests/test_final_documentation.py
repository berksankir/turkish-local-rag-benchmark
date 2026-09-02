from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


PROVENANCE_MARKERS = (
    "creation_method=ai_assisted",
    "dataset_release_approved=true",
    "approved_by=berksankir",
    "approval_scope=dataset_level_with_sample_audit",
    "all_records_human_reviewed=false",
    "20/50",
    "final_gold=false",
)


def test_english_and_turkish_readmes_cross_link_and_match_provenance() -> None:
    english = Path("README.md").read_text(encoding="utf-8")
    turkish = Path("README.tr.md").read_text(encoding="utf-8")

    assert "[Türkçe](README.tr.md)" in english
    assert "[English](README.md)" in turkish
    for marker in PROVENANCE_MARKERS:
        assert marker in english
        assert marker in turkish
    assert "not presented as a human-reviewed gold set" in english
    assert "human-reviewed gold set olarak sunulmamaktadır" in turkish


def test_documented_cli_modules_exist() -> None:
    for module in (
        "download",
        "corpus_lock",
        "extract",
        "chunk",
        "index",
        "setup_generation",
        "query",
        "review",
        "evaluate",
        "evaluate_generation",
    ):
        assert importlib.util.find_spec(f"turkish_local_rag.{module}") is not None


def test_final_error_analysis_preserves_reported_limitations() -> None:
    analysis = Path("docs/error_analysis.md").read_text(encoding="utf-8")

    for marker in PROVENANCE_MARKERS:
        assert marker in analysis
    for heading in (
        "Retrieval miss",
        "Correct document, wrong physical page",
        "Evidence-gate false abstention",
        "Unanswerable question answered",
        "Generator/schema failure",
        "Citation mismatch",
        "Token F1 below 0.50",
        "Incomplete key-fact coverage",
    ):
        assert heading in analysis
    assert "0.700" in analysis
    assert "0.500" in analysis
    assert "test split was not used" in analysis.lower()


def test_readmes_warn_about_pymupdf_and_identify_repository_license() -> None:
    for path in (Path("README.md"), Path("README.tr.md")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert "AGPL-3.0" in text
        assert "commercial" in text or "ticari" in text
        assert "[MIT](LICENSE)" in text
        assert "third-party" in text or "üçüncü taraf" in text


def test_repository_has_selected_mit_license_consistently() -> None:
    license_text = Path("LICENSE").read_text(encoding="utf-8")
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Berk Sankır" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert project["project"]["license"] == "MIT"
