"""The whole loop on tiny data: train, register, promote, export, serve, evaluate, watch for drift."""

from pathlib import Path

import pytest

from fraud_detection.cli import main
from fraud_detection.metrics import CostModel
from fraud_detection.pipeline import drift, evaluate, train
from fraud_detection.service import load_scorer


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts next to the working directory
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_train_promote_export_and_serve(tiny_parquet: Path, tracking_uri: str, tmp_path: Path) -> None:
    first = train(tiny_parquet, tracking_uri, CostModel())
    assert first.promoted
    assert first.champion_cost is None
    assert "## Cost curve" in first.report

    # The same data and settings give the same model, so the same cost: not strictly better, not promoted.
    second = train(tiny_parquet, tracking_uri, CostModel())
    assert not second.promoted
    assert second.champion_cost == pytest.approx(first.candidate_cost)

    model_dir = tmp_path / "champion"
    assert main(["export", "--tracking-uri", tracking_uri, "--out", str(model_dir)]) == 0
    scorer = load_scorer(model_dir, tiny_parquet)
    assert scorer.version == first.version

    assert "Champion: version 1" in evaluate(tiny_parquet, tracking_uri)


def test_the_drift_scenario_raises_alerts_that_the_quiet_one_does_not(tiny_parquet: Path, tracking_uri: str) -> None:
    train(tiny_parquet, tracking_uri, CostModel())
    shifted = drift(tiny_parquet, tracking_uri, "shifted")
    assert {"is_foreign", "model score"} <= set(shifted.alerting)
    assert {row.name: row.status for row in shifted.rows}["amount_log"] != "stable"
    assert drift(tiny_parquet, tracking_uri, "none").alerting == []
