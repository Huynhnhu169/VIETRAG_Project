from pathlib import Path

from vietrag.config import load_config


def test_experiment_config_extends_default() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "P0_bm25.yaml")
    assert config["project"]["seed"] == 42
    assert config["retrieval"]["mode"] == "bm25"
    assert config["retrieval"]["bm25"]["k1"] == 1.5
    assert config["reranking"]["enabled"] is False
