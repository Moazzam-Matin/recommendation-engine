import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_baseline", SCRIPT_PATH)
run_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_baseline)

from contextlib import nullcontext


def test_main_logs_parameters_and_metrics(monkeypatch):
    logged_experiment = []
    logged_params = []
    logged_metrics = []

    sample_results = {
        "population": {
            "users_evaluated": 10,
            "users_with_relevant_test_items": 8,
            "catalog_size": 100,
        },
        "diagnostics": {
            "reachable_catalog_size": 75,
            "reachability_rate": 75.0,
            "users_with_empty_recommendations": 0,
        },
        "metrics": {
            5: {
                "precision": 0.2,
                "recall": 0.3,
                "ndcg": 0.25,
                "catalog_coverage": 60.0,
                "completion": 1.0,
                "full_list_rate": 1.0,
            },
        },
    }

    monkeypatch.setattr(
        run_baseline.mlflow,
        "set_experiment",
        lambda name: logged_experiment.append(name),
    )
    monkeypatch.setattr(
        run_baseline.mlflow,
        "start_run",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        run_baseline.mlflow,
        "log_params",
        lambda params: logged_params.append(params),
    )
    monkeypatch.setattr(
        run_baseline.mlflow,
        "log_metrics",
        lambda metrics: logged_metrics.append(metrics),
    )
    monkeypatch.setattr(
        run_baseline,
        "run_experiment",
        lambda **params: sample_results,
    )

    run_baseline.main()

    assert logged_experiment == ["recommendation-engine"]

    assert len(logged_params) == 1
    assert logged_params[0]["n_neighbours"] == 50
    assert logged_params[0]["min_co_ratings"] == 5

    assert len(logged_metrics) == 2

    all_metrics = {}
    for metrics in logged_metrics:
        all_metrics.update(metrics)

    assert len(all_metrics) == 12
    assert all_metrics["users_evaluated"] == 10
    assert all_metrics["reachability_rate"] == 75.0
    assert all_metrics["precision_at_5"] == 0.2
    assert all_metrics["ndcg_at_5"] == 0.25
