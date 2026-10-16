"""Run the item-based collaborative-filtering baseline."""

import pprint
import sys
from pathlib import Path

import mlflow
import pandas as pd

# Allow running this script directly from the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendation_engine.evaluation import evaluate_model, split_ratings
from recommendation_engine.models.collaborative import (
    ItemBasedCollaborativeFilter,
)


def run_experiment(
    test_fraction: float = 0.2,
    random_state: int = 42,
    min_user_ratings: int = 5,
    n_neighbours: int = 50,
    min_co_ratings: int = 5,
) -> dict:

    data_path = PROJECT_ROOT / "data" / "raw" / "ml-1m" / "ratings.dat"

    print("Loading MovieLens 1M ratings...")
    ratings = pd.read_csv(
        data_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )
    ratings = ratings[["UserID", "MovieID", "Rating"]]

    print(f"Raw rating records: {len(ratings):,}")

    train, test = split_ratings(
        ratings,
        min_user_ratings=min_user_ratings,
        test_fraction=test_fraction,
        random_state=random_state,
    )

    print(f"Training records: {len(train):,}")
    print(f"Test records: {len(test):,}")
    print("\nRunning item-based collaborative-filtering baseline...")

    model = ItemBasedCollaborativeFilter(
        n_neighbours=n_neighbours,
        min_co_ratings=min_co_ratings,
    )

    results = evaluate_model(
        model,
        train,
        test,
        k_values=(5, 10, 20),
        full_ratings=ratings,
    )

    print("\n=== BASELINE RESULTS ===")
    pprint.pprint(results, sort_dicts=False, width=100)
    return results


def main() -> None:
    mlflow.set_experiment("recommendation-engine")
    with mlflow.start_run(run_name = "item-cf-neighbours-5_min_co_ratings-2"):
        params = {
            "test_fraction": 0.2,
            "random_state": 42,
            "min_user_ratings": 5,
            "n_neighbours": 5,
            "min_co_ratings": 2,
        }
        mlflow.log_params(params)
        results = run_experiment(**params)
        mlflow.log_metrics({
            "users_evaluated": results["population"]["users_evaluated"],
            "users_with_relevant_test_items": results["population"]["users_with_relevant_test_items"],
            "catalog_size": results["population"]["catalog_size"],
            "reachable_catalog_size": results["diagnostics"]["reachable_catalog_size"],
            "reachability_rate": results["diagnostics"]["reachability_rate"],
            "users_with_empty_recommendations": results["diagnostics"]["users_with_empty_recommendations"],
        })

        for k, metrics in results["metrics"].items():
            mlflow.log_metrics({
                f"precision_at_{k}": metrics["precision"],
                f"recall_at_{k}": metrics["recall"],
                f"ndcg_at_{k}": metrics["ndcg"],
                f"catalog_coverage_at_{k}": metrics["catalog_coverage"],
                f"completion_at_{k}": metrics["completion"],
                f"full_list_rate_at_{k}": metrics["full_list_rate"],
            })

if __name__ == "__main__":
    main()
