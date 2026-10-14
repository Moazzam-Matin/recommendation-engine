"""Run the item-based collaborative-filtering baseline."""

import pprint
import sys
from pathlib import Path

import pandas as pd

# Allow running this script directly from the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommendation_engine.evaluation import evaluate_model, split_ratings
from recommendation_engine.models.collaborative import (
    ItemBasedCollaborativeFilter,
)


def main() -> None:
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
        min_user_ratings=5,
        test_fraction=0.2,
        random_state=42,
    )

    print(f"Training records: {len(train):,}")
    print(f"Test records: {len(test):,}")
    print("\nRunning item-based collaborative-filtering baseline...")

    model = ItemBasedCollaborativeFilter(
        n_neighbours=50,
        min_co_ratings=5,
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


if __name__ == "__main__":
    main()
