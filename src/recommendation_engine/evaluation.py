"""Data Prepration and splitting utilities for offline evaluation"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"UserID", "MovieID", "Rating"}

def prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Validate ratings and average duplicate user-movie pairs."""

    if not isinstance(ratings, pd.DataFrame):
        raise TypeError("Ratings must be pandas DataFrame.")

    missing_columns = REQUIRED_COLUMNS - set(ratings.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"ratings is missing required columns: {missing}."
        )

    if ratings.empty:
        raise ValueError("ratings must not be empty.")

    ratings = ratings[["UserID", "MovieID", "Rating"]].copy()

    if ratings.isnull().any().any():
        raise ValueError("ratings must not contain missing values in UserID, MovieId and Rating.")

    if not pd.api.types.is_numeric_dtype(ratings["Rating"]):
        raise TypeError("ratings must be numeric")

    if not pd.api.types.is_numeric_dtype(ratings["UserID"]):
            raise TypeError("UserID values must be numeric.")

    if not pd.api.types.is_numeric_dtype(ratings["MovieID"]):
            raise TypeError("MovieID values must be numeric.")

    if not ratings["Rating"].between(1, 5).all():
         raise ValueError("ratings must be between 1 and 5.")

     # Combine duplicate ratings for the same user and movie and calculate their average.
    ratings = (
         ratings.groupby(
              ["UserID", "MovieID"],
              as_index = False, #as_index=False keeps UserID and MovieID as ordinary columns.
              sort = True
         )["Rating"].mean()
    )

    return ratings


def split_ratings( 
          ratings: pd.DataFrame,
            *, 
            min_user_ratings: int = 5, 
            test_fraction: float = 0.2, 
            random_state: int = 42, 
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible per-user train/test split.

    Duplicate user-movie pairs are aggregated before eligibility is checked. 
    Users with fewer than min_user_ratings unique movies are excluded. 
    Each retained user gets at least one test rating and at least one training rating. 
    """

    if (not isinstance(min_user_ratings, int) or isinstance(min_user_ratings, bool)):
        raise TypeError("min_user_ratings must be an integer.")

    if min_user_ratings < 2:
         raise ValueError("min_user_ratings must be at least 2.")

    if not isinstance(test_fraction, (int, float)) or isinstance(test_fraction, bool):
         raise TypeError("test_fraction must be numeric.")

    if not 0 < test_fraction < 1:
         raise ValueError("test_fraction must be between 0 and 1.")

    if not isinstance(random_state, int) or isinstance(random_state, bool):
         raise TypeError("random_state must be an integer.")

    prepared = prepare_ratings(ratings)

    user_counts = prepared.groupby("UserID")["MovieID"].transform("size")
    eligible = prepared.loc[user_counts >= min_user_ratings].copy()

    if eligible.empty:
         raise ValueError("No users meet the minimum unique-rating threshold.")

    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, user_ratings in eligible.groupby("UserID", sort=True):
         user_ratings = user_ratings.reset_index(drop=True)
         n_ratings = len(user_ratings)

         n_test = max(1, math.floor(n_ratings * test_fraction))
         n_test = min(n_test, n_ratings - 1)

         test_indices = rng.choice(
              n_ratings,
              size = n_test,
              replace = False,
         )
         test_mask = user_ratings.index.isin(test_indices)

         train_parts.append(user_ratings.loc[~test_mask])
         test_parts.append(user_ratings.loc[test_mask])

    train = pd.concat(train_parts, ignore_index = True)
    test = pd.concat(test_parts, ignore_index = True)

    return train, test


def precision_at_k(recommendations, relevant_items, k: int) -> float:
    """Calculate Precision@K for a single user."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    recommended = list(recommendations)[:k]
    relevant = set(relevant_items)

    hits = sum(item in relevant for item in recommended)

    return hits / k


def recall_at_k(recommendations, relevant_items, k: int) -> float:
    """Calculate Recall@K for a single user."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = set(relevant_items)

    if not relevant:
        raise ValueError("relevant_items must not be empty.")

    recommended = list(recommendations)[:k]
    hits = sum(item in relevant for item in recommended)

    return hits / len(relevant)


def ndcg_at_k(recommendations, relevant_items, k: int) -> float:
    """Calculate binary NDCG@K for a single user."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = set(relevant_items)

    if not relevant:
        raise ValueError("relevant_items must not be empty.")

    recommended = list(recommendations)[:k]

    discounts = [
        1 / math.log2(rank + 2)
        for rank in range(k)
    ]

    dcg = sum(
        discounts[rank]
        for rank, item in enumerate(recommended)
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(discounts[:ideal_hits])

    return dcg / idcg



def evaluate_model(
    model,
    train_ratings,
    test_ratings,
    *,
    k_values=(5, 10, 20),
    full_ratings: pd.DataFrame |None =None,
):
    """Evaluate a recommendation model using held-out ratings."""
    train = prepare_ratings(train_ratings)
    test = prepare_ratings(test_ratings)

    # Validate K values.
    try:
        k_values = tuple(k_values)
    except TypeError as exc:
        raise TypeError("k_values must be an iterable of integers.") from exc

    if not k_values:
        raise ValueError("k_values must not be empty.")

    if any(
        not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_))
        for k in k_values
    ):
        raise TypeError("Every K value must be an integer.")

    if any(k <= 0 for k in k_values):
        raise ValueError("Every K value must be greater than 0.")

    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must not contain duplicates.")

    # Train and test must not contain the same user-movie pairs.
    train_pairs = set(
        zip(train["UserID"], train["MovieID"])
    )
    test_pairs = set(
        zip(test["UserID"], test["MovieID"])
    )

    if train_pairs & test_pairs:
        raise ValueError(
            "Training and test data contain overlapping user-movie pairs."
        )

    # Use the complete dataset to calculate catalog coverage correctly.
    if full_ratings is None:
        catalog_source = pd.concat([train, test], ignore_index=True)
    else:
        catalog_source = prepare_ratings(full_ratings)

    catalog = set(catalog_source["MovieID"])

    evaluated_movies = set(train["MovieID"]) | set(test["MovieID"])
    if not evaluated_movies.issubset(catalog):
        raise ValueError(
            "full_ratings must contain every movie in training and test data."
        )

    if train.empty or test.empty:
        raise ValueError("Training and test data must both be non-empty.")

    # Fit only on training data. Never train on held-out ratings.
    model.fit(train)

    test_users = sorted(test["UserID"].unique())
    relevant_by_user = {
        user_id: set(
            user_test.loc[user_test["Rating"] >= 4, "MovieID"]
        )
        for user_id, user_test in test.groupby("UserID")
    }

    # Request the longest list once per user. This also helps estimate
    # the catalog that the model can reach for these users.
    request_size = max(max(k_values), len(catalog))
    recommendations_by_user = {}
    recommendation_errors = {}

    for user_id in test_users:
        try:
            raw_recommendations = model.recommend(
                user_id, n=request_size
            )
            recommendations = list(raw_recommendations)

            # Recommendation lists should contain unique movie IDs.
            recommendations = list(dict.fromkeys(recommendations))

            recommendations_by_user[user_id] = recommendations

        except Exception as exc:  # noqa: BLE001
            # Keep the user in the evaluation population, even if
            # recommendation generation fails.
            recommendations_by_user[user_id] = []
            error_name = type(exc).__name__
            recommendation_errors[error_name] = (
                recommendation_errors.get(error_name, 0) + 1
            )

    users_with_relevant_items = [
        user_id
        for user_id in test_users
        if relevant_by_user.get(user_id)
    ]

    metrics = {}
    diagnostics_by_k = {}

    for k in k_values:
        precisions = []
        recalls = []
        ndcgs = []
        completions = []
        users_with_full_lists = 0
        empty_users = 0
        incomplete_users = 0
        recommended_movies = set()

        for user_id in test_users:
            recommendations = recommendations_by_user[user_id]
            top_k = recommendations[:k]
            relevant_items = relevant_by_user.get(user_id, set())

            precisions.append(
                precision_at_k(recommendations, relevant_items, k)
            )

            completions.append(min(len(top_k), k) / k)

            if len(top_k) == k:
                users_with_full_lists += 1

            if not top_k:
                empty_users += 1

            if len(top_k) < k:
                incomplete_users += 1

            recommended_movies.update(top_k)

            # Recall and NDCG are undefined when the test set has
            # no relevant items for this user.
            if relevant_items:
                recalls.append(
                    recall_at_k(recommendations, relevant_items, k)
                )
                ndcgs.append(
                    ndcg_at_k(recommendations, relevant_items, k)
                )

        metrics[k] = {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)) if recalls else None,
            "ndcg": float(np.mean(ndcgs)) if ndcgs else None,
            "catalog_coverage": (
                100.0 * len(recommended_movies) / len(catalog)
                if catalog else 0.0
            ),
            "completion": float(np.mean(completions)),
            "full_list_rate": users_with_full_lists / len(test_users),
            "users": {
                "precision": len(precisions),
                "recall": len(recalls),
                "ndcg": len(ndcgs),
                "completion": len(completions),
            },
        }

        diagnostics_by_k[k] = {
            "empty_users": empty_users,
            "incomplete_users": incomplete_users,
            "users_with_full_lists": users_with_full_lists,
        }

    reachable_movies = {
        movie_id
        for recommendations in recommendations_by_user.values()
        for movie_id in recommendations
    }

    return {
        "metrics": metrics,
        "population": {
            "original_users": int(catalog_source["UserID"].nunique())
            if full_ratings is not None
            else int(
                pd.concat([train, test])["UserID"].nunique()
            ),
            "users_evaluated": len(test_users),
            "users_with_relevant_test_items": len(
                users_with_relevant_items
            ),
            "precision_users": len(test_users),
            "recall_users": len(users_with_relevant_items),

            "ndcg_users": len(users_with_relevant_items),
            "catalog_size": len(catalog),
        },
        "diagnostics": {
            "by_k": diagnostics_by_k,
            "recommendation_errors": recommendation_errors,
            "reachable_catalog_size": len(reachable_movies),
            "reachability_rate": (
                100.0 * len(reachable_movies) / len(catalog)
                if catalog else 0.0
            ),
            "users_with_empty_recommendations": sum(
                not recommendations
                for recommendations in recommendations_by_user.values()
            ),
        },
    }

