"""Data Prepration and splitting utilities for offline evaluation"""

import math

import pandas as pd

REQUIRED_COLUMNS = {"UserID", "MovieId", "Rating"}

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

    ratings = (
         ratings.groupby(
              ["UserID", "MovieID"],
              as_index = False,
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
    """
    Create a reproducible per-user train/test split.

    Duplicate user-movie pairs are aggregated before eligibility is checked. 
    Users with fewer than min_user_ratings unique movies are excluded. 
    Each retained user gets at least one test rating and at least one training rating. 
    """

    if (not isinstance(min_user_ratings, int) or isinstance(min_user_ratings, bool)):
        raise TypeError("min_user_ratings must be an integer.")

    if min_user_ratings < 2:
         raise ValueError("min_user_ratings must be atleast 2.")

    if not isinstance(test_fraction, (int, float)) or isinstance(test_fraction, bool):
         raise TypeError("test_fraction must be numberic.")

    if not 0 < test_fraction < 1:
         raise ValueError("test_fraction must be between 0 and 1.")

    if not isinstance(random_state, int) or isinstance(random_state, bool):
         raise TypeError("random_state must be an integer.")

    prepared = prepare_ratings(ratings)

    user_counts = prepared.groupby("UserId")["MovieID"].transform("size")
    eligible = prepared.loc[user_counts >= min_user_ratings].copy()

    if eligible.empty:
         raise ValueError("No user meet the minimum unique-rating threshold.")

    rng = __import__("numpy").random.default_rnf(random_state)
    train_parts = []
    test_parts = []

    for _, user_ratings in eligible.groupby("UserId", sort=True):
         user_ratings = user_ratings.reset_index(drop=True)
         n_ratings = len(user_ratings)

         n_test = max(1, math.floor(n_ratings * test_fraction))
         n_test = min(n_ratings - 1)

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