
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from recommendation_engine.evaluation import (
    evaluate_model,
    ndcg_at_k,
    precision_at_k,
    prepare_ratings,
    recall_at_k,
    split_ratings,
)

# --------------------------------------------------
# prepare_ratings() tests
# --------------------------------------------------

def test_prepare_ratings_averages_duplicate_user_movie_pairs():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 1, 2],
            "MovieID": [101, 101, 102, 101],
            "Rating": [4.0, 2.0, 5.0, 3.0],
        }
    )

    result = prepare_ratings(ratings)

    duplicate_rating = result.loc[
        (result["UserID"] == 1) & (result["MovieID"] == 101),
        "Rating",
    ].iloc[0]

    assert duplicate_rating == 3.0
    assert len(result) == 3


def test_prepare_ratings_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        prepare_ratings([1, 2, 3])


def test_prepare_ratings_rejects_missing_user_id():
    ratings = pd.DataFrame(
        {"MovieID": [101], "Rating": [4.0]}
    )

    with pytest.raises(ValueError, match="UserID"):
        prepare_ratings(ratings)


def test_prepare_ratings_rejects_missing_movie_id():
    ratings = pd.DataFrame(
        {"UserID": [1], "Rating": [4.0]}
    )

    with pytest.raises(ValueError, match="MovieID"):
        prepare_ratings(ratings)


def test_prepare_ratings_rejects_missing_rating():
    ratings = pd.DataFrame(
        {"UserID": [1], "MovieID": [101]}
    )

    with pytest.raises(ValueError, match="Rating"):
        prepare_ratings(ratings)


def test_prepare_ratings_rejects_empty_dataframe():
    ratings = pd.DataFrame(
        columns=["UserID", "MovieID", "Rating"]
    )

    with pytest.raises(ValueError, match="empty"):
        prepare_ratings(ratings)


def test_prepare_ratings_rejects_missing_values():
    ratings = pd.DataFrame(
        {
            "UserID": [1],
            "MovieID": [101],
            "Rating": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        prepare_ratings(ratings)


@pytest.mark.parametrize("invalid_rating", [0, 6, -1, 5.1])
def test_prepare_ratings_rejects_ratings_outside_valid_range(
    invalid_rating,
):
    ratings = pd.DataFrame(
        {
            "UserID": [1],
            "MovieID": [101],
            "Rating": [invalid_rating],
        }
    )

    with pytest.raises(ValueError, match="between 1 and 5"):
        prepare_ratings(ratings)


@pytest.mark.parametrize(
    "column",
    ["UserID", "MovieID"],
)
def test_prepare_ratings_rejects_non_numeric_identifiers(column):
    ratings = pd.DataFrame(
        {
            "UserID": [1],
            "MovieID": [101],
            "Rating": [4.0],
        }
    )
    ratings[column] = ["invalid"]

    with pytest.raises(TypeError, match="numeric"):
        prepare_ratings(ratings)


def test_prepare_ratings_rejects_non_numeric_ratings():
    ratings = pd.DataFrame(
        {
            "UserID": [1],
            "MovieID": [101],
            "Rating": ["excellent"],
        }
    )

    with pytest.raises(TypeError, match="numeric"):
        prepare_ratings(ratings)


def test_prepare_ratings_preserves_original_dataframe():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1],
            "MovieID": [101, 101],
            "Rating": [4.0, 2.0],
            "ExtraColumn": ["a", "b"],
        }
    )
    original = ratings.copy(deep=True)

    prepare_ratings(ratings)

    pd.testing.assert_frame_equal(ratings, original)


def test_prepare_ratings_returns_sorted_unique_pairs():
    ratings = pd.DataFrame(
        {
            "UserID": [2, 1, 1],
            "MovieID": [102, 103, 101],
            "Rating": [3.0, 4.0, 5.0],
        }
    )

    result = prepare_ratings(ratings)

    assert not result.duplicated(["UserID", "MovieID"]).any()
    assert list(zip(result["UserID"], result["MovieID"])) == [
        (1, 101),
        (1, 103),
        (2, 102),
    ]


# --------------------------------------------------
# split_ratings() tests
# --------------------------------------------------

def make_split_test_data():
    """Create a small dataset with eligible and ineligible users."""
    rows = []

    # User 1: 10 unique ratings; eligible.
    for movie_id in range(101, 111):
        rows.append((1, movie_id, 4.0))

    # User 2: 5 unique ratings; eligible.
    for movie_id in range(201, 206):
        rows.append((2, movie_id, 3.0))

    # User 3: 4 unique ratings; ineligible.
    for movie_id in range(301, 305):
        rows.append((3, movie_id, 5.0))

    return pd.DataFrame(
        rows,
        columns=["UserID", "MovieID", "Rating"],
    )


def test_split_ratings_excludes_users_below_minimum():
    ratings = make_split_test_data()

    train, test = split_ratings(ratings)

    retained_users = set(train["UserID"]) | set(test["UserID"])

    assert retained_users == {1, 2}


def test_split_ratings_keeps_train_and_test_for_each_eligible_user():
    ratings = make_split_test_data()

    train, test = split_ratings(ratings)

    for user_id in [1, 2]:
        assert user_id in set(train["UserID"])
        assert user_id in set(test["UserID"])


def test_split_ratings_has_no_user_movie_overlap():
    ratings = make_split_test_data()

    train, test = split_ratings(ratings)

    train_pairs = set(zip(train["UserID"], train["MovieID"]))
    test_pairs = set(zip(test["UserID"], test["MovieID"]))

    assert train_pairs.isdisjoint(test_pairs)


def test_split_ratings_preserves_all_eligible_pairs():
    ratings = make_split_test_data()

    train, test = split_ratings(ratings)

    actual_pairs = set(
        zip(
            pd.concat([train, test])["UserID"],
            pd.concat([train, test])["MovieID"],
        )
    )
    expected_pairs = set(
        zip(
            ratings.loc[ratings["UserID"].isin([1, 2]), "UserID"],
            ratings.loc[ratings["UserID"].isin([1, 2]), "MovieID"],
        )
    )

    assert actual_pairs == expected_pairs


def test_split_ratings_uses_expected_test_count():
    ratings = make_split_test_data()

    _, test = split_ratings(ratings)

    # User 1: floor(10 * 0.2) = 2 test ratings.
    # User 2: floor(5 * 0.2) = 1 test rating.
    test_counts = test.groupby("UserID").size().to_dict()

    assert test_counts == {1: 2, 2: 1}


def test_split_ratings_is_reproducible():
    ratings = make_split_test_data()

    train_a, test_a = split_ratings(ratings, random_state=42)
    train_b, test_b = split_ratings(ratings, random_state=42)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_split_ratings_changes_with_different_seed():
    ratings = make_split_test_data()

    _, test_a = split_ratings(ratings, random_state=42)
    _, test_b = split_ratings(ratings, random_state=123)

    assert set(zip(test_a["UserID"], test_a["MovieID"])) != set(
        zip(test_b["UserID"], test_b["MovieID"])
    )


def test_split_ratings_aggregates_duplicates_before_splitting():
    ratings = make_split_test_data()

    # Add a duplicate rating for an existing user-movie pair.
    duplicate = pd.DataFrame(
        {"UserID": [1], "MovieID": [101], "Rating": [2.0]}
    )
    ratings = pd.concat([ratings, duplicate], ignore_index=True)

    train, test = split_ratings(ratings)

    all_pairs = pd.concat([train, test])

    assert not all_pairs.duplicated(["UserID", "MovieID"]).any()
    assert len(all_pairs[all_pairs["UserID"] == 1]) == 10


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        (
            {"min_user_ratings": 1},
            ValueError,
            "at least 2",
        ),
        (
            {"min_user_ratings": 2.5},
            TypeError,
            "integer",
        ),
        (
            {"min_user_ratings": True},
            TypeError,
            "integer",
        ),
        (
            {"test_fraction": 0},
            ValueError,
            "between 0 and 1",
        ),
        (
            {"test_fraction": 1},
            ValueError,
            "between 0 and 1",
        ),
        (
            {"test_fraction": -0.1},
            ValueError,
            "between 0 and 1",
        ),
        (
            {"test_fraction": 1.1},
            ValueError,
            "between 0 and 1",
        ),
        (
            {"test_fraction": "0.2"},
            TypeError,
            "numeric",
        ),
        (
            {"test_fraction": True},
            TypeError,
            "numeric",
        ),
        (
            {"random_state": 42.5},
            TypeError,
            "integer",
        ),
        (
            {"random_state": True},
            TypeError,
            "integer",
        ),
    ],
)
def test_split_ratings_rejects_invalid_parameters(
    kwargs,
    error_type,
    message,
):
    ratings = make_split_test_data()

    with pytest.raises(error_type, match=message):
        split_ratings(ratings, **kwargs)


def test_split_ratings_rejects_no_eligible_users():
    ratings = pd.DataFrame(
        {
            "UserID": [1, 1, 2, 2],
            "MovieID": [101, 102, 201, 202],
            "Rating": [4.0, 3.0, 5.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="No users meet"):
        split_ratings(ratings)


def test_split_ratings_rejects_invalid_input():
    with pytest.raises(TypeError, match="DataFrame"):
        split_ratings([1, 2, 3])


def test_precision_at_k_counts_relevant_recommendations():
    recommendations = [101, 102, 103, 104, 105]
    relevant_items = {101, 103, 105}

    result = precision_at_k(recommendations, relevant_items, k=5)

    assert result == 3 / 5


def test_precision_at_k_uses_fixed_k_for_short_list():
    recommendations = [101, 102]
    relevant_items = {101, 103}

    result = precision_at_k(recommendations, relevant_items, k=5)

    assert result == 1 / 5


def test_precision_at_k_returns_zero_for_empty_recommendations():
    result = precision_at_k([], {101, 102}, k=5)

    assert result == 0.0


def test_precision_at_k_rejects_non_integer_k():
    import pytest

    with pytest.raises(TypeError, match="k must be an integer"):
        precision_at_k([101], {101}, k=2.5)


def test_precision_at_k_rejects_non_positive_k():
    import pytest

    with pytest.raises(ValueError, match="k must be greater than 0"):
        precision_at_k([101], {101}, k=0)

def test_recall_at_k_counts_relevant_items_found():
    recommendations = [101, 102, 103, 104, 105]
    relevant_items = {101, 103, 106, 107}

    result = recall_at_k(recommendations, relevant_items, k=5)

    assert result == 2 / 4


def test_recall_at_k_uses_only_top_k_recommendations():
    recommendations = [101, 102, 103, 104, 105]
    relevant_items = {105}

    result = recall_at_k(recommendations, relevant_items, k=3)

    assert result == 0.0


def test_recall_at_k_returns_one_when_all_relevant_items_are_found():
    recommendations = [101, 102, 103]
    relevant_items = {101, 103}

    result = recall_at_k(recommendations, relevant_items, k=3)

    assert result == 1.0


def test_recall_at_k_returns_zero_for_empty_recommendations():
    result = recall_at_k([], {101, 102}, k=5)

    assert result == 0.0


def test_recall_at_k_rejects_empty_relevant_items():
    import pytest

    with pytest.raises(ValueError, match="relevant_items must not be empty"):
        recall_at_k([101, 102], set(), k=2)


def test_recall_at_k_rejects_non_integer_k():
    import pytest

    with pytest.raises(TypeError, match="k must be an integer"):
        recall_at_k([101], {101}, k=2.5)


def test_recall_at_k_rejects_non_positive_k():
    import pytest

    with pytest.raises(ValueError, match="k must be greater than 0"):
        recall_at_k([101], {101}, k=0)

def test_ndcg_at_k_returns_one_for_perfect_ranking():
    recommendations = [101, 102, 103]
    relevant_items = {101, 102, 103}

    result = ndcg_at_k(recommendations, relevant_items, k=3)

    assert result == 1.0


def test_ndcg_at_k_rewards_relevant_items_ranked_higher():
    relevant_items = {101}

    high_rank = ndcg_at_k([101, 102, 103], relevant_items, k=3)
    low_rank = ndcg_at_k([102, 103, 101], relevant_items, k=3)

    assert high_rank > low_rank
    assert high_rank == 1.0


def test_ndcg_at_k_returns_zero_for_empty_recommendations():
    result = ndcg_at_k([], {101, 102}, k=5)

    assert result == 0.0


def test_ndcg_at_k_handles_fewer_recommendations_than_k():
    result = ndcg_at_k([101], {101, 102}, k=5)

    assert 0.0 < result < 1.0


def test_ndcg_at_k_rejects_empty_relevant_items():
    import pytest

    with pytest.raises(ValueError, match="relevant_items must not be empty"):
        ndcg_at_k([101], set(), k=5)


def test_ndcg_at_k_rejects_non_integer_k():
    import pytest

    with pytest.raises(TypeError, match="k must be an integer"):
        ndcg_at_k([101], {101}, k=2.5)


def test_ndcg_at_k_rejects_non_positive_k():
    import pytest

    with pytest.raises(ValueError, match="k must be greater than 0"):
        ndcg_at_k([101], {101}, k=0)


def make_evaluation_data():
    """Create a small dataset for testing model evaluation."""
    train = pd.DataFrame(
        [
            (1, 101, 5.0),
            (1, 102, 4.0),
            (2, 101, 4.0),
            (2, 103, 3.0),
        ],
        columns=["UserID", "MovieID", "Rating"],
    )

    test = pd.DataFrame(
        [
            (1, 103, 5.0),
            (1, 104, 2.0),
            (2, 102, 3.0),
            (2, 104, 2.0),
        ],
        columns=["UserID", "MovieID", "Rating"],
    )

    full = pd.concat([train, test], ignore_index=True)
    return train, test, full


def test_evaluate_model_fits_only_training_data():
    train, test, full = make_evaluation_data()
    model = Mock()
    model.recommend.return_value = [103, 104]

    evaluate_model(
        model,
        train,
        test,
        k_values=(1, 2),
        full_ratings=full,
    )

    fitted_data = model.fit.call_args.args[0]
    pd.testing.assert_frame_equal(
        fitted_data.reset_index(drop=True),
        train.reset_index(drop=True),
    )


def test_evaluate_model_returns_metrics_and_denominators():
    train, test, full = make_evaluation_data()
    model = Mock()
    model.recommend.return_value = [103, 104]

    result = evaluate_model(
        model,
        train,
        test,
        k_values=(1, 2),
        full_ratings=full,
    )

    assert set(result["metrics"]) == {1, 2}

    for k in (1, 2):
        metrics = result["metrics"][k]

        assert metrics["users"] == {
            "precision": 2,
            "recall": 1,
            "ndcg": 1,
            "completion": 2,
        }

    assert result["population"]["users_evaluated"] == 2
    assert result["population"]["users_with_relevant_test_items"] == 1


def test_evaluate_model_uses_full_catalog_for_coverage():
    train, test, full = make_evaluation_data()

    # Movie 999 is in the full catalog but not in this evaluation split.
    extra_movie = pd.DataFrame(
        [(99, 999, 5.0)],
        columns=["UserID", "MovieID", "Rating"],
    )
    full = pd.concat([full, extra_movie], ignore_index=True)

    model = Mock()
    model.recommend.return_value = [103]

    result = evaluate_model(
        model,
        train,
        test,
        k_values=(1,),
        full_ratings=full,
    )

    assert result["population"]["catalog_size"] == 5
    assert result["metrics"][1]["catalog_coverage"] == 20.0


def test_evaluate_model_records_recommendation_failures():
    train, test, full = make_evaluation_data()
    model = Mock()
    model.recommend.side_effect = ValueError("Unknown user")

    result = evaluate_model(
        model,
        train,
        test,
        k_values=(1,),
        full_ratings=full,
    )

    assert result["diagnostics"]["recommendation_errors"] == {
        "ValueError": 2
    }
    assert result["diagnostics"]["users_with_empty_recommendations"] == 2
    assert result["metrics"][1]["precision"] == 0.0
    assert result["metrics"][1]["users"]["precision"] == 2


def test_evaluate_model_rejects_overlapping_train_and_test():
    train, test, full = make_evaluation_data()
    overlapping_test = pd.concat(
        [test, train.iloc[[0]]],
        ignore_index=True,
    )

    model = Mock()

    with pytest.raises(ValueError, match="overlapping"):
        evaluate_model(
            model,
            train,
            overlapping_test,
            k_values=(1,),
            full_ratings=full,
        )

    model.fit.assert_not_called()


@pytest.mark.parametrize(
    "k_values",
    [(0,), (-1,), (), (1, 1)],
)
def test_evaluate_model_rejects_invalid_k_values(k_values):
    train, test, full = make_evaluation_data()

    with pytest.raises(ValueError):
        evaluate_model(
            Mock(),
            train,
            test,
            k_values=k_values,
            full_ratings=full,
        )


def test_evaluate_model_rejects_non_integer_k():
    train, test, full = make_evaluation_data()

    with pytest.raises(TypeError):
        evaluate_model(
            Mock(),
            train,
            test,
            k_values=(1.5,),
            full_ratings=full,
        )

