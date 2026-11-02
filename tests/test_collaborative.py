import pandas as pd
import pytest

from recommendation_engine.models.collaborative import (
    ItemBasedCollaborativeFilter,
    NotFittedError,
)


@pytest.fixture
def ratings():
    return pd.DataFrame(
        [
            [1, 101, 5],
            [1, 102, 4],
            [1, 103, 2],

            [2, 101, 5],
            [2, 102, 4],
            [2, 103, 2],

            [3, 101, 4],
            [3, 102, 5],
            [3, 104, 5],

            [4, 101, 5],
            [4, 102, 4],
            [4, 104, 4],

            [5, 103, 5],
            [5, 104, 4],
        ],
        columns=["UserID", "MovieID", "Rating"],
    )


@pytest.fixture
def model():
    return ItemBasedCollaborativeFilter(
        n_neighbours=2,
        min_co_ratings=2,
    )


def test_fit_returns_self(model, ratings):
    result = model.fit(ratings)

    assert result is model
    assert model._is_fitted is True


def test_fit_builds_similarity_relationships(model, ratings):
    model.fit(ratings)

    assert 101 in model._similar_items
    assert 102 in model._similar_items
    assert 103 in model._similar_items
    assert 104 in model._similar_items

    for similar_items in model._similar_items.values():
        assert len(similar_items) <= 2


def test_min_co_ratings_filters_relationships(model, ratings):
    model.fit(ratings)

    similar_to_103 = dict(model._similar_items[103])

    assert 104 not in similar_to_103


def test_recommend_returns_expected_movies(model, ratings):
    model.fit(ratings)

    recommendations = model.recommend(user_id=1, n=2)

    assert recommendations == [104]


def test_recommend_returns_available_candidates(model, ratings):
    model.fit(ratings)

    recommendations = model.recommend(user_id=5, n=10)

    assert recommendations == [101, 102]


def test_recommend_rejects_unknown_user(model, ratings):
    model.fit(ratings)

    with pytest.raises(ValueError, match="Unknown user id: 999"):
        model.recommend(user_id=999, n=2)


def test_recommend_rejects_invalid_n(model, ratings):
    model.fit(ratings)

    with pytest.raises(ValueError, match="n must be greater than 0"):
        model.recommend(user_id=1, n=0)

    with pytest.raises(TypeError, match="n must be an integer"):
        model.recommend(user_id=1, n="1")

    with pytest.raises(TypeError, match="n must be an integer"):
        model.recommend(user_id=1, n=True)


def test_recommend_requires_fitted_model(model):
    with pytest.raises(
        NotFittedError,
        match="The model must be fitted before generating recommendations",
    ):
        model.recommend(user_id=1, n=2)


def test_fit_averages_duplicate_user_movie_ratings(model, ratings):
    duplicate_ratings = pd.concat(
        [
            ratings,
            pd.DataFrame(
                [[1, 101, 3]],
                columns=["UserID", "MovieID", "Rating"],
            ),
        ],
        ignore_index=True,
    )

    model.fit(duplicate_ratings)

    rating = model._ratings.loc[
        (model._ratings["UserID"] == 1)
        & (model._ratings["MovieID"] == 101),
        "Rating",
    ].iloc[0]

    assert rating == 4.0

def test_save_and_load_model(model, ratings, tmp_path):
    model.fit(ratings)

    model_path = tmp_path / "cf_model.joblib"

    model.save(model_path)

    loaded_model = ItemBasedCollaborativeFilter.load(model_path)

    assert isinstance(loaded_model, ItemBasedCollaborativeFilter)
    assert loaded_model._is_fitted is True

def test_loaded_model_produces_same_recommendations(model, ratings, tmp_path):
    model.fit(ratings)

    original_recommendations = model.recommend(user_id=1, n=2)

    model_path = tmp_path / "cf_model.joblib"
    model.save(model_path)

    loaded_model = ItemBasedCollaborativeFilter.load(model_path)

    loaded_recommendations = loaded_model.recommend(user_id=1, n=2)

    assert loaded_recommendations == original_recommendations

def test_save_requires_fitted_model(model, tmp_path):
    model_path = tmp_path / "cf_model.joblib"

    with pytest.raises(
        NotFittedError,
        match="The model must be fitted before it can be saved",
    ):
        model.save(model_path)


def test_load_rejects_wrong_object_type(tmp_path):
    import joblib

    model_path = tmp_path / "wrong_model.joblib"

    joblib.dump({"model": "not a recommender"}, model_path)

    with pytest.raises(
        TypeError,
        match="The saved model is not a ItemBasedCollaborativeFilter model",
    ):
        ItemBasedCollaborativeFilter.load(model_path)


def test_load_rejects_unfitted_model(model, tmp_path):
    model_path = tmp_path / "unfitted_model.joblib"

    import joblib
    joblib.dump(model, model_path)

    with pytest.raises(
        NotFittedError,
        match="The saved model is not fitted",
    ):
        ItemBasedCollaborativeFilter.load(model_path)
