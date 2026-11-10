
import pandas as pd
import pytest

from recommendation_engine.models.collaborative import (
    ItemBasedCollaborativeFilter,
)


@pytest.fixture
def api_model_path(tmp_path, monkeypatch):
    ratings = pd.DataFrame(
        [
            (44, 1, 5.0),
            (10, 1, 5.0),
            (10, 2, 4.0),
            (10, 3, 4.0),
            (10, 4, 4.0),
            (10, 5, 4.0),
            (10, 6, 4.0),
            (11, 1, 5.0),
            (11, 2, 3.0),
            (11, 3, 3.0),
            (11, 4, 3.0),
            (11, 5, 3.0),
            (11, 6, 3.0),
        ],
        columns=["UserID", "MovieID", "Rating"],
    )

    model = ItemBasedCollaborativeFilter(
        n_neighbours=5,
        min_co_ratings=2,
    )
    model.fit(ratings)

    model_path = tmp_path / "cf_model.joblib"
    model.save(model_path)

    monkeypatch.setenv("MODEL_PATH", str(model_path))

    return model_path
