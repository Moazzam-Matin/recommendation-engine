from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from sklearn.exceptions import NotFittedError

from recommendation_engine.models.collaborative import (
    ItemBasedCollaborativeFilter,
    UnknownUserError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "artifacts" / "cf_model.joblib"

    app.state.model = ItemBasedCollaborativeFilter.load(model_path)

    print(f"Loaded model from: {model_path}")

    yield

app = FastAPI(
    title = "Recommendation Engine API",
    description = "An API for generating movie recommendations",
    version = "0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    model = getattr(app.state, "model", None)

    if model is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = Query(default=5, ge=1, le=100)):
    model = app.state.model

    try:
        recommendations = model.recommend(
            user_id=user_id,
            n=n,
        )
    except UnknownUserError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except NotFittedError as exc:
        raise HTTPException(
            status_code=500,
            detail="Recommendation model is not ready.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return{
        "user_id" : user_id,
        "recommendations" : recommendations,
    }