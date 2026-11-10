# Recommendation Engine

A movie recommendation engine built with Python, featuring item-based collaborative filtering, offline evaluation, MLflow experiment tracking, model persistence, and a Dockerized FastAPI inference service.

The project demonstrates practical machine learning engineering workflows, from model development and evaluation to experiment tracking, testing, model serving, and containerization.

## Project Overview

This project uses the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset to build and evaluate a movie recommendation model.

The current implementation uses **item-based collaborative filtering**. It identifies movies with similar rating patterns and recommends movies a user has not rated yet.

The focus is not only on generating recommendations, but also on measuring model performance, tracking experiments, validating behavior through tests, persisting trained models, and serving recommendations through an API.

## Current Features

- Item-based collaborative filtering
- Data validation and duplicate-rating aggregation
- Top-N recommendations for known users
- Offline model evaluation
- Precision@K, Recall@K, and NDCG@K
- Catalog coverage and recommendation-completion diagnostics
- MLflow experiment tracking
- Model serialization and loading using joblib
- FastAPI inference API
- API and model tests using pytest
- Docker image creation and containerized deployment
- Docker Compose application management

## Tech Stack

| Technology | Purpose |
| --- | --- |
| Python | Core implementation |
| Pandas and NumPy | Data processing |
| scikit-learn | Supporting machine learning utilities |
| SciPy | Sparse matrix operations |
| joblib | Model serialization and loading |
| MLflow | Experiment tracking |
| FastAPI | Model serving and API endpoints |
| Uvicorn | ASGI server |
| Docker | Application containerization |
| Docker Compose | Container lifecycle management |
| pytest | Automated testing |

## Dataset

The project uses the MovieLens 1M movie-rating dataset.

The dataset contains approximately one million ratings from 6,040 users across 3,706 movies.

The evaluation workflow separates ratings into training and test sets so recommendations can be assessed against held-out user preferences.

## Project Structure

```text
recommendation-engine/
├── artifacts/
│   └── cf_model.joblib
├── data/
├── notebooks/
├── scripts/
│   └── run_baseline.py
├── src/
│   └── recommendation_engine/
│       ├── api/
│       │   └── main.py
│       ├── evaluation.py
│       └── models/
│           └── collaborative.py
├── tests/
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-docker.txt
├── pyproject.toml
└── README.md
```

The structure highlights the main application components. Additional test files, configuration files, and supporting modules are also present.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Moazzam-Matin/recommendation-engine.git
cd recommendation-engine
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Ensure the MovieLens dataset is available in the location expected by the training script.

### 4. Run the tests

```bash
pytest -q
```

The test suite covers model behavior, evaluation metrics, validation, model persistence, MLflow tracking, and API functionality.

## Running the Baseline

Run the baseline evaluation and MLflow tracking script from the project root:

```bash
python scripts/run_baseline.py
```

The script evaluates the configured recommendation model and records experiment information in MLflow.

The trained model is serialized to:

```text
artifacts/cf_model.joblib
```

## Viewing MLflow Experiments

Start the MLflow UI:

```bash
mlflow ui
```

Open the address displayed in your terminal, typically:

```text
http://127.0.0.1:5000
```

The UI lets you inspect recorded runs, compare parameters and metrics, and review experiment results.

The tracking location depends on the project's MLflow configuration.

## Model Evaluation

The evaluation workflow measures recommendation quality at different values of K.

| Metric | What it measures |
| --- | --- |
| Precision@K | Proportion of recommended items that are relevant |
| Recall@K | Proportion of relevant held-out items that are recommended |
| NDCG@K | Recommendation quality while accounting for ranking position |
| Catalog coverage | Proportion of the item catalog recommended across users |

Additional diagnostics examine recommendation-list completion and how much of the catalog is reachable by the model.

### Current Baseline Results

The current baseline uses the following configuration:

| Parameter | Value |
| --- | ---: |
| `n_neighbours` | 5 |
| `min_co_ratings` | 2 |

Recorded evaluation results:

| Metric | K = 5 | K = 10 | K = 20 |
| --- | ---: | ---: | ---: |
| Precision@K | 0.0616 | 0.0587 | 0.0624 |
| Recall@K | 0.0253 | 0.0500 | 0.1061 |
| NDCG@K | 0.0675 | 0.0707 | 0.0899 |
| Catalog coverage | 49.43% | 62.01% | 71.40% |

These results are a reference point for the current evaluation setup, not a claim that the model is production-ready or universally optimal.

## Running the Inference API

The project includes a FastAPI application that loads the persisted model during application startup.

First, ensure the trained model exists at:

```text
artifacts/cf_model.joblib
```

Start the API locally:

```bash
uvicorn recommendation_engine.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

### API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Check API and model readiness |
| `/recommend/{user_id}` | GET | Generate movie recommendations |
| `/docs` | GET | Interactive Swagger API documentation |

Example recommendation request:

```text
http://localhost:8000/recommend/50?n=5
```

Example health check:

```text
http://localhost:8000/health
```

## Running with Docker

Docker packages the application, its runtime dependencies, and the serialized model into a container image.

### Prerequisites

- Docker Desktop or Docker Engine
- Docker Compose

### Build the Docker image

From the project root:

```powershell
docker build -t recommendation-engine:0.1.0 .
```

### Run the container

```powershell
docker run -d --name re-api -p 8000:8000 recommendation-engine:0.1.0
```

### Verify the API

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Recommendations:

```powershell
Invoke-RestMethod "http://localhost:8000/recommend/50?n=5"
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### View container logs

```powershell
docker logs re-api
```

### Stop and remove the container

```powershell
docker stop re-api
docker rm re-api
```

## Running with Docker Compose

Docker Compose provides a convenient way to build, start, and manage the application.

Build the image and start the application:

```powershell
docker compose up -d --build
```

Check the running services:

```powershell
docker compose ps
```

View application logs:

```powershell
docker compose logs api
```

Stop the application without removing its containers:

```powershell
docker compose stop
```

Start existing stopped containers:

```powershell
docker compose start
```

Stop and remove the containers and default network:

```powershell
docker compose down
```

The Docker image and local project files remain available after running `docker compose down`.

## Testing

Automated tests cover model behavior, evaluation metrics, validation errors, model persistence, MLflow tracking, and API functionality.

Run the full test suite:

```bash
pytest -q
```

## Current Limitations

- The current recommender uses item-based collaborative filtering only.
- Recommendations depend on the rating data available to the model.
- Evaluation results depend on the dataset, train/test split, and relevance definition.
- The current API loads a serialized model during application startup.
- Operational monitoring and production deployment infrastructure are not yet implemented.

## Roadmap

Potential future improvements include:

- CI/CD pipeline for automated testing and image builds
- API performance and operational monitoring
- Cloud deployment
- Model versioning and deployment management
- Improved production-readiness and observability

## Author **--Moazzam Matin--**

- GitHub: [Moazzam-Matin](https://github.com/Moazzam-Matin)
- Repository: [recommendation-engine](https://github.com/Moazzam-Matin/recommendation-engine)
