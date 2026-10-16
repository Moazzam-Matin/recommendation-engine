# Recommendation Engine

A recommendation engine built with Python, featuring item-based collaborative filtering, offline evaluation, and MLflow experiment tracking.

The project demonstrates practical machine learning engineering workflows, from model development and evaluation to experiment tracking, testing, and preparation for deployment.

## Project Overview

This project uses the MovieLens 1M dataset to build and evaluate a movie recommendation model.

The current implementation uses **item-based collaborative filtering**. It identifies items with similar rating patterns and recommends movies a user has not rated yet.

The focus is not only on generating recommendations, but also on measuring model performance, tracking experiments, validating behavior through tests, and building toward a deployable ML system.

## Current Features

- Item-based collaborative filtering
- Data validation and duplicate-rating aggregation
- Top-N recommendations for known users
- Offline model evaluation
- Precision@K, Recall@K, and NDCG@K
- Catalog coverage and recommendation-completion diagnostics
- MLflow experiment tracking for the baseline run
- Automated tests using pytest

## Tech Stack

- **Python** — core implementation
- **Pandas and NumPy** — data processing
- **scikit-learn** — supporting machine learning utilities
- **MLflow** — experiment tracking
- **pytest** — automated testing

FastAPI, Redis, PostgreSQL, SQLAlchemy, and Prometheus client are included in the project's dependencies for potential serving, storage, caching, and monitoring work. Their inclusion does not mean those integrations are already implemented.

## Dataset

The project uses the **MovieLens 1M** movie-rating dataset.

The dataset contains approximately one million ratings from 6,040 users across 3,706 movies.

The evaluation workflow separates ratings into training and test sets so that recommendations can be assessed against held-out user preferences.

## Project Structure

```text
recommendation-engine/
├── data/
├── notebooks/
├── scripts/
│   └── run_baseline.py
├── src/
│   └── recommendation_engine/
│       ├── evaluation.py
│       └── models/
│           └── collaborative.py
├── tests/
├── requirements.txt
└── README.md
```

The structure above highlights the main project components; additional test files and supporting modules may also be present.

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
pip install -r requirements.txt
pip install pytest
```

### 4. Run the tests

```bash
pytest -q
```

The project had **76 passing tests** at the time of the v0.1.1 release.

## Running the Baseline

Run the baseline evaluation and MLflow tracking script from the project root:

```bash
python scripts/run_baseline.py
```

The script evaluates the configured recommendation model and records experiment information in MLflow.

Ensure the dataset is available in the location expected by the script before running it.

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
|---|---|
| Precision@K | The proportion of recommended items that are relevant |
| Recall@K | The proportion of relevant held-out items that are recommended |
| NDCG@K | Recommendation quality while accounting for ranking position |
| Catalog coverage | The proportion of the item catalog recommended across users |

Additional diagnostics examine whether users receive complete recommendation lists and how much of the catalog is reachable by the model.

### Current Baseline Results

The current baseline uses the following configuration:

| Parameter | Value |
|---|---:|
| `n_neighbours` | 5 |
| `min_co_ratings` | 2 |

In the recorded evaluation, the model produced these results:

| Metric | K = 5 | K = 10 | K = 20 |
|---|---:|---:|---:|
| Precision@K | 0.0616 | 0.0587 | 0.0624 |
| Recall@K | 0.0253 | 0.0500 | 0.1061 |
| NDCG@K | 0.0675 | 0.0707 | 0.0899 |
| Catalog coverage | 49.43% | 62.01% | 71.40% |

These results are a reference point for the current evaluation setup, not a claim that the model is production-ready or universally optimal.

## Testing

Automated tests cover model behavior, evaluation metrics, validation errors, and MLflow tracking.

Run the full test suite with:

```bash
pytest -q
```

## Current Limitations

- The current recommender is based on collaborative filtering.
- Recommendations are limited by the rating data available to the model.
- Evaluation results depend on the dataset, train/test split, and relevance definition.
- A production inference API and deployment workflow remain future work.
- Persistent model saving and loading, along with operational monitoring, are planned extensions.

## Roadmap

Potential next steps include:

- Model persistence and loading
- A FastAPI inference endpoint
- API and integration tests
- Continuous integration improvements
- Monitoring and deployment workflows

These are planned extensions rather than claims about current functionality.

## Author

**Moazzam Matin**

- GitHub: [Moazzam-Matin](https://github.com/Moazzam-Matin)
- Repository: [recommendation-engine](https://github.com/Moazzam-Matin/recommendation-engine)
