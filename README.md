# MLOps Demand Forecasting

### Caio G. B. Balieiro

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/WSL-0D1117?style=for-the-badge&logo=linux&logoColor=white" alt="WSL"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Compose"/>
  <img src="https://img.shields.io/badge/Apache_Airflow-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white" alt="Apache Airflow"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white" alt="MinIO"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions"/>
  <img src="https://img.shields.io/badge/GitHub_Container_Registry-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Container Registry"/>
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white" alt="Grafana"/>
  <img src="https://img.shields.io/badge/Evidently-6C63FF?style=for-the-badge&logoColor=white" alt="Evidently"/>
</p>

A complete end-to-end MLOps project for demand forecasting, developed on WSL using Python, Docker, Apache Airflow, MLflow, PostgreSQL, MinIO, FastAPI, automated testing, monitoring, and CI/CD with GitHub Actions.

## Project Objective

The objective of this project is to implement a complete Machine Learning lifecycle, covering data generation, data validation, feature engineering, model training, experiment tracking, model registration, deployment, monitoring, and continuous integration and delivery.

## Main Features

* Synthetic demand data generation
* Data validation and preprocessing
* Feature engineering
* Multiple Machine Learning model training
* Model evaluation and comparison
* Experiment tracking with MLflow
* Model versioning with MLflow Model Registry
* Workflow orchestration with Apache Airflow
* REST API for model inference using FastAPI
* Containerized infrastructure with Docker Compose
* Automated tests with Pytest
* Code quality checks with Ruff
* Continuous integration with GitHub Actions
* Docker image publishing with GitHub Container Registry
* API monitoring with Prometheus and Grafana
* Data and model drift monitoring with Evidently

## Technology Stack

| Category                | Technologies                       |
| ----------------------- | ---------------------------------- |
| Programming Language    | Python                             |
| Development Environment | WSL 2 and Ubuntu                   |
| Containerization        | Docker and Docker Compose          |
| Workflow Orchestration  | Apache Airflow                     |
| Experiment Tracking     | MLflow                             |
| Model Registry          | MLflow Model Registry              |
| Database                | PostgreSQL                         |
| Artifact Storage        | MinIO                              |
| Machine Learning        | Scikit-learn                       |
| Data Processing         | Pandas and NumPy                   |
| API                     | FastAPI and Uvicorn                |
| Testing                 | Pytest and Pytest Coverage         |
| Code Quality            | Ruff                               |
| Monitoring              | Prometheus, Grafana, and Evidently |
| CI/CD                   | GitHub Actions                     |
| Container Registry      | GitHub Container Registry          |

## Project Architecture

```text
                        GitHub Repository
                               |
                               v
                    GitHub Actions CI/CD
                               |
               +---------------+---------------+
               |                               |
               v                               v
        Automated Tests                Docker Image Build
               |                               |
               v                               v
        Code Quality Checks          GitHub Container Registry
                                               |
                                               v
                                      Docker Compose Environment
                                               |
         +------------------+------------------+------------------+
         |                  |                  |                  |
         v                  v                  v                  v
      Airflow             MLflow            FastAPI          Monitoring
         |                  |                  |                  |
         v                  v                  v                  v
   ML Pipeline       Experiment Tracking   Predictions     Prometheus/Grafana
         |
         v
 Data Generation -> Validation -> Feature Engineering -> Training
         |
         v
 Evaluation -> Model Registration -> Model Promotion -> Deployment
```

## Project Structure

```text
mlops-demand-forecasting/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker.yml
├── airflow/
│   ├── dags/
│   │   └── training_pipeline.py
│   ├── logs/
│   └── plugins/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── artifacts/
├── configs/
├── data/
│   ├── processed/
│   └── raw/
├── docker/
│   ├── airflow.Dockerfile
│   ├── api.Dockerfile
│   ├── mlflow.Dockerfile
│   ├── trainer.Dockerfile
│   └── postgres/
│       └── init.sql
├── monitoring/
│   ├── grafana/
│   └── prometheus.yml
├── scripts/
├── src/
│   ├── data/
│   │   ├── generate_data.py
│   │   └── validation.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   └── monitoring/
│       └── drift.py
├── tests/
│   ├── integration/
│   ├── unit/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Local Environment Setup

Navigate to the projects directory:

```bash
cd ~/projects
```

Enter the project folder:

```bash
cd mlops-demand-forecasting
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Upgrade the Python package manager:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project dependencies:

```bash
pip install -r requirements-dev.txt
```

## Environment Variables

Create a local environment file:

```bash
cp .env.example .env
```

Set the Airflow user ID according to the current WSL user:

```bash
sed -i "s/AIRFLOW_UID=50000/AIRFLOW_UID=$(id -u)/" .env
```

## Running the Project

Build the Docker images:

```bash
docker compose build
```

Start all services:

```bash
docker compose up -d
```

Check the running containers:

```bash
docker compose ps
```

View the service logs:

```bash
docker compose logs -f
```

Stop the services:

```bash
docker compose down
```

Stop the services and remove the associated volumes:

```bash
docker compose down -v
```

## Available Services

After starting the Docker Compose environment, the services will be available at:

| Service            | Address                       |
| ------------------ | ----------------------------- |
| FastAPI            | `http://localhost:8000`       |
| FastAPI Swagger UI | `http://localhost:8000/docs`  |
| FastAPI ReDoc      | `http://localhost:8000/redoc` |
| Apache Airflow     | `http://localhost:8080`       |
| MLflow             | `http://localhost:5000`       |
| MinIO API          | `http://localhost:9000`       |
| MinIO Console      | `http://localhost:9001`       |
| Prometheus         | `http://localhost:9090`       |
| Grafana            | `http://localhost:3000`       |

## Machine Learning Pipeline

The training pipeline includes the following stages:

```text
Data Generation
      |
      v
Data Validation
      |
      v
Data Preprocessing
      |
      v
Feature Engineering
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Experiment Tracking
      |
      v
Model Registration
      |
      v
Model Promotion
      |
      v
API Deployment
      |
      v
Model Monitoring
```

## MLflow

MLflow is used to track:

* Experiments
* Training runs
* Model parameters
* Evaluation metrics
* Model artifacts
* Dataset information
* Registered model versions
* Candidate and champion model aliases

The MLflow interface is available at:

```text
http://localhost:5000
```

## Apache Airflow

Apache Airflow orchestrates the Machine Learning pipeline, including:

* Data generation
* Data validation
* Feature engineering
* Model training
* Model evaluation
* Model registration
* Model promotion
* Monitoring report generation

The Airflow interface is available at:

```text
http://localhost:8080
```

## FastAPI

The FastAPI application exposes the trained model through REST endpoints.

Main endpoints:

```text
GET  /health
GET  /ready
GET  /model-info
GET  /metrics
POST /predict
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Running Tests

Run all automated tests:

```bash
pytest
```

Run tests with code coverage:

```bash
pytest --cov=src --cov=api
```

Generate an HTML coverage report:

```bash
pytest --cov=src --cov=api --cov-report=html
```

Open the generated report:

```bash
code htmlcov/index.html
```

## Code Quality

Run code quality checks:

```bash
ruff check .
```

Automatically fix supported issues:

```bash
ruff check . --fix
```

Format the project source code:

```bash
ruff format .
```

## Makefile Commands

The project provides commands to simplify local development:

```bash
make build
make up
make down
make restart
make ps
make logs
make generate
make train
make test
make lint
make format
make clean
```

## Git Workflow

Create a new feature branch:

```bash
git checkout -b feature/feature-name
```

Add the modified files:

```bash
git add .
```

Create a commit:

```bash
git commit -m "feat: add feature description"
```

Push the branch to GitHub:

```bash
git push -u origin feature/feature-name
```

## CI/CD Pipeline

The GitHub Actions pipeline is responsible for:

1. Checking out the repository
2. Setting up Python
3. Installing project dependencies
4. Running Ruff checks
5. Running automated tests
6. Generating the test coverage report
7. Building Docker images
8. Scanning images for vulnerabilities
9. Publishing images to GitHub Container Registry
10. Executing application health checks

## Development Roadmap

* [x] Create the initial project structure
* [ ] Implement synthetic data generation
* [ ] Implement data validation
* [ ] Implement feature engineering
* [ ] Implement model training
* [ ] Configure MLflow Tracking
* [ ] Configure MLflow Model Registry
* [ ] Configure PostgreSQL
* [ ] Configure MinIO
* [ ] Create the Apache Airflow training pipeline
* [ ] Create the FastAPI inference service
* [ ] Add automated tests
* [ ] Configure GitHub Actions
* [ ] Publish Docker images to GHCR
* [ ] Configure Prometheus
* [ ] Configure Grafana
* [ ] Implement model drift monitoring with Evidently
* [ ] Document the complete MLOps workflow

## Repository

```text
https://github.com/caiogbb/mlops-demand-forecasting
```

## Author

**Caio G. B. Balieiro**

Data Scientist, Artificial Intelligence Specialist, Statistician, M.Sc. in Statistics, and Ph.D. Candidate.

* GitHub: [caiogbb](https://github.com/caiogbb)
* LinkedIn: [Caio Balieiro](https://www.linkedin.com/in/caio-balieiro-m-sc-a0711b188/)

## License

This project is intended for educational, research, and portfolio purposes.
