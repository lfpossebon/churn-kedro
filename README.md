# churn

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.6.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)

---

## Pipelines

Four pipelines, run in dependency order by a bare `kedro run`:

| Pipeline | What it does | Key outputs |
|---|---|---|
| `data_engineering` | clean → split → fit/transform encoders → fit/transform scalers | `master_table`, `modelling_encoders`, `modelling_scalers` |
| `modelling` | baseline (LogisticRegression) and grid search (CatBoost), both scored by one `evaluate_model` | `baseline_metrics`, `optimized_metrics` |
| `refit` | refits the same artefacts on every split, for production | `production_encoders`, `production_scalers`, `production_model` |
| `inference` | transform-only scoring with the production artefacts | `inference_predictions` |

The `fit_*` nodes learn only from the splits named in `split_to_fit` (just `train` during modelling); `transform_*` applies to every row. That separation is what keeps evaluation honest, and it is why `refit` and `inference` reuse the data_engineering nodes with different parameters instead of duplicating them.

```bash
uv sync
uv run kedro run                          # all four
uv run kedro run --pipeline=inference     # just one
uv run kedro viz                          # inspect the DAG
```

## API

```bash
uv run uvicorn churn.api:app --host 0.0.0.0 --port 8000
```

Interactive docs at `/docs`.

| Method | Route | Behaviour |
|---|---|---|
| `GET` | `/health` | Status plus whether the production artefacts exist |
| `POST` | `/train` | Starts data_engineering → modelling → refit in the background, returns a `run_id` |
| `GET` | `/train/{run_id}` | `running` / `completed` / `failed` for that run |
| `POST` | `/inference` | Synchronous scoring from JSON |
| `POST` | `/batch-inference` | Scores the catalog's inference file in the background |

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/inference \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"tenure":24,"MonthlyCharges":90.35,"TotalCharges":2238.5,
       "gender":"Male","SeniorCitizen":"0","Partner":"No","Dependents":"No",
       "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
       "TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes",
       "Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check"}]}'
```

Two design points worth knowing before changing this code:

- **Every handler is a plain `def`, never `async def`.** FastAPI sends plain functions to a worker thread, keeping the event loop free. An `async def` with no `await` would run the CPU-bound pipeline on the loop and block the whole server.
- **`/inference` writes nothing to disk.** The request payload is injected into the catalog as a `MemoryDataset` and every intermediate is overridden the same way, so the exact same `inference` pipeline serves both batch and online. Without those overrides, concurrent requests would write over each other's CSVs.

## Docker

```bash
docker compose up --build -d
curl -s localhost:8000/health
docker compose logs -f api
docker compose down
```

`conf/` is mounted read-only and `data/` read-write, so configuration and data stay out of the image — the same image serves any environment by changing what is mounted. Change the left side of `"8000:8000"` in `docker-compose.yml` if that port is taken.

## Tests

```bash
uv run pytest          # unit tests for nodes, integration for the DAG, e2e for the API
uv run ruff check src tests
```

### macOS gotcha

`uv` writes the editable-install `.pth` files with the macOS `UF_HIDDEN` flag, and Python 3.14 skips hidden `.pth` files — so after any `uv sync` or `uv add`, `import churn` can fail with `ModuleNotFoundError` even though the package is installed. `kedro run` still works (Kedro puts `src/` on the path itself), which makes it confusing.

```bash
chflags nohidden .venv/lib/python*/site-packages/*.pth
```

The pytest config sets `pythonpath = ["src"]` so the test suite does not depend on that install.
