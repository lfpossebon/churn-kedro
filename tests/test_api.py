"""End-to-end tests for the API, driven through TestClient (no server needed)."""

import pytest
from fastapi.testclient import TestClient

from churn.api import app
from churn.api.service import PROJECT_PATH

CUSTOMER = {
    "tenure": 24,
    "MonthlyCharges": 90.35,
    "TotalCharges": 2238.5,
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

HTTP_OK = 200
HTTP_ACCEPTED = 202
HTTP_NOT_FOUND = 404
HTTP_UNPROCESSABLE = 422

N_INSTANCES = 2


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_reports_ok_and_artefact_availability(self, client):
        response = client.get("/health")

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["status"] == "ok"
        assert isinstance(body["production_model_available"], bool)


class TestOnlineInference:
    def test_scores_every_instance(self, client):
        response = client.post("/inference", json={"instances": [CUSTOMER, CUSTOMER]})

        assert response.status_code == HTTP_OK
        body = response.json()
        assert body["n"] == N_INSTANCES
        assert [p["index"] for p in body["predictions"]] == [0, 1]
        for item in body["predictions"]:
            assert item["prediction"] in (0, 1)
            assert 0.0 <= item["probability"] <= 1.0

    def test_identical_inputs_score_identically(self, client):
        """Cheap determinism check: no state leaks between rows or requests."""
        body = client.post("/inference", json={"instances": [CUSTOMER, CUSTOMER]}).json()

        first, second = body["predictions"]
        assert first["probability"] == second["probability"]

    def test_long_contract_churns_less_than_month_to_month(self, client):
        """The strongest signal in the EDA must survive the whole pipeline."""
        loyal = {**CUSTOMER, "Contract": "Two year", "tenure": 60}
        body = client.post("/inference", json={"instances": [CUSTOMER, loyal]}).json()

        monthly_prob, loyal_prob = (p["probability"] for p in body["predictions"])
        assert loyal_prob < monthly_prob

    def test_empty_payload_is_rejected(self, client):
        response = client.post("/inference", json={"instances": []})

        assert response.status_code == HTTP_UNPROCESSABLE

    def test_nothing_is_written_to_disk(self, client):
        """The whole point of the MemoryDataset overrides."""
        watched = [
            PROJECT_PATH / "data" / "02_intermediate" / "cleaned_telco-churn-inference.csv",
            PROJECT_PATH / "data" / "04_feature" / "encoded_telco-churn-inference.csv",
            PROJECT_PATH / "data" / "05_model_input" / "scaled_telco-churn-inference.csv",
            PROJECT_PATH / "data" / "07_model_output" / "inference_predictions.json",
        ]
        before = {path: path.stat().st_mtime_ns if path.exists() else None for path in watched}

        client.post("/inference", json={"instances": [CUSTOMER]})

        after = {path: path.stat().st_mtime_ns if path.exists() else None for path in watched}
        assert before == after


class TestTrainingRuns:
    def test_unknown_run_id_is_404(self, client):
        response = client.get("/train/does-not-exist")

        assert response.status_code == HTTP_NOT_FOUND

    def test_train_returns_a_run_id_immediately(self, client, mocker):
        """Must return without waiting for the grid search to finish.

        The worker is stubbed out: running the real pipelines here would write
        to data/ from a daemon thread that pytest kills on exit, which can
        leave half-written CSVs behind. What is under test is the HTTP
        contract, not the pipelines — those have their own tests.
        """
        worker = mocker.patch("churn.api.service._run_pipelines")

        response = client.post("/train")

        assert response.status_code == HTTP_ACCEPTED
        body = response.json()
        assert body["status"] == "running"
        assert body["pipelines"] == ["data_engineering", "modelling", "refit"]

        # The run is registered and queryable straight away.
        status_response = client.get(f"/train/{body['run_id']}")
        assert status_response.status_code == HTTP_OK
        assert status_response.json()["kind"] == "train"

        worker.assert_called_once()

    def test_batch_inference_starts_the_inference_pipeline(self, client, mocker):
        worker = mocker.patch("churn.api.service._run_pipelines")

        response = client.post("/batch-inference")

        assert response.status_code == HTTP_ACCEPTED
        assert response.json()["pipelines"] == ["inference"]
        worker.assert_called_once()
