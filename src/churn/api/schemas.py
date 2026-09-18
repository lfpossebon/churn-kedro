"""Pydantic contracts for the API.

These validate types and shapes only — no business rules. Keeping them here
means the request/response contract is readable in one place, and FastAPI
turns them into the OpenAPI docs served at /docs for free.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    production_model_available: bool = Field(
        description="Whether the refit pipeline has produced the production artefacts "
        "that /inference needs."
    )


class InferenceRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        min_length=1,
        description="One object per customer, using the raw column names "
        "(tenure, MonthlyCharges, Contract, ...). Unknown columns are ignored "
        "and missing ones are imputed by the cleaning node.",
        examples=[
            [
                {
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
            ]
        ],
    )


class Prediction(BaseModel):
    index: int
    prediction: int = Field(description="1 = churn, 0 = stays.")
    probability: float = Field(description="Probability of the positive (churn) class.")


class InferenceResponse(BaseModel):
    n: int
    predictions: list[Prediction]


class RunStartedResponse(BaseModel):
    run_id: str
    status: Literal["running"]
    pipelines: list[str]


class RunStatusResponse(BaseModel):
    run_id: str
    kind: str
    status: Literal["running", "completed", "failed"]
    pipelines: list[str]
    started_at: str
    finished_at: str | None = None
    error: str | None = None
