"""Pipeline 'refit': rebuild every artefact on the full dataset for production.

Same functions as data_engineering, different parameters — the encoders and
scalers are fitted on train + test + validate instead of train alone.
"""

from kedro.pipeline import Node, Pipeline

from churn.pipelines.data_engineering.nodes import fit_encoders, fit_scalers

from .nodes import refit_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=fit_encoders,
                inputs=["split_churn_data", "params:columns", "params:refit_encoders"],
                outputs="production_encoders",
                name="refit_encoders",
            ),
            Node(
                func=fit_scalers,
                inputs=["split_churn_data", "params:columns", "params:refit_scalers"],
                outputs="production_scalers",
                name="refit_scalers",
            ),
            Node(
                func=refit_model,
                inputs=[
                    "master_table",
                    "params:columns",
                    "optimized_model",
                    "params:refit_model",
                ],
                outputs="production_model",
                name="refit_model",
            ),
        ]
    )
