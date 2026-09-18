"""Pipeline 'inference': score new data with the production artefacts.

Transform only — the encoders, scalers, and model all come from the refit
pipeline. Three of the five nodes are data_engineering functions reused as-is.
"""

from kedro.pipeline import Node, Pipeline

from churn.pipelines.data_engineering.nodes import (
    clean_data,
    transform_encoders,
    transform_scalers,
)

from .nodes import predict, to_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=to_dataframe,
                inputs="raw_inference_data",
                outputs="raw_inference_dataframe",
                name="to_dataframe",
            ),
            Node(
                func=clean_data,
                inputs=["raw_inference_dataframe", "params:columns"],
                outputs="cleaned_inference_data",
                name="clean_inference_data",
            ),
            Node(
                func=transform_encoders,
                inputs=["cleaned_inference_data", "production_encoders"],
                outputs="encoded_inference_data",
                name="encode_inference_data",
            ),
            Node(
                func=transform_scalers,
                inputs=["encoded_inference_data", "production_scalers"],
                outputs="scaled_inference_data",
                name="scale_inference_data",
            ),
            Node(
                func=predict,
                inputs=["production_model", "scaled_inference_data"],
                outputs="inference_predictions",
                name="predict",
            ),
        ]
    )
