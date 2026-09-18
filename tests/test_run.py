"""Project-level tests: pipeline registration and DAG wiring.

These are the integration layer of the test pyramid — they check that the four
pipelines are discoverable and that their inputs and outputs connect, without
actually running any data through them.
"""

from kedro.framework.project import find_pipelines

from churn.pipeline_registry import register_pipelines

EXPECTED_PIPELINES = {"data_engineering", "modelling", "refit", "inference"}

# 6 data_engineering + 4 modelling + 3 refit + 5 inference
EXPECTED_NODE_COUNT = 18


class TestPipelineRegistry:
    def test_all_pipelines_are_discovered(self):
        pipelines = find_pipelines(raise_errors=True)

        assert EXPECTED_PIPELINES <= set(pipelines)

    def test_default_pipeline_contains_every_node(self):
        """`kedro run` with no --pipeline must cover all four pipelines."""
        default = register_pipelines()["__default__"]

        assert len(default.nodes) == EXPECTED_NODE_COUNT

    def test_production_artefacts_connect_refit_to_inference(self):
        pipelines = find_pipelines(raise_errors=True)
        produced = pipelines["refit"].outputs()
        consumed = pipelines["inference"].inputs()

        assert {"production_encoders", "production_scalers", "production_model"} <= produced
        assert {"production_encoders", "production_scalers", "production_model"} <= consumed

    def test_master_table_connects_data_engineering_to_modelling(self):
        pipelines = find_pipelines(raise_errors=True)

        assert "master_table" in pipelines["data_engineering"].outputs()
        assert "master_table" in pipelines["modelling"].inputs()

    def test_inference_pipeline_fits_nothing(self):
        """No artefact may be *created* by the inference pipeline except predictions."""
        pipelines = find_pipelines(raise_errors=True)
        node_names = {node.name for node in pipelines["inference"].nodes}

        assert not any(name.startswith(("fit_", "refit_")) for name in node_names)
