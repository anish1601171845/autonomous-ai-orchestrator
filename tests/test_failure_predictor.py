import numpy as np
import pytest

from app.ml import failure_predictor as fp

REC = {"task_category": "development", "task_priority": "high", "agent_id": "devops-1", "task_complexity": 5,
       "dependency_count": 3, "estimated_duration": 12.0, "current_workload": 4, "deadline_slack": 0.9}
EASY = {**REC, "agent_id": "coder-1", "task_complexity": 1, "current_workload": 0, "dependency_count": 0, "deadline_slack": 2.5}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("fp")
    orig_model, orig_metrics = fp.MODEL_PATH, fp.METRICS_PATH
    fp.MODEL_PATH, fp.METRICS_PATH = d / "m.joblib", d / "metrics.json"
    fp.load_bundle.cache_clear()
    report = fp.train_failure_model(fp.prepare_failure_dataset(n_workflows=250, seed=7))
    yield report
    fp.MODEL_PATH, fp.METRICS_PATH = orig_model, orig_metrics
    fp.load_bundle.cache_clear()


def test_dataset_schema_and_synthetic_label():
    df = fp.prepare_failure_dataset(n_workflows=20, seed=1)
    assert set(fp.FEATURES + fp.TARGETS + ["workflow_id"]) <= set(df.columns)
    assert (df["source"] == "synthetic").all()
    assert df[fp.FEATURES].notna().all().all()


def test_leakage_prevention():
    assert not set(fp.FEATURES) & fp.POST_EXECUTION
    with pytest.raises(ValueError):
        fp.assert_no_leakage(fp.FEATURES + ["actual_duration"])


def test_training_reports_split_by_workflow_and_baseline(trained):
    for target in fp.TARGETS:
        t = trained["targets"][target]
        assert "dummy_prior" in t["candidates"] and t["selected"] != "dummy_prior"
        m = t["test_calibrated"]
        assert {"pr_auc", "precision", "recall", "f1", "brier", "false_positive_rate", "confusion_matrix"} <= m.keys()
        assert m["pr_auc"] > t["test_dummy"]["pr_auc"]  # better than the base-rate baseline
    assert "workflow" in trained["split"]


def test_persistence_and_probability_output(trained):
    assert fp.MODEL_PATH.exists()
    fp.load_bundle.cache_clear()
    p = fp.predict_failure_probability([REC, EASY])
    d = fp.predict_deadline_risk([REC, EASY])
    assert all(0 <= x <= 1 for x in p + d)
    assert p[0] > p[1] and d[0] > d[1]  # risk rises with complexity/workload, falls with slack


def test_missing_inputs_are_handled(trained):
    p = fp.predict_failure_probability([{"task_category": "testing"}, {}])
    assert len(p) == 2 and not np.isnan(p).any()


def test_risk_categories_and_explanation(trained):
    assert fp.risk_category(0.01) == "Low" and fp.risk_category(0.99) == "High"
    e = fp.explain_risk_prediction(REC, "deadline_missed")
    assert e["category"] in ("Low", "Medium", "High") and e["factors"] and "risk" in e["explanation"]
    assert e["method"] in ("shap", "linear_coefficients")
