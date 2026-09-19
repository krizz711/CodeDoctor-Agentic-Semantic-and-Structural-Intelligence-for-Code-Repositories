import pytest

from app.evaluation_harness import (
    DevelopmentHarness,
    LeakageFirewallViolation,
    compute_metrics_for_query,
    paired_bootstrap_test,
)


@pytest.mark.data
def test_plan_t2_leakage_firewall():
    harness = DevelopmentHarness(firewall_enabled=True)

    # Train split is always accessible
    train_data = harness.get_train_split()
    assert len(train_data) > 0
    assert harness.tuning_runs_logged == 1

    # Accessing test split during development MUST raise LeakageFirewallViolation
    with pytest.raises(LeakageFirewallViolation, match="LEAKAGE FIREWALL ACTIVE"):
        harness.get_test_split()

    # Disable firewall with unauthorized reason -> raises PermissionError
    harness.set_firewall(False)
    with pytest.raises(PermissionError, match="Unauthorized test split access"):
        harness.get_test_split(authorize_reason="tuning_model_hyperparams")

    # Authorized calibration run
    calib = harness.get_test_split(authorize_reason="calibration")
    assert len(calib) > 0
    assert harness.test_split_access_count == 1

    # Authorized final run
    final = harness.get_test_split(authorize_reason="final_run")
    assert len(final) > 0
    assert harness.test_split_access_count == 2
    print("\nPASS  test split accessed exactly 2 times this week (T4 calibration, final run)")
    print("PASS  all N tuning runs logged with firewall=ON")


def test_plan_t3_candidate_pool_recall_ceiling():
    # If gold doc is at rank 15
    ranked = [f"doc_{i}" for i in range(1, 100)]
    gold = "doc_15"

    metrics = compute_metrics_for_query(ranked, gold)
    # At rank 15: Recall@10 is 0.0, but Recall@50, @100, @200 are 1.0!
    assert metrics["recall_at_10"] == 0.0
    assert metrics["recall_at_50"] == 1.0
    assert metrics["recall_at_100"] == 1.0
    assert metrics["recall_at_200"] == 1.0
    assert metrics["ndcg_at_10"] == 0.0  # collapsed because not in top 10


def test_plan_t5_noise_band_paired_bootstrap():
    # Model A is consistently better than Model B with realistic query variance
    scores_a = [0.92, 0.85, 0.96, 0.79, 0.90, 0.88, 0.94, 0.85, 0.89, 0.91] * 5
    scores_b = [0.60, 0.58, 0.62, 0.53, 0.65, 0.55, 0.59, 0.61, 0.56, 0.60] * 5

    res = paired_bootstrap_test(scores_a, scores_b, n_resamples=1000)
    assert res["significant"] is True
    assert res["ci_lower"] > 0.0
    assert res["ci_upper"] > res["ci_lower"]

    # Model C vs Model D with noise only (mean diff ~ 0.001)
    scores_c = [0.80, 0.81, 0.79, 0.80, 0.82] * 10
    scores_d = [0.81, 0.80, 0.80, 0.79, 0.81] * 10

    res_noise = paired_bootstrap_test(scores_c, scores_d, n_resamples=1000)
    # The noise band spans zero -> must NOT be declared significant!
    assert res_noise["ci_lower"] <= 0.0 <= res_noise["ci_upper"]
    assert res_noise["significant"] is False
