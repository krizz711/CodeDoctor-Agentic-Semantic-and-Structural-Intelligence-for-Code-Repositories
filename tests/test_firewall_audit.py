import pytest
from app.evaluation_harness import DevelopmentHarness, LeakageFirewallViolation


def test_firewall_blocks_test_access_by_default():
    """Plan T2: Leakage firewall blocks test access by default."""
    harness = DevelopmentHarness(firewall_enabled=True)
    with pytest.raises(LeakageFirewallViolation):
        harness.get_test_split()


def test_test_split_touched_exactly_twice_end_to_end():
    """Plan T2: Firewall authorizes only designated runs and enforces audit count."""
    harness = DevelopmentHarness(firewall_enabled=False)
    split1 = harness.get_test_split(authorize_reason="calibration")
    assert len(split1) > 0
    split2 = harness.get_test_split(authorize_reason="final_run")
    assert len(split2) > 0
    assert harness.test_split_access_count == 2

    with pytest.raises(PermissionError):
        harness.get_test_split(authorize_reason="just checking")


def test_train_split_loads_real_coir_apps_data():
    """Plan Remediation §3.6: Verify train split contains real APPS queries, not template strings."""
    harness = DevelopmentHarness(firewall_enabled=True)
    train = harness.get_train_split()
    assert len(train) == 5000, f"Expected 5,000 real train rows, got {len(train)}"
    first_item = train[0]
    assert "query_id" in first_item
    assert "query" in first_item
    assert "gold_id" in first_item
    assert not first_item["query"].startswith("python function "), (
        f"Query appears to be a synthetic template string: {first_item['query'][:50]}"
    )
    assert len(first_item["query"]) > 10, "Query string is unexpectedly short"


def test_persistent_audit_log_budget_exhaustion(tmp_path):
    """Section 0: Persistent audit log enforces maximum of 2 test split accesses lifetime."""
    log_file = tmp_path / "test_split_audit_log.json"
    harness = DevelopmentHarness(firewall_enabled=False, audit_log_path=log_file)

    # Touch 1: calibration
    s1 = harness.get_test_split(authorize_reason="calibration")
    assert len(s1) > 0
    assert harness.persistent_access_count(log_file) == 1

    # Touch 2: final_run
    s2 = harness.get_test_split(authorize_reason="final_run")
    assert len(s2) > 0
    assert harness.persistent_access_count(log_file) == 2

    # Touch 3: MUST be blocked as budget exhausted
    with pytest.raises(PermissionError, match="TEST SPLIT BUDGET EXHAUSTED"):
        harness.get_test_split(authorize_reason="final_run")
