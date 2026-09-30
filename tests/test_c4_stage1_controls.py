"""Synthetic tests for the C4 validity boundary, not benchmark evidence."""
import json
import numpy as np
from scripts.c4_stage1 import compare


def write_arm(folder, name, vectors, *, seconds=1.0, tokens=None):
    np.save(folder / f"{name}-vectors.npy", vectors, allow_pickle=False)
    data = {"qids": [f"q{i}" for i in range(100)], "seconds": [seconds] * 100,
            "token_ids": tokens or [[1, 151645]] * 100, "edge_ids": [[1, 151645]] * 7}
    (folder / f"{name}-raw.json").write_text(json.dumps(data))


def vectors():
    arr = np.zeros((100, 2048), dtype=np.float32)
    arr[:, 0] = 1
    return arr


def test_f32_control_failure_voids_before_treatment(tmp_path):
    baseline = vectors()
    rotated = np.zeros_like(baseline)
    rotated[:, 1] = 1
    write_arm(tmp_path, "torch", baseline)
    write_arm(tmp_path, "f32", rotated)
    result = compare(tmp_path, {"peak_process_tree_rss_bytes": 10})
    assert result["status"] == "void"
    assert "F32" in result["reason"]


def test_q8_failure_is_negative_after_valid_control(tmp_path):
    baseline = vectors()
    write_arm(tmp_path, "torch", baseline)
    write_arm(tmp_path, "f32", baseline)
    write_arm(tmp_path, "q8", baseline, seconds=2.0)
    result = compare(tmp_path, {"peak_process_tree_rss_bytes": 10}, {"peak_process_tree_rss_bytes": 11})
    assert result["status"] == "negative"
    assert result["adoption"] is False


def test_token_mismatch_voids_control(tmp_path):
    baseline = vectors()
    write_arm(tmp_path, "torch", baseline)
    write_arm(tmp_path, "f32", baseline, tokens=[[2, 151645]] * 100)
    assert compare(tmp_path, {"peak_process_tree_rss_bytes": 10})["status"] == "void"
