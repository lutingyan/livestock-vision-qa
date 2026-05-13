import json
import pytest

@pytest.fixture
def eval_results():
    with open("data/eval_results.json") as f:
        return json.load(f)

def test_precision(eval_results):
    assert eval_results["precision"] >= 0.75, f"Precision {eval_results['precision']} too low"

def test_recall(eval_results):
    assert eval_results["recall"] >= 0.90, f"Recall {eval_results['recall']} too low"

def test_f1(eval_results):
    assert eval_results["f1"] >= 0.80, f"F1 {eval_results['f1']} too low"

def test_fps(eval_results):
    assert eval_results["avg_fps"] >= 15, f"FPS {eval_results['avg_fps']} too low"