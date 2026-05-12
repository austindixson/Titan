from titan.evals import run_accuracy_eval


def test_accuracy_eval_matrix_all_passes():
    results = run_accuracy_eval()
    assert len(results) >= 4
    assert all(r.passed for r in results)
