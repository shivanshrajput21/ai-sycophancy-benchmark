from src.benchmark.result_manager import ResultManager


def test_jsonl_storage_and_completed_index(tmp_path):
    manager = ResultManager(tmp_path / "raw.jsonl")
    manager.append({"question_id": "q1", "stage": "initial", "status": "completed"})
    assert len(manager.read()) == 1
    assert ("q1", "initial", "") in manager.completed()
