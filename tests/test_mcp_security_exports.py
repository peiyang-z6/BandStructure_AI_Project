"""Sibling export and duplicate-source compatibility checks for security fixes."""

import csv
import json


def test_review_csv_neutralizes_metadata_and_preserves_original_json(tmp_path):
    from tests.test_figure_candidate_review import fixtures
    from mcp_server.figure_review_queue import load_figure_review_queue
    from mcp_server.visual_batch_review import save_visual_batch, finish_visual_review
    from scripts.export_oa_visual_review import export_review

    dataset, pre = fixtures(tmp_path)
    path = next((dataset.parent / "records").glob("*.json"))
    record = json.loads(path.read_bytes())
    record["source"]["doi"] = "=1+1"
    path.write_text(json.dumps(record), encoding="utf-8")
    queue = load_figure_review_queue(dataset, pre)
    review = tmp_path / "reviews"
    out = tmp_path / "export"
    save_visual_batch(
        queue,
        0,
        [{"i": 1, "target": "yes", "boxes": [[10, 20, 90, 80]], "note": "=SUM(1,2)"}],
        review,
    )
    finish_visual_review(queue, review)
    export_review(dataset, pre, review, out)
    with (out / "figure_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["doi"] == "'=1+1" and row["observation"] == "'=SUM(1,2)"
    original = json.loads((out / "figure_review.json").read_bytes())[0]
    assert original["doi"] == "=1+1" and original["observation"] == "=SUM(1,2)"
    assert original["human_audited"] is False


def test_duplicate_images_keep_distinct_candidates_without_retained_bytes(tmp_path):
    from tests.test_figure_candidate_preannotator import make_dataset
    from src.vision.figure_candidate_preannotator import _load_snapshot, preannotate_snapshot

    path = make_dataset(tmp_path)
    data = json.loads(path.read_bytes())
    first = data["records"][0]
    record = json.loads((path.parent / first["record_path"]).read_bytes())
    for i in range(1, 4):
        identifier = f"epmc-band-{i:024x}"
        relative = f"records/{identifier}.json"
        (path.parent / relative).write_text(
            json.dumps({**record, "candidate_id": identifier}), encoding="utf-8"
        )
        data["records"].append({**first, "candidate_id": identifier, "record_path": relative})
    data["record_count"] = 4
    path.write_text(json.dumps(data), encoding="utf-8")
    _, loaded = _load_snapshot(path)
    assert len(loaded) == 4 and all(not isinstance(item[1], bytes) for item in loaded)
    assert len({str(item[1]) for item in loaded}) == 1
    result = preannotate_snapshot(path, tmp_path / "pre", workers=2)
    assert result["record_count"] == 4


def test_csv_prefix_scan_does_not_copy_every_suffix():
    from src.utils.csv_safety import spreadsheet_text

    class Instrumented(str):
        slices = 0

        def __getitem__(self, key):
            if isinstance(key, slice):
                type(self).slices += 1
                return type(self)(super().__getitem__(key))
            return super().__getitem__(key)

    text = Instrumented(" " * 4096 + "=1+1")
    assert spreadsheet_text(text) == "'" + text
    assert Instrumented.slices <= 1


def test_pipe_thread_start_failure_kills_child_and_closes_stdin(monkeypatch):
    import io
    from mcp_server import resource_limits as module

    class Process:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None
            self.killed = False
            self.waited = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -1

        def wait(self, **kwargs):
            self.waited = True
            return self.returncode

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)

    class Thread:
        count = 0

        def __init__(self, **kwargs):
            pass

        def start(self):
            type(self).count += 1
            if type(self).count == 3:
                raise RuntimeError("synthetic thread exhaustion")

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(module.threading, "Thread", Thread)
    import pytest

    with pytest.raises(RuntimeError, match="thread exhaustion"):
        module.run_bounded(["unused"], input=b"", env={}, timeout=1)
    assert process.killed and process.waited and process.stdin.closed
