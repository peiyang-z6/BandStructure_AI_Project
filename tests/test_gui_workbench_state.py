"""Real Tk raster/SHA race regressions. Every asset is SYNTHETIC ONLY."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
from test_p2_query_contracts import query_assets


def test_tk_display_and_saved_sha_use_same_bytes_during_source_replacement(query_assets, tmp_path, monkeypatch):
    from scripts import gui_workbench as gui
    a = query_assets
    config = tmp_path / 'SYNTHETIC_ONLY_query_config.json'
    config.write_text(json.dumps(dict(store_root=str(a['store'].root), index_name='tiny',
                                     encoder_path=str(a['model']), norm_path=str(a['norm']))))
    monkeypatch.setenv('BANDSTRUCTURE_P2_QUERY_CONFIG', str(config))
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    app = gui.BandStructureWorkbench()  # real Tcl/Tk, real canvas
    old_bytes = a['image'].read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    original = app._canvas.load_image
    def replace_after_display(pil):
        original(pil)
        # Deterministic on-disk race: OLD decoded white image, NEW blue file.
        Image.new('RGB', (200, 200), 'blue').save(a['image'])
    monkeypatch.setattr(app._canvas, 'load_image', replace_after_display)
    try:
        app._file_var.set(str(a['image']))
        app._load_file()
        assert app._canvas._pil_original.getpixel((0, 0)) == (255, 255, 255)
        with Image.open(a['image']) as disk: assert disk.getpixel((0, 0)) == (0, 0, 255)
        assert app._current_source_sha256 == old_sha, 'SHA describes a file never displayed'
        app._canvas.restore_annotations(a['annotations'])
        app._x1_var.set(0.); app._x2_var.set(1.)
        app._y1_var.set(-3.); app._y2_var.set(3.)
        app._retrieval_manual_var.set(True)  # programmatic SYNTHETIC ONLY, not review
        app._retrieval_button.invoke()
        assert app._last_retrieval_result is None
        assert 'image changed' in app._retrieval_result.get('1.0', 'end')
        # Restoring the EXACT displayed byte buffer removes only the race;
        # real frozen TF -> FAISS query should then succeed, uncalibrated.
        a['image'].write_bytes(old_bytes)
        app._retrieval_button.invoke()
        result = app._last_retrieval_result
        assert result['query_provenance']['image_sha256'] == old_sha
        assert result['query_provenance']['normalization_count'] == 1
        assert result['retrieval_kind'] == 'band-band'
        assert result['candidates'][0]['material_id'] == 'SYNTHETIC-Si'
        assert result['score_semantics'] == 'uncalibrated_cosine_similarity_not_probability'
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()


@pytest.mark.parametrize('format', ['PNG', 'PDF'])
def test_tk_decoder_cannot_reopen_replaced_path_after_snapshot(tmp_path, monkeypatch, format):
    import io
    from scripts import gui_workbench as gui
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    def payload(color):
        if format == 'PNG':
            out = io.BytesIO(); Image.new('RGB', (200, 200), color).save(out, format='PNG')
            return out.getvalue()
        import fitz
        with fitz.open() as doc:
            page = doc.new_page(width=72, height=72)
            if color == 'blue': page.draw_rect(page.rect, fill=(0, 0, 1))
            return doc.tobytes()
    old, new = payload('white'), payload('blue')
    source = tmp_path / ('SYNTHETIC_ONLY_source.' + format.lower())
    source.write_bytes(old)
    reads = []
    original = Path.read_bytes
    def replace_on_read(path):
        result = original(path)
        if path == source:
            reads.append(result)
            source.write_bytes(new)
        return result
    monkeypatch.setattr(Path, 'read_bytes', replace_on_read)
    app = gui.BandStructureWorkbench()
    try:
        app._file_var.set(str(source)); app._load_file()
        assert len(reads) == 1
        assert app._current_source_sha256 == hashlib.sha256(old).hexdigest()
        assert app._canvas._pil_original.getpixel((10, 10)) == (255, 255, 255)
        assert tuple(app.root.tk.call(str(app._canvas._tk_image), 'get', 10, 10)) == (255, 255, 255)
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()


def test_existing_tk_recognize_guard_and_supervised_save_remain_separate(tmp_path, monkeypatch):
    from scripts import gui_workbench as gui
    from test_p2_query_contracts import _manual
    monkeypatch.setattr(gui, 'WORKBENCH_STATE_DIR', tmp_path / 'state')
    exports = tmp_path / 'SYNTHETIC_ONLY_supervised_exports'
    exports.mkdir()
    monkeypatch.setattr(gui, 'TRAINING_DATA_DIR', exports)
    app = gui.BandStructureWorkbench()
    try:
        app._submit_recognition()  # original missing-input guard, no real model access
        assert 'Missing: draw the yellow panel box.' in app._recog_result.get('1.0', 'end')
        ann, cal = _manual()
        app._canvas.restore_annotations(ann)
        app._mat_id_var.set('SYNTHETIC_ONLY_NOT_HUMAN_REVIEW')
        app._save_training()  # real original export, not ANN/authentication
        saved = [json.loads(p.read_text()) for p in exports.glob('*.json')]
        assert len(saved) == 1
        assert saved[0]['material_id'] == 'SYNTHETIC_ONLY_NOT_HUMAN_REVIEW'
        assert saved[0]['annotations']['vbm'] == ann['vbm']
        assert json.loads((exports / 'training_manifest.jsonl').read_text()) == saved[0]
        assert 'review' not in saved[0] and 'authenticated' not in saved[0]
        assert app._last_retrieval_result is None
    finally:
        for timer in app.root.tk.call('after', 'info'): app.root.after_cancel(timer)
        app.root.destroy()
