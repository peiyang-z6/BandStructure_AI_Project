"""Synthetic analytic PDF oracles, not real-paper or human calibration."""
import hashlib
import pymupdf
import pytest
from src.vision import multi_format_parser as parser


def pdf_document():
    with pymupdf.open() as doc:
        for index in range(2):
            page = doc.new_page(width=400, height=500)
            page.insert_text((25, 25), f'Synthetic page {index + 1}')
        return doc.tobytes()


def digitize(payload, **kwargs):
    fn = getattr(parser, 'digitize_pdf_band_panels', None)
    assert callable(fn), 'Missing bounded PDF band digitization entry point'
    return fn(payload, **kwargs)


def test_blank_page_returns_honest_geometry_envelope():
    payload = pdf_document()
    out = digitize(payload, page_number=2)
    assert out['status'] == 'geometry_only'
    assert out['page_number'] == 2 and out['total_pages'] == 2
    assert out['source_sha256'] == hashlib.sha256(payload).hexdigest()
    assert out['panels'] == [] and out['band_data'] is None
    assert out['confidence'] is None and out['ood_flag'] is None
    assert out['human_audited'] is False and out['calibration_verified'] is False


@pytest.mark.parametrize('page_number', [True, False, 0, -1, 1.0, '1', 3])
def test_rejects_invalid_page_selector(page_number):
    with pytest.raises(ValueError, match='page_number'):
        digitize(pdf_document(), page_number=page_number)


@pytest.mark.parametrize('case', ['empty', 'text', 'mutable', 'oversize'])
def test_rejects_invalid_pdf_payload(case):
    valid = pdf_document()
    payload = {'empty': b'', 'text': 'not bytes', 'mutable': bytearray(valid),
               'oversize': valid + b' ' * (10 * 1024 * 1024)}[case]
    with pytest.raises(ValueError, match='payload'):
        digitize(payload)


def draw_panel(page, box, unit='eV', style='filled', label='(a)'):
    x0, y0, x1, y1 = box
    s = (x1 - x0) / 200
    if style == 'stroke':
        page.draw_rect(box, color=(0, 0, 0), width=.4 * s)
    else:
        t = .4 * s
        for r in [(x0-t/2, y0-t/2, x0+t/2, y1+t/2),
                  (x1-t/2, y0-t/2, x1+t/2, y1+t/2),
                  (x0-t/2, y0-t/2, x1+t/2, y0+t/2),
                  (x0-t/2, y1-t/2, x1+t/2, y1+t/2)]:
            page.draw_rect(r, fill=(.1, .1, .1), color=None)
    if unit:
        page.insert_text((x0-36*s, (y0+y1)/2), f'({unit})', fontsize=8*s)
    page.insert_text((x0, y0-8*s), label, fontsize=9*s)


@pytest.mark.parametrize('scale,offset,page_number,style',
                         [(1, 0, 1, 'filled'), (.5, 20, 2, 'filled'), (1.5, 11, 2, 'stroke')])
def test_selects_closed_ev_frames_without_confusing_mev(scale, offset, page_number, style):
    def box(y):
        return [offset+v*scale for v in (80, y, 280, y+140)]
    with pymupdf.open() as doc:
        for _ in range(page_number):
            page = doc.new_page(width=400*scale+offset, height=700*scale+offset)
        draw_panel(page, box(80), style=style)
        draw_panel(page, box(280), unit='meV', style=style, label='(b)')
        draw_panel(page, box(480), unit='', style=style, label='(c)')
        out = digitize(doc.tobytes(), page_number=page_number)
    assert len(out['panels']) == 1
    p = out['panels'][0]
    assert p['bbox'] == pytest.approx(box(80), abs=.01*scale)
    assert p['panel_id'] and p['label'] == '(a)'
    assert p['unit'] == 'eV' and p['unit_evidence'][0]['engine'] == 'pymupdf_text'
    assert p['calibration_verified'] is False
    assert sorted(c['reason'] for c in out['calibration_candidates']) == ['missing_ev_unit', 'non_electronic_unit_meV']


def thin_polyline(page, points, thickness=.6, color=(0, 0, 0), shape=None):
    shape = shape or page.new_shape()
    polygon = [(x, y-thickness/2) for x, y in points] + [(x, y+thickness/2) for x, y in points[::-1]]
    shape.draw_polyline(polygon + [polygon[0]])
    return shape


def test_extracts_separate_black_filled_subpaths_not_red_or_grid():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220])
        shape = thin_polyline(page, [(90, 130), (130, 142), (170, 154)])
        thin_polyline(page, [(190, 160), (225, 170), (270, 180)], shape=shape)
        shape.finish(fill=(0, 0, 0), color=None)
        shape.commit()
        black_id = len(page.get_drawings())-1
        shape = thin_polyline(page, [(90, 105), (270, 120)], thickness=5)
        shape.finish(fill=(1, 0, 0), color=None)
        shape.commit()
        page.draw_line((180, 80), (180, 220), color=(.4, .4, .4), width=.4)
        page.draw_line((80, 150), (280, 150), color=(.4, .4, .4), width=.4)
        out = digitize(doc.tobytes())
    traces = out['panels'][0]['traces']
    assert len(traces) == 2
    assert {t['source_path_id'] for t in traces} == {black_id}
    assert len({t['source_subpath_id'] for t in traces}) == 2
    for trace in traces:
        assert trace['energy_eV'] is None and trace['band_data'] is None
        assert len(trace['points_pdf']) >= 3
        assert trace['k_unit'] == 'relative'
        assert trace['uncertainty_reasons'] and trace['physical_energy_uncertainty_eV'] is None
        x, y = zip(*trace['points_pdf'])
        assert min(x) >= 90 and max(x) <= 270
        if max(x) < 180:
            assert list(y) == pytest.approx([130+.3*(xx-90) for xx in x], abs=.01)
        assert trace['k_relative'] == pytest.approx([(xx-80)/200 for xx in x])


def test_splits_traces_at_high_symmetry_guides():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220])
        shape = thin_polyline(page, [(90, 120), (270, 180)])
        shape.finish(fill=(0, 0, 0), color=None)
        shape.commit()
        # Filled dash outlines exercise a publisher-style separator.
        guide = page.new_shape()
        for y in range(80, 220, 4):
            guide.draw_rect([179.8, y, 180.2, y+2])
        guide.finish(fill=(0, 0, 0), color=None)
        guide.commit()
        out = digitize(doc.tobytes())
    panel = out['panels'][0]
    assert len(panel['traces']) == 2
    assert panel['k_breaks_relative'] == pytest.approx([.5])
    for trace in panel['traces']:
        assert max(trace['k_relative']) < .5 or min(trace['k_relative']) > .5
        assert 'high_symmetry_boundaries_not_interpolated' in trace['uncertainty_reasons']


def calibrated_pdf(amplitude=3, reverse=False, omit=None):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220], unit='')
        # Keep the unit separate from the numeric labels, including the middle tick.
        page.insert_text((44, 118), '(eV)', fontsize=8)
        sign = -1 if reverse else 1
        for y, factor in [(80, 1), (150, 0), (220, -1)]:
            page.draw_line((77, y), (83, y), color=(0, 0, 0), width=.4)
            if omit != 'ticks':
                page.insert_text((58, y+3), f'{2+factor*amplitude*sign:g}', fontsize=8)
        page.draw_line((80, 150), (280, 150), color=(0, 0, 0), width=.4, dashes='[2 2] 0')
        if omit != 'fermi':
            page.insert_text((286, 153), 'EF', fontsize=8)
        if omit != 'bands':
            shape = thin_polyline(page, [(90, 120), (180, 150), (270, 180)])
            shape.finish(fill=(0, 0, 0), color=None)
            shape.commit()
        return doc.tobytes()


@pytest.mark.parametrize('amplitude,reverse', [(.25, False), (3.5, False), (3.5, True)])
def test_text_tick_geometry_and_explicit_ef_enable_unverified_mapping(amplitude, reverse):
    out = digitize(calibrated_pdf(amplitude, reverse))
    panel = out['panels'][0]
    assert panel['status'] == 'digitized_unverified'
    assert panel['calibration_verified'] is False
    assert len(panel['tick_evidence']) == 3 and panel['fermi_evidence']['label_text'] == 'EF'
    mapping = panel['energy_mapping']
    sign = -1 if reverse else 1
    assert mapping['axis_value_at_fermi_eV'] == pytest.approx(2)
    assert mapping['slope_eV_per_pdf_point'] == pytest.approx(-amplitude/70*sign)
    for trace in panel['traces']:
        expected = [(150-y)*amplitude/70*sign for x, y in trace['points_pdf']]
        assert trace['energy_eV'] == pytest.approx(expected, abs=1e-7)
        band = trace['band_data']
        assert band['energies_eV'] == [trace['energy_eV']]
        assert band['fermi_eV'] == 0 and band['k_unit'] == 'relative'
        assert band['k_distance'] == trace['k_relative']
        assert band['segment_ids'] == [0]*len(trace['points_pdf'])
        assert 'band_roles' not in band


def test_local_ocr_recovers_raster_tick_text_with_geometry_evidence():
    with pymupdf.open(stream=calibrated_pdf(), filetype='pdf') as source:
        clip = pymupdf.Rect(54, 69, 77, 231)
        image = source[0].get_pixmap(matrix=pymupdf.Matrix(6, 6), clip=clip).tobytes('png')
    with pymupdf.open(stream=calibrated_pdf(omit='ticks'), filetype='pdf') as target:
        target[0].insert_image(clip, stream=image)
        out = digitize(target.tobytes())
    panel = out['panels'][0]
    assert panel['status'] == 'digitized_unverified', {
        'ticks': panel['tick_evidence'], 'warnings': panel.get('calibration_warnings'),
        'ocr': {k: v for k, v in panel.get('tick_strip_ocr', {}).items() if k != 'png_base64'}}
    assert len(panel['tick_evidence']) == 3
    assert {t['engine'] for t in panel['tick_evidence']} == {'rapidocr_onnxruntime'}
    evidence = panel['tick_strip_ocr']
    assert evidence['regions'] and evidence['image_size'][0]*evidence['image_size'][1] <= 4_000_000
    assert evidence['clip_bbox_pdf'][2] <= panel['bbox'][0]
    assert evidence['engine_version'] and evidence['model_sha256']
    assert panel['energy_mapping']['slope_eV_per_pdf_point'] == pytest.approx(-3/70)


def test_hidden_vector_geometry_is_not_reconstructed_through_overlay():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220])
        shape = thin_polyline(page, [(90, 120), (270, 180)])
        shape.finish(fill=(0, 0, 0), color=None)
        shape.commit()
        page.draw_rect([150, 90, 210, 210], fill=(1, 1, 1), color=None)
        shape = thin_polyline(page, [(150, 140), (210, 160)], thickness=4)
        shape.finish(fill=(1, 0, 0), color=None)
        shape.commit()
        out = digitize(doc.tobytes())
    panel = out['panels'][0]
    assert len(panel['traces']) == 2
    assert panel['visibility_evidence']['method'] == 'rendered_neutral_ink_at_vector_midline'
    for trace in panel['traces']:
        xx = [p[0] for p in trace['points_pdf']]
        assert max(xx) < 150 or min(xx) > 210
        assert 'occlusion_gaps_not_interpolated' in trace['uncertainty_reasons']


@pytest.mark.parametrize('scale', [.5, 1.5])
def test_adaptive_cubic_outline_midline_matches_analytic_parabola(scale):
    point = lambda x, y: (x*scale, y*scale)
    with pymupdf.open() as doc:
        page = doc.new_page(width=400*scale, height=400*scale)
        draw_panel(page, [v*scale for v in [80, 80, 280, 220]])
        shape = page.new_shape()
        shape.draw_bezier(point(90, 179.7), point(150, 99.7), point(210, 99.7), point(270, 179.7))
        shape.draw_line(point(270, 179.7), point(270, 180.3))
        shape.draw_bezier(point(270, 180.3), point(210, 100.3), point(150, 100.3), point(90, 180.3))
        shape.draw_line(point(90, 180.3), point(90, 179.7))
        shape.finish(fill=(0, 0, 0), color=None)
        shape.commit()
        out = digitize(doc.tobytes())
    traces = out['panels'][0]['traces']
    assert traces
    points = [point for trace in traces for point in trace['points_pdf']]
    assert len(points) > 200
    for x, y in points:
        t = (x/scale-90)/180
        assert abs(y/scale-(180-240*t+240*t*t))/140 < 2e-4


def test_generated_svg_and_csv_roundtrip_only_the_returned_fragments():
    import csv
    import io
    import xml.etree.ElementTree as ET
    out = digitize(calibrated_pdf())
    svg = ET.fromstring(out['clean_svg'])
    assert svg.tag == '{http://www.w3.org/2000/svg}svg'
    polylines = svg.findall('.//{http://www.w3.org/2000/svg}polyline')
    traces = [t for p in out['panels'] for t in p['traces']]
    assert len(polylines) == len(traces) > 0
    rows = list(csv.DictReader(io.StringIO(out['csv_text'])))
    assert len(rows) == sum(len(t['points_pdf']) for t in traces)
    for trace in traces:
        selected = [r for r in rows if r['trace_id'] == trace['trace_id']]
        assert [float(r['k_relative']) for r in selected] == trace['k_relative']
        assert [float(r['energy_eV']) for r in selected] == trace['energy_eV']
        assert all(int(r['source_path_id']) == trace['source_path_id'] for r in selected)
    assert 'script' not in out['clean_svg'] and 'http://' not in out['csv_text']
    assert len(out['clean_svg'].encode()) <= 4*1024*1024
    assert out['status'] == 'digitized_unverified' and out['band_data'] is None
    assert out['limitations'] and out['needs_ocr_review'] is False


def test_rejects_oversized_page_before_geometry_parsing(monkeypatch):
    with pymupdf.open() as doc:
        doc.new_page(width=10000, height=10000)
        payload = doc.tobytes()
    def forbidden(*args, **kwargs):
        pytest.fail('oversized page reached geometry allocation')
    monkeypatch.setattr(pymupdf.Page, 'get_drawings', forbidden)
    with pytest.raises(ValueError, match='page.*budget'):
        digitize(payload)


@pytest.mark.parametrize('kind', ['drawings', 'items'])
def test_rejects_decoded_vector_complexity_before_panel_search(monkeypatch, kind):
    # Mock only the hostile decoder boundary, not geometry or numeric results.
    drawing = {'items': [('re', pymupdf.Rect(80, 80, 80.2, 220), 1)],
               'rect': pymupdf.Rect(80, 80, 80.2, 220), 'fill': (0, 0, 0)}
    drawings = [drawing]*50001 if kind == 'drawings' else [{**drawing, 'items': drawing['items']*500001}]
    monkeypatch.setattr(pymupdf.Page, 'get_drawings', lambda *a, **k: drawings)
    def forbidden(*a, **k):
        pytest.fail('decoded geometry exceeded admission cap but reached panel search')
    monkeypatch.setattr(parser, '_pdf_band_frames', forbidden)
    with pytest.raises(ValueError, match='geometry.*budget'):
        digitize(pdf_document())


def test_missing_numeric_ticks_retains_geometry_candidates_without_energy():
    out = digitize(calibrated_pdf(omit='ticks'))
    panel = out['panels'][0]
    assert panel['status'] == 'needs_ocr_review' and panel['energy_mapping'] is None
    assert len(panel['tick_geometry_candidates']) == 3
    assert len(panel['reference_line_candidates']) == 1
    assert panel['tick_strip_ocr']['regions']
    assert all(t['energy_eV'] is None and t['band_data'] is None for t in panel['traces'])
    assert out['needs_ocr_review'] is True


@pytest.mark.parametrize('resource', ['words', 'frame_comparisons', 'panels', 'flattened_points',
                                    'section_edge_tests', 'cross_section_samples', 'visibility_checks',
                                    'svg_bytes', 'csv_bytes', 'json_bytes'])
def test_fixed_work_budget_is_accounted_and_enforced(monkeypatch, resource):
    out = digitize(calibrated_pdf())
    budget = out['resource_budget']
    assert 0 < budget['used'][resource] <= budget['limits'][resource]
    # Only the admission cap is overridden to exercise exhaustion on a tiny real PDF.
    monkeypatch.setitem(parser._PDF_DIGITIZATION_LIMITS, resource, 0)
    with pytest.raises(ValueError, match='budget.*'+resource):
        digitize(calibrated_pdf())


def test_unfilled_cubic_stroke_uses_its_centerline_not_a_closed_polygon():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220], style='stroke')
        page.draw_bezier((90, 180), (150, 100), (210, 100), (270, 180),
                         color=(0, 0, 0), width=.6, closePath=False)
        stroke_id = len(page.get_drawings())-1
        out = digitize(doc.tobytes())
    traces = out['panels'][0]['traces']
    assert traces
    assert {t['source_path_id'] for t in traces} == {stroke_id}
    for trace in traces:
        assert trace['geometry_kind'] == 'stroke_centerline'
        for x, y in trace['points_pdf']:
            t = (x-90)/180
            assert abs(y-(180-240*t+240*t*t))/140 < 2e-4


def test_rotated_page_is_refused_instead_of_mixing_coordinate_systems():
    with pymupdf.open(stream=calibrated_pdf(), filetype='pdf') as doc:
        doc[0].set_rotation(90)
        with pytest.raises(ValueError, match='rotated'):
            digitize(doc.tobytes())


def test_calibrated_axes_without_traces_are_not_digitization_success():
    out = digitize(calibrated_pdf(omit='bands'))
    assert out['panels'][0]['energy_mapping'] is not None
    assert out['panels'][0]['traces'] == []
    assert out['panels'][0]['status'] == 'geometry_only'
    assert out['status'] == 'geometry_only' and out['band_data'] is None


# Additional first-observation regressions below are not claimed as TDD RED cycles.
def test_existing_missing_ef_guard_never_assumes_the_zero_tick_is_fermi():
    out = digitize(calibrated_pdf(omit='fermi'))
    panel = out['panels'][0]
    assert panel['energy_mapping'] is None and panel['status'] == 'needs_ocr_review'
    assert len(panel['tick_evidence']) == 3
    assert all(t['energy_eV'] is None and t['band_data'] is None for t in panel['traces'])


@pytest.mark.parametrize('kind', ['closed_loop', 'multivalued_stroke'])
def test_existing_ambiguous_stroke_guard_does_not_invent_a_band(kind):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        draw_panel(page, [80, 80, 280, 220])
        if kind == 'closed_loop':
            page.draw_circle((180, 150), 30, color=(0, 0, 0), width=.6)
        else:
            page.draw_polyline([(100, 110), (250, 190), (100, 190), (250, 110)],
                               color=(0, 0, 0), width=.6, closePath=False)
        out = digitize(doc.tobytes())
    assert out['panels'][0]['traces'] == []


@pytest.mark.parametrize('scale', [.5, 1.5])
def test_existing_calibration_is_invariant_to_form_scale_translation_and_page(scale):
    with pymupdf.open(stream=calibrated_pdf(), filetype='pdf') as source:
        with pymupdf.open() as doc:
            doc.new_page(width=900, height=1000)
            page = doc.new_page(width=900, height=1000)
            page.show_pdf_page(pymupdf.Rect(120, 230, 120+400*scale, 230+400*scale), source, 0)
            out = digitize(doc.tobytes(), page_number=2)
    panel = out['panels'][0]
    assert panel['status'] == 'digitized_unverified'
    assert panel['energy_mapping']['slope_eV_per_pdf_point'] == pytest.approx(-3/(70*scale), rel=1e-5)
    assert panel['energy_mapping']['fermi_y_pdf'] == pytest.approx(230+150*scale, abs=.001)
    assert panel['traces']
    for trace in panel['traces']:
        for (x, y), energy in zip(trace['points_pdf'], trace['energy_eV']):
            assert energy == pytest.approx((150-(y-230)/scale)*3/70, abs=1e-5)


def visibility_pdf(case):
    # Independent-review geometry, recreated for a portable regression fixture.
    with pymupdf.open() as doc:
        page=doc.new_page(width=400,height=320)
        page.insert_text((20,24),'SYNTHETIC DEVELOPMENT CONTROL: '+case,fontsize=8)
        page.draw_rect([80,80,280,220],color=(0,0,0),width=.4)
        page.insert_text((80,67),'(a)',fontsize=8)
        page.insert_text((42,118),'(meV)' if case in ('mev_control','unit_closer') else '(eV)',fontsize=8)
        for y,text in [(80,'5'),(150,'2'),(220,'-1')]:
            page.draw_line((77,y),(83,y),color=(0,0,0),width=.4)
            page.insert_text((64,y+3),text,fontsize=8,render_mode=3 if case=='invisible_ticks' else 0,
                             color=(1,1,1) if case=='white_ticks' else (0,0,0),
                             fill_opacity=0 if case=='transparent_ticks' else 1)
        page.draw_line((80,150),(280,150),color=(0,0,0),width=.4,dashes='[2 2] 0')
        page.insert_text((286,153),'EF',fontsize=8)
        page.insert_text((155,236),'relative k',fontsize=8)
        page.draw_polyline([(92,115),(180,146),(268,185)],color=(0,0,0),width=.8,closePath=False)
        if case=='covered_ticks':page.draw_rect([62,66,76,233],fill=(1,1,1),color=None)
        if case=='covered_ef':page.draw_rect([284,141,306,157],fill=(1,1,1),color=None)
        if case=='unit_closer':page.insert_text((60,132),'(eV)',fontsize=8,render_mode=3)
        return doc.tobytes()


def review_visibility_fixture(case):
    import os
    from pathlib import Path
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    return (Path(frozen)/'vis'/(case+'.pdf')).read_bytes() if frozen else visibility_pdf(case)


@pytest.mark.parametrize('case',['covered_ticks','covered_ef','invisible_ticks','unit_closer'])
def test_invisible_calibration_text_cannot_authorize_numeric_bands(case):
    out=digitize(review_visibility_fixture(case))
    traces=[t for p in out['panels'] for t in p['traces']]
    assert not any(t['band_data'] is not None for t in traces), 'PDF-VIS-001: nonvisible text authorized numerical bands'
    assert all(t['energy_eV'] is None for t in traces)
    assert out['confidence'] is None and out['calibration_verified'] is False


def test_visible_review_control_still_authorizes_unverified_numeric_bands():
    out=digitize(review_visibility_fixture('control'))
    assert out['status']=='digitized_unverified'
    assert any(t['band_data'] is not None for p in out['panels'] for t in p['traces'])


def visible_unit_conflict_pdf():
    with pymupdf.open(stream=visibility_pdf('mev_control'),filetype='pdf') as doc:
        doc[0].insert_text((60,132),'(eV)',fontsize=8)
        return doc.tobytes()


def test_conflicting_visible_units_are_not_resolved_by_nearest_text():
    import os
    from pathlib import Path
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    raw=(Path(frozen)/'conflict/fixture.pdf').read_bytes() if frozen else visible_unit_conflict_pdf()
    out=digitize(raw)
    assert out['panels']==[], 'PDF-VIS-001: competing visible units require ambiguity, not nearest unit'
    assert len(out['calibration_candidates'])==1
    assert out['calibration_candidates'][0]['reason']=='conflicting_visible_energy_units'


def partial_grid_pdf():
    with pymupdf.open(stream=calibrated_pdf(omit='bands'),filetype='pdf') as doc:
        doc[0].draw_line((95,110),(265,110),color=(0,0,0),width=.4,dashes='[5 3] 0')
        return doc.tobytes()


def test_partial_dashed_grid_is_geometry_not_quantitative_band():
    import os
    from pathlib import Path
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    raw=(Path(frozen)/'grid/partial_dashed_grid_only.pdf').read_bytes() if frozen else partial_grid_pdf()
    out=digitize(raw)
    traces=[t for p in out['panels'] for t in p['traces']]
    assert traces, 'Retain visible diagnostic geometry rather than globally disabling extraction'
    assert not any(t['band_data'] is not None for t in traces), 'PDF-GRID-002: grid primitive became a numerical band'
    assert all(t['energy_eV'] is None and t['quantitative_exclusion_reason']=='flat_or_dashed_source_ambiguous_with_guide' for t in traces)
    assert out['status']=='geometry_only'


def calibration_with_prose_pdf():
    with pymupdf.open(stream=calibrated_pdf(),filetype='pdf') as doc:
        for _ in range(1600):
            doc[0].insert_text((20,350),'prose',fontsize=8)
        return doc.tobytes(garbage=4,deflate=True)


def test_visibility_budget_is_spent_on_calibration_not_unrelated_prose():
    import os
    from pathlib import Path
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    raw=(Path(frozen)/'prose/fixture.pdf').read_bytes() if frozen else calibration_with_prose_pdf()
    out=digitize(raw)
    assert out['status']=='digitized_unverified'
    assert any(t['band_data'] is not None for p in out['panels'] for t in p['traces'])


def additional_visibility_pdf(case):
    if case in ('white_ticks','transparent_ticks'):
        return visibility_pdf(case)
    import io
    from PIL import Image
    image=io.BytesIO();Image.new('RGB',(88,64),'black' if case=='black_image_ef' else 'white').save(image,format='PNG')
    with pymupdf.open(stream=visibility_pdf('control'),filetype='pdf') as doc:
        doc[0].insert_image(pymupdf.Rect(284,141,306,157),stream=image.getvalue())
        return doc.tobytes()


# Post-fix coverage of the same visibility gate, not additional claimed RED cycles.
@pytest.mark.parametrize('case',['white_ticks','transparent_ticks','white_image_ef','black_image_ef'])
def test_common_invisible_paint_and_image_cover_remain_nonquantitative(case):
    import os
    from pathlib import Path
    frozen=os.environ.get('LIMITED_FIX_FIXTURES')
    raw=(Path(frozen)/'extras'/(case+'.pdf')).read_bytes() if frozen else additional_visibility_pdf(case)
    out=digitize(raw)
    assert len(out['panels'])==1 and out['panels'][0]['traces']
    assert out['panels'][0]['energy_mapping'] is None
    assert all(t['band_data'] is None and t['energy_eV'] is None for t in out['panels'][0]['traces'])


# PDF-VIS-001-R1: independent tick RED precedes the native-contribution repair.
def same_color_calibration_pdf(kind, present=True, background=0., foreground=None, noise=False):
    foreground = background if foreground is None else foreground
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=320)
        boxes = {'ticks': (62, 66, 76, 233), 'ef': (284, 141, 306, 157),
                 'unit': (40, 106, 64, 121)}
        page.draw_rect(boxes[kind], fill=(background,)*3, color=None)
        if noise:
            # Unrelated light marks inside the texttrace descender boxes, below
            # these non-descending glyphs. They must not count as glyph contrast.
            marks = {'ticks': [(64.5, y+3.5) for y in (80, 150, 220)],
                     'ef': [(286.5, 153.5)], 'unit': [(47, 118.5)]}
            for x, y in marks[kind]:
                page.draw_rect((x, y, x+.5, y+.5), fill=(.9,)*3, color=None)
        page.draw_rect((80, 80, 280, 220), color=(0,)*3, width=.4)
        page.insert_text((80, 67), '(a)', fontsize=8)
        if kind != 'unit' or present:
            page.insert_text((42, 118), '(eV)', fontsize=8,
                             color=(foreground,)*3 if kind == 'unit' else (0,)*3)
        for y, text in [(80, '5'), (150, '2'), (220, '-1')]:
            page.draw_line((77, y), (83, y), color=(0,)*3, width=.4)
            if kind != 'ticks' or present:
                page.insert_text((64, y+3), text, fontsize=8,
                                 color=(foreground,)*3 if kind == 'ticks' else (0,)*3)
        page.draw_line((80, 150), (280, 150), color=(0,)*3, width=.4, dashes='[2 2] 0')
        if kind != 'ef' or present:
            page.insert_text((286, 153), 'EF', fontsize=8,
                             color=(foreground,)*3 if kind == 'ef' else (0,)*3)
        page.draw_polyline([(92, 115), (180, 146), (268, 185)], color=(0,)*3,
                           width=.8, closePath=False)
        return doc.tobytes()


def rendered_calibration_rgb(payload):
    with pymupdf.open(stream=payload, filetype='pdf') as doc:
        return doc[0].get_pixmap(matrix=pymupdf.Matrix(4, 4),
                                colorspace=pymupdf.csRGB, alpha=False).samples


def assert_no_calibration(payload):
    out = digitize(payload)
    assert out['status'] == 'geometry_only'
    assert all(p['energy_mapping'] is None for p in out['panels'])
    assert all(t['band_data'] is None and t['energy_eV'] is None
               for p in out['panels'] for t in p['traces'])
    assert out['confidence'] is None and out['calibration_verified'] is False
    return out


def test_same_color_ticks_do_not_change_render_or_authorize_calibration():
    absent = same_color_calibration_pdf('ticks', present=False)
    hidden = same_color_calibration_pdf('ticks')
    assert rendered_calibration_rgb(hidden) == rendered_calibration_rgb(absent)
    assert_no_calibration(absent)
    assert_no_calibration(hidden)


# First observation after tick repair: regression coverage, NOT a new RED cycle.
def test_same_color_ef_does_not_change_render_or_authorize_calibration():
    absent = same_color_calibration_pdf('ef', present=False)
    hidden = same_color_calibration_pdf('ef')
    assert rendered_calibration_rgb(hidden) == rendered_calibration_rgb(absent)
    assert_no_calibration(absent)
    assert_no_calibration(hidden)


# Existing shared-gate regression; not an independent RED/GREEN claim.
def test_same_color_unit_does_not_change_render_or_select_ev_panel():
    absent = same_color_calibration_pdf('unit', present=False)
    hidden = same_color_calibration_pdf('unit')
    assert rendered_calibration_rgb(hidden) == rendered_calibration_rgb(absent)
    assert_no_calibration(absent)
    out = assert_no_calibration(hidden)
    assert out['panels'] == []
    assert out['calibration_candidates'][0]['reason'] == 'missing_ev_unit'


# Adjacent cases of the same gate: first-observation regressions, not RED cycles.
@pytest.mark.parametrize('kind', ['ticks', 'ef', 'unit'])
@pytest.mark.parametrize('background,noise', [(0., True), (.4, False), (.4, True)])
def test_same_color_gray_or_noisy_background_cannot_supply_glyph_evidence(kind, background, noise):
    hidden = same_color_calibration_pdf(kind, background=background, noise=noise)
    absent = same_color_calibration_pdf(kind, present=False, background=background, noise=noise)
    assert rendered_calibration_rgb(hidden) == rendered_calibration_rgb(absent)
    assert_no_calibration(absent)
    assert_no_calibration(hidden)


@pytest.mark.parametrize('kind', ['ticks', 'ef', 'unit'])
@pytest.mark.parametrize('foreground,noise', [(0., False), (0., True), (.4, False), (.4, True)])
def test_matched_visible_text_on_light_background_stays_quantitative(kind, foreground, noise):
    visible = same_color_calibration_pdf(kind, background=.9, foreground=foreground, noise=noise)
    absent = same_color_calibration_pdf(kind, present=False, background=.9, noise=noise)
    assert rendered_calibration_rgb(visible) != rendered_calibration_rgb(absent)
    out = digitize(visible)
    assert out['status'] == 'digitized_unverified'
    panel = out['panels'][0]
    assert panel['energy_mapping']['slope_eV_per_pdf_point'] == pytest.approx(-3/70)
    assert len(panel['tick_evidence']) == 3 and panel['fermi_evidence']['label_text'] == 'EF'
    assert any(t['band_data'] is not None for t in panel['traces'])
    assert panel['calibration_verified'] is False and out['confidence'] is None


def test_rejected_native_ticks_do_not_reenter_ocr_calibration(monkeypatch):
    # Record the real calibration inputs; no OCR or scientific result is mocked.
    original = parser._pdf_calibrate_panel
    seen = []
    def record(panel, drawings, words):
        seen.append([dict(word) for word in words])
        return original(panel, drawings, words)
    monkeypatch.setattr(parser, '_pdf_calibrate_panel', record)
    out = assert_no_calibration(same_color_calibration_pdf('ticks', noise=True))
    assert out['panels'][0]['tick_strip_ocr']['engine'] == 'rapidocr_onnxruntime'
    assert len(seen) == 2  # Native attempt plus actual rendered-raster OCR fallback.
    assert all(not (word['engine'] == 'pymupdf_text' and word['text'] in ('5', '2', '-1'))
               for attempt in seen for word in attempt)


# IR-CLIP-001: exact original pair can be supplied read-only for RED/replay.
def ir_clip_fixture(name):
    import os
    from pathlib import Path
    frozen = os.environ.get('IR_CLIP_FIXTURES')
    if frozen:
        return (Path(frozen) / name).read_bytes()
    if name == 'visible.pdf':
        return same_color_calibration_pdf('ef', background=.9, foreground=0.)
    kind = {'ef_sliver.pdf': 'ef', 'unit_sliver.pdf': 'unit',
            'ticks_sliver.pdf': 'ticks'}.get(name, 'ef')
    with pymupdf.open(stream=same_color_calibration_pdf(kind, background=1., foreground=0.),
                      filetype='pdf') as doc:
        page = doc[0]
        for xref in page.get_contents():
            raw = doc.xref_stream(xref)
            targets = {'ef': [(286, 153, 'EF')], 'unit': [(42, 118, '(eV)')],
                       'ticks': [(64, 83, '5'), (64, 153, '2'), (64, 223, '-1')]}[kind]
            for x, y, text in targets:
                token = ('<' + text.encode().hex() + '>').encode()
                if token not in raw:
                    continue
                narrow_pair = name in ('EF.pdf', 'EE.pdf')
                paths = ' '.join(f'{x+pymupdf.get_text_length(text[:i], fontsize=8)+1} '
                                 f'{169 if narrow_pair else 320-y-2} .5 '
                                 f'{2 if narrow_pair else 12} re' for i in range(len(text)))
                if name == 'EE.pdf':
                    raw = raw.replace(token, b'<4545>')
                doc.update_stream(xref, ('q '+paths+' W n\n').encode()+raw+b'\nQ')
        return doc.tobytes()


def test_ir_clip_pixel_identical_ef_ee_cannot_change_quantitative_eligibility():
    ef, ee = ir_clip_fixture('EF.pdf'), ir_clip_fixture('EE.pdf')
    for scale in (1, 4, 8):
        with pymupdf.open(stream=ef, filetype='pdf') as a, pymupdf.open(stream=ee, filetype='pdf') as b:
            assert a[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale)).samples == b[0].get_pixmap(
                matrix=pymupdf.Matrix(scale, scale)).samples
    # Preserve the matching full-visible positive: no blanket disable.
    control = digitize(ir_clip_fixture('visible.pdf'))
    assert any(t['band_data'] is not None for p in control['panels'] for t in p['traces'])
    outcomes = [digitize(payload) for payload in (ef, ee)]
    counts = [sum(t['band_data'] is not None for p in out['panels'] for t in p['traces'])
              for out in outcomes]
    assert counts == [0, 0], 'IR-CLIP-001: identical clipped glyph pixels use hidden native semantics'


# Shared clipping-gate regressions observed after GREEN; NOT independent TDD cycles.
@pytest.mark.parametrize('name', ['ef_sliver.pdf', 'unit_sliver.pdf', 'ticks_sliver.pdf'])
def test_ir_clip_original_label_slivers_remain_nonquantitative(name):
    assert_no_calibration(ir_clip_fixture(name))


@pytest.mark.parametrize('name', ['EF.pdf', 'EE.pdf', 'visible.pdf', 'unit_sliver.pdf'])
@pytest.mark.parametrize('scale', [.5, 1.5])
def test_ir_clip_nested_scaled_form_preserves_label_qualification(name, scale):
    raw = ir_clip_fixture(name)
    for _ in range(2):
        with pymupdf.open(stream=raw, filetype='pdf') as source, pymupdf.open() as target:
            page = target.new_page(width=400*scale+30, height=320*scale+30)
            page.show_pdf_page(pymupdf.Rect(15, 15, 400*scale+15, 320*scale+15), source, 0)
            raw = target.tobytes()
    out = digitize(raw)
    quantitative = [t for p in out['panels'] for t in p['traces'] if t['band_data'] is not None]
    assert bool(quantitative) == (name == 'visible.pdf')


def ir_clip_control_with_path(path, operator='W', target='EF'):
    with pymupdf.open(stream=ir_clip_fixture('visible.pdf'), filetype='pdf') as doc:
        page = doc[0]
        if target == 'distractor':
            page.insert_text((330, 280), 'prose', fontsize=8)
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            token = b'<4546>' if target == 'EF' else b'<70726f7365>'
            if token in stream:
                doc.update_stream(xref, ('q '+path+' '+operator+' n\n').encode()+stream+b'\nQ')
        return doc.tobytes()


@pytest.mark.parametrize('path,operator', [
    ('286 166 2 10 re', 'W'),  # one narrow rectangle
    ('280 160 30 25 re 289 166 8 10 re', 'W*'),  # containing bbox with a hole
    ('280 160 m 310 160 l 310 185 l 280 185 l 291 169 l h', 'W'),  # concave path
])
def test_ir_clip_nonrectangular_or_partial_clip_cannot_authorize_full_native_ef(path, operator):
    assert_no_calibration(ir_clip_control_with_path(path, operator))


def test_ir_clip_containing_rectangle_and_restored_clip_keep_visible_positive():
    for raw in [ir_clip_control_with_path('280 160 30 25 re'),
                ir_clip_control_with_path('330 38 1 4 re', target='distractor')]:
        out = digitize(raw)
        assert any(t['band_data'] is not None for p in out['panels'] for t in p['traces'])
        evidence = out['panels'][0]['fermi_evidence']['visibility_evidence']['clip_evidence']
        assert evidence['method'] == 'mupdf_paint_order_full_text_bounds_in_rectangular_clip'


def test_ir_clip_rejected_native_labels_do_not_reenter_calibration(monkeypatch):
    original = parser._pdf_calibrate_panel
    seen = []
    def record(panel, drawings, words):
        seen.append([dict(word) for word in words])
        return original(panel, drawings, words)
    monkeypatch.setattr(parser, '_pdf_calibrate_panel', record)
    assert_no_calibration(ir_clip_fixture('EF.pdf'))
    assert seen and all(not (w['engine'] == 'pymupdf_text' and w['text'] == 'EF')
                        for attempt in seen for w in attempt)


@pytest.mark.parametrize('cap', [0, 1, 8, 20])
def test_ir_clip_callback_budget_exhaustion_preserves_valueerror_contract(monkeypatch, cap):
    monkeypatch.setitem(parser._PDF_DIGITIZATION_LIMITS, 'visibility_checks', cap)
    with pytest.raises(ValueError, match='budget.*visibility_checks'):
        digitize(ir_clip_fixture('EF.pdf'))
