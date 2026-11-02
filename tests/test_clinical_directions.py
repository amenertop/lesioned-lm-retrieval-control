"""Clinical direction checks use paired observations and never fill missing evidence with zero."""
import numpy as np
import pandas as pd

from lesion import clinical_directions as cd


def test_direction_uses_expected_sign_and_classification_bounds():
    assert cd.direction_interval(-.4, -1) == (1., 1.)
    assert cd.direction_interval(0., 1) == (.5, .5)
    assert cd.direction_interval(None, 1, -.2, .4) == (0., 1.)
    assert cd.direction_interval(None, -1, -.4, -.1) == (1., 1.)
    assert cd.direction_interval(None, 1, -1e-17, .025) == (.5, 1.)
    assert cd.direction_interval(None, 1) == (None, None)


def test_family_weights_and_missing_coverage():
    rows = [dict(family=family, marker=marker, estimate=1., expected_direction=1)
            for family, markers in cd.MARKERS.items() for marker in markers]
    for row in rows:
        if row['family'] == 'frequency':
            row['estimate'] = -1.
    result = cd.scorecard(rows)
    assert result['lower'] == result['upper'] == .8
    rows[-1]['estimate'] = None
    result = cd.scorecard(rows)
    assert result['n_covered_families'] == 4
    assert result['lower'] is None and result['upper'] is None


def test_cue_uses_subordinate_meaning_and_ties_unresolved_answers():
    records = []
    for context in ('low', 'high'):
        for family in range(4):
            for state in ('sd', 'sa'):
                for cue in ('none', 'correct'):
                    records.append(dict(context=context, dominance='subordinate', family=str(family),
                        state=state, cue=cue, semantic_correct=pd.NA,
                        response_id=f'{family}-{cue}'))
        # Unknown dominant meanings cannot invalidate a subordinate cue contrast.
        records.append(dict(context=context, dominance='dominant', family='extra', state='sd',
                            cue='none', semantic_correct=pd.NA, response_id='unused'))
    rows = pd.DataFrame(records)
    rows['semantic_correct'] = pd.array(rows.semantic_correct, dtype='boolean')
    result = cd.cue_markers(rows, 'sd', 'sa', 0, 1910)
    assert all(row['estimate'] is None for row in result)
    assert all(row['classification_bounds']['lower'] == row['classification_bounds']['upper'] == 0 for row in result)
    assert all(row['n_families'] == 4 for row in result)


def test_associates_use_semantic_errors_and_zero_denominator_stays_missing():
    rows = pd.DataFrame([
        dict(item_id=str(i), state=state, letters=0, category=category)
        for state, categories in [('sd', ['coordinate', 'coordinate', 'unrelated', 'correct']),
                                  ('sa', ['associate', 'superordinate', 'format_invalid', 'correct'])]
        for i, category in enumerate(categories)])
    result = cd.error_marker(rows, 'sd', 'sa', 0)
    assert result['estimate'] == .5
    assert result['denominators'] == {'sd': 2, 'sa': 2}
    rows.loc[rows.state == 'sd', 'category'] = 'correct'
    result = cd.error_marker(rows, 'sd', 'sa', 0)
    assert np.isnan(result['estimate'])


def test_ledger_keeps_classification_bounds_out_of_sampling_intervals():
    rows = [dict(family=family, marker=marker, estimate=None, expected_direction=1,
                 unit='proportion difference', n_families=4,
                 classification_bounds=dict(lower=0., upper=.5, sampling_envelope=dict(lower=-.1, upper=.6)))
            for family, markers in cd.MARKERS.items() for marker in markers]
    rows[-2]['baseline'] = dict(sd=dict(lower=.7, upper=.9), sa=dict(lower=.8, upper=1.))
    rows[-1]['denominators'] = dict(sd=10, sa=6)
    report = cd.ledger_report(dict(meta={}, sd_state='sd', sa_state='sa', n_boot=100, seed=1910, markers=rows))
    value = report['contrasts']['low_context_gain']
    assert value['estimate'] is None and 'ci_low' not in value
    assert value['classification_bounds'] == {'lower': 0., 'upper': .5}
    assert value['sampling_envelope'] == {'lower': -.1, 'upper': .6}
    assert report['directions']['overall']['lower'] == .5
    baseline = report['contrasts']['high_context_gain']['baseline']
    assert baseline['sa']['recovery_room']['lower'] == 0.
    assert np.isclose(baseline['sd']['recovery_room']['upper'], .3)
    assert '10 个' in report['contrasts']['associative_fraction']['note']
    assert '6 个' in report['contrasts']['associative_fraction']['note']
