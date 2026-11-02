"""Remove semantic descriptions while keeping the existing M3 letter cues fixed."""
import hashlib

import numpy as np
import pandas as pd

from . import clinical_alignment, m3_cued, m3_scoring as ms


def prompts(items, chat=True):
    """Original description, removed description, and explicit word completion at the same cue lengths.

    The removed-description condition keeps the original instruction and demos. The explicit completion
    condition changes both instruction and demos, measuring the combined task framework.
    Outputs without a description measure target-word completion, not recovered semantic knowledge.
    """
    original = {(p.item_id, p.condition): p for p in m3_cued.prompts(items, chat)}
    naming_instruction = ms.NAMING_INSTRUCTION + (ms.NAMING_REPLY_RULE if chat else '')
    completion_instruction = 'Complete each beginning to make an English word.' + (ms.NAMING_REPLY_RULE if chat else '')
    out = []
    for row in items.to_dict('records'):
        word, item = row['target'], row['item_id']
        if len(word) < 3:
            raise ValueError('cue content comparison needs words of at least three letters')
        baseline = original[item, 'cue0']
        out.append(ms.NamingPrompt(item, 'definition_none', baseline.text, baseline.candidates))
        for length, j in [('two', 2), ('final', len(word)-1)]:
            source = original[item, f'cue{j}']
            out.append(ms.NamingPrompt(item, 'definition_'+length, source.text, source.candidates))
            demos = [ms._naming_block(d['definition'], d['target'][:min(j, len(d['target'])-1)], d['target'])
                     for d in ms.DEMOS]
            block = ms.HINT.format(xx=word[:j])+'\nAnswer:'
            text = '\n\n'.join([naming_instruction, *demos, block])
            out.append(ms.NamingPrompt(item, 'removed_'+length, text, source.candidates))
            demos = [ms.HINT.format(xx=d['target'][:min(j, len(d['target'])-1)])+'\nAnswer: '+d['target']
                     for d in ms.DEMOS]
            text = '\n\n'.join([completion_instruction, *demos, block])
            out.append(ms.NamingPrompt(item, 'completion_'+length, text, source.candidates))
    return out


def classify(generated, items, review=None, chat=True):
    """Separate semantic judgements from exact target completion. Keep raw strings unchanged."""
    if generated.duplicated(['state', 'item_id', 'condition']).any():
        raise ValueError('duplicate cue-content observations')
    rows = generated.merge(items, on='item_id', validate='many_to_one')
    rows['normalized'] = rows.answer.str.lower().str.strip().str.strip('.!,?"\'')
    rows['target_exact'] = rows.normalized.eq(rows.target)
    demo_words = [demo['target'] for demo in ms.DEMOS]
    rows['demo_sequence_replay'] = rows.normalized.str.split().apply(lambda words: words == demo_words)
    rows['prefix_cluster'] = rows.target.str[:2]
    rows['n_letters'] = rows.target.str.len()
    rows['length'] = rows.condition.str.rsplit('_', n=1).str[-1]
    rows['letters'] = [0 if length == 'none' else 2 if length == 'two' else len(word)-1
                       for length, word in zip(rows.length, rows.target)]
    source_prompts = {(p.item_id, p.condition): p.text for p in prompts(items, chat)}
    rows['prompt_id'] = [hashlib.sha256(source_prompts[item, condition].encode()).hexdigest()
                         for item, condition in zip(rows.item_id, rows.condition)]
    rows['category'] = 'unclassified'
    rows['semantic_correct'] = pd.Series(pd.NA, index=rows.index, dtype='boolean')
    definition = generated[generated.condition.str.startswith('definition_')].copy()
    definition['content_condition'] = definition.condition
    lengths = rows.set_index(['state', 'item_id', 'condition']).letters
    definition['condition'] = ['cue'+str(lengths.loc[state, item, condition])
                              for state, item, condition in definition[['state', 'item_id', 'condition']].itertuples(index=False, name=None)]
    judged = clinical_alignment.classify_cued(definition, items)
    if review is not None:
        used = review[review.response_id.isin(judged.response_id)]
        judged = clinical_alignment.classify_cued(definition, items, used)
    categories = judged.set_index(['state', 'item_id', 'content_condition']).category
    for i, row in rows.iterrows():
        if row.condition.startswith('definition_'):
            category = categories.loc[row.state, row.item_id, row.condition]
            rows.loc[i, 'category'] = category
            if category != 'unclassified':
                rows.loc[i, 'semantic_correct'] = category == 'correct'
        elif not row.normalized:
            rows.loc[i, 'category'] = 'empty'
        elif not row.normalized.isascii() or not row.normalized.isalpha():
            rows.loc[i, 'category'] = 'format_invalid'
        elif row.target_exact:
            rows.loc[i, 'category'] = 'target'
        elif row.normalized.startswith(row.target[:row.letters]) and len(row.normalized) > row.letters:
            rows.loc[i, 'category'] = 'other_completion'
        else:
            rows.loc[i, 'category'] = 'invalid_completion'
    # other_completion is a prefix-compatible string, not a dictionary judgement or a semantic error.
    return rows, judged[judged.category == 'unclassified'][['response_id', 'target', 'synset', 'definition', 'answer']].drop_duplicates('response_id')


def summarize(rows, candidates, n_boot=2000, seed=1910):
    """Cluster identical short prefixes and keep each state's original semantic error set fixed."""
    scores = candidates[candidates.role == 'target'][['state', 'item_id', 'condition', 'logprob', 'n_tokens']]
    rows = rows.merge(scores, on=['state', 'item_id', 'condition'], validate='one_to_one')
    if (rows.n_tokens <= 0).any() or (rows.groupby('item_id').n_tokens.nunique() != 1).any():
        raise ValueError('target token counts must be positive and identical across conditions')
    rows['logprob_per_token'] = rows.logprob / rows.n_tokens
    summaries, contrasts, error_sets = {}, {}, {}
    conditions = {f'{task}_{length}' for task in ('definition', 'removed', 'completion') for length in ('two', 'final')} | {'definition_none'}
    for state, group in rows.groupby('state', sort=True):
        ids = sorted(group.item_id.unique())
        if set(group.condition) != conditions or any(set(part.item_id) != set(ids) for _, part in group.groupby('condition')):
            raise ValueError('all cue-content conditions require the same concepts')
        baseline = group[group.condition == 'definition_none'].set_index('item_id').reindex(ids)
        known = baseline.semantic_correct.notna().all()
        errors = ~baseline.semantic_correct.to_numpy(bool) if known else None
        error_sets[state] = dict(status='reviewed' if known else 'needs_blind_response_review',
                                 item_ids=baseline.index[errors].tolist() if known else None)
        prefixes = pd.Categorical(baseline.prefix_cluster)
        counts = len(prefixes.categories)
        draws = np.random.default_rng(seed).multinomial(counts, np.full(counts, 1/counts), size=n_boot)
        membership = np.eye(counts)[prefixes.codes]

        def estimate(values, mask):
            if mask is None:
                return dict(estimate=None, ci_low=None, ci_high=None, n=None, n_unit='概念', status='baseline_needs_review')
            values = np.asarray(values, dtype=float)
            n = int(mask.sum())
            if not n or not np.isfinite(values[mask]).all():
                return dict(estimate=None, ci_low=None, ci_high=None, n=n, n_unit='概念', status='missing_or_unreviewed')
            numerator = membership.T @ np.where(mask, values, 0.)
            denominator = membership.T @ mask.astype(float)
            total = draws @ denominator
            boot = np.divide(draws @ numerator, total, out=np.full(n_boot, np.nan), where=total > 0)
            valid = boot[np.isfinite(boot)]
            ci = np.quantile(valid, [.025, .975]).tolist() if n_boot and len(valid) >= .9*n_boot else [None, None]
            return dict(estimate=float(values[mask].mean()), ci_low=ci[0], ci_high=ci[1], n=n, n_unit='概念',
                        status='estimated' if ci[0] is not None else 'point_only_or_sparse')

        long_words = baseline.n_letters.to_numpy() > 3
        subsets = dict(all=np.ones(len(ids), dtype=bool), initial_errors=errors, length_gt3=long_words,
                       initial_errors_length_gt3=errors & long_words if known else None)
        paired = {metric: group.pivot(index='item_id', columns='condition', values=metric).reindex(ids)
                  for metric in ('target_exact', 'logprob', 'logprob_per_token')}
        for name, mask in subsets.items():
            summary, difference = {}, {}
            for condition, part in group.groupby('condition'):
                part = part.set_index('item_id').reindex(ids)
                metrics = dict(target_exact=dict(**estimate(part.target_exact, mask), unit='proportion'),
                               target_logprob=dict(**estimate(part.logprob, mask), unit='nat'),
                               target_logprob_per_token=dict(**estimate(part.logprob_per_token, mask), unit='nat/token'))
                metrics['format_invalid'] = dict(**estimate(part.category.eq('format_invalid'), mask), unit='proportion')
                metrics['demo_sequence_replay'] = dict(**estimate(part.demo_sequence_replay, mask), unit='proportion')
                if condition.startswith('definition_'):
                    metrics['semantic_correct'] = dict(**estimate(part.semantic_correct.to_numpy(dtype=float, na_value=np.nan), mask), unit='proportion')
                summary[condition] = metrics
            for length in ('two', 'final'):
                for first, second in [('removed', 'definition'), ('completion', 'removed')]:
                    for metric, unit in [('target_exact', 'proportion difference'), ('logprob', 'nat'), ('logprob_per_token', 'nat/token')]:
                        value = paired[metric][first+'_'+length].astype(float)-paired[metric][second+'_'+length].astype(float)
                        difference[f'{first}_minus_{second}_{length}/{metric}'] = dict(**estimate(value, mask), unit=unit)
            summaries[state+'/'+name], contrasts[state+'/'+name] = summary, difference
    prompt_counts = rows[['condition', 'item_id', 'prompt_id']].drop_duplicates().groupby('condition').agg(n_concepts=('item_id', 'size'), n_unique_prompts=('prompt_id', 'nunique'))
    return dict(meta=dict(n_boot=n_boot, seed=seed, resampling='same first-two-letter clusters across conditions',
                         error_sets=error_sets, prompts=prompt_counts.reset_index().to_dict('records'),
                         posthoc_response_diagnostics=['format_invalid', 'demo_sequence_replay'],
                         demo_sequence_replay='entire normalized answer consists of the three demonstration targets in order',
                         other_completion='prefix-compatible alphabetic string, lexical validity not automatically asserted'),
                summaries=summaries, contrasts=contrasts)
