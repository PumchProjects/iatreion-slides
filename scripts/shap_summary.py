"""Fold-wise, positive-class SHAP summaries on matched available samples."""

import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

GROUP = ['run_id', 'setting', 'task', 'cohort', 'module', 'data_scope', 'outer_fold']
IDENT = [
    'run_id',
    'setting',
    'task',
    'cohort',
    'model',
    'explanation_engine',
    'reference_engine',
    'module',
    'data_scope',
    'outer_fold',
    'output_space',
    'label_version',
    'training_visit_selection',
]


def normalize_importance(values):
    importance = np.mean(np.abs(values), axis=0)
    total = importance.sum()
    relative = importance / total if total > 0 else np.full_like(importance, np.nan)
    return importance, relative


def positive_arrays(z, meta):
    index = list(z['output_names']).index(meta['positive_label'])
    values = z['values'] if z['values'].ndim == 2 else z['values'][:, :, index]
    base = z['base_values']
    if base.ndim == 2:
        base = base[:, index]
    elif len(base) == len(z['output_names']) and len(base) != len(z['sample_id']):
        base = np.full(len(z['sample_id']), base[index])
    reference = z['reference_output'].reshape(len(z['sample_id']), -1)[:, index]
    return values, base, reference


def aggregate_folds(data):
    keys = [k for k in IDENT if k != 'outer_fold'] + ['output']
    rows = []
    for key, g in data.groupby(keys, dropna=False, sort=False):
        folds = g.outer_fold.nunique()
        f = (
            g.groupby('feature')
            .agg(
                relative_sum=('relative_importance', 'sum'),
                squared_sum=('relative_importance', lambda x: (x * x).sum()),
                raw_sum=('mean_abs_shap', 'sum'),
            )
            .reset_index()
        )
        f['relative_mean'] = f.relative_sum / folds
        f['relative_sd'] = np.sqrt(
            np.maximum(0, f.squared_sum / folds - f.relative_mean**2)
        )
        f['mean_abs_shap'] = f.raw_sum / folds
        f['n_folds'] = folds
        f['n_explained'] = g.drop_duplicates('outer_fold').n_explained.sum()
        f['same_feature_set'] = g.same_feature_set.all()
        f['rank'] = f.relative_mean.rank(method='min', ascending=False)
        for name, value in zip(keys, key, strict=True):
            f[name] = value
        rows.append(f.drop(columns=['relative_sum', 'squared_sum', 'raw_sum']))
    return pd.concat(rows, ignore_index=True)


def summarize(source, runs, root, save):
    bundles = []
    pools = defaultdict(list)
    feature_sets = defaultdict(set)
    valid_groups = defaultdict(list)
    for path in sorted((source / 'shap').glob('**/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['run_id'] not in runs:
            continue
        meta.setdefault('outer_fold', -1)
        meta.setdefault(
            'positive_label', 'fast' if meta['task'] == 'MMSE' else meta['task'] + '+'
        )
        assert meta['explanation_engine'] == meta['reference_engine'], path
        key = tuple(meta[k] for k in GROUP)
        with np.load(path, allow_pickle=False) as z:
            available = (
                z['available']
                if 'available' in z
                else np.ones(len(z['sample_id']), bool)
            )
            pools[key].append(set(z['sample_id'][available].astype(str)))
            signature = hashlib.sha256(
                '\n'.join(sorted(z['feature_names'])).encode()
            ).hexdigest()
            feature_sets[key].add(signature)
            values, base, reference = positive_arrays(z, meta)
            error = np.abs(base + values.sum(axis=1) - reference)[available]
            valid_groups[key].append(
                bool(np.isfinite(error).all() and (error <= 1e-4).all())
            )
        bundles.append((path, meta, key, signature))
    common = {key: set.intersection(*sets) for key, sets in pools.items()}
    root.mkdir(parents=True, exist_ok=True)
    global_rows, validation, manifest = [], [], []
    with gzip.open(root / 'shap_values.csv.gz', 'wt', encoding='utf-8') as stream:
        first = True
        for path, meta, key, signature in bundles:
            with np.load(path, allow_pickle=False) as z:
                ids = z['sample_id'].astype(str)
                available = (
                    z['available'] if 'available' in z else np.ones(len(ids), bool)
                )
                use = available & np.isin(ids, sorted(common[key]))
                values, base, reference = positive_arrays(z, meta)
                features, x = z['feature_names'], z['data']
                identity = {k: meta[k] for k in IDENT}
                sample = dict(
                    source_file=str(path),
                    source_sha256=meta['source_sha256'],
                    feature_signature=signature,
                    same_feature_set=len(feature_sets[key]) == 1,
                    output=meta['positive_label'],
                    n_available=int(available.sum()),
                    n_explained=int(use.sum()),
                    n_sampled=len(ids),
                    sample_policy='common_available_across_models',
                )
                error = np.abs(base + values.sum(axis=1) - reference)[available]
                sample['max_additivity_error'] = (
                    float(np.max(error)) if len(error) else np.nan
                )
                sample['reference_available'] = bool(
                    len(error) and np.isfinite(error).all()
                )
                sample['additivity_passed'] = bool(
                    np.isfinite(error).all() and (error <= 1e-4).all()
                )
                sample['comparison_eligible'] = len(feature_sets[key]) == 1 and all(
                    valid_groups[key]
                )
                sample['comparison_exclusion'] = (
                    'reference_output_unavailable'
                    if not sample['reference_available']
                    else 'feature_set_differs'
                    if len(feature_sets[key]) != 1
                    else 'group_additivity_difference'
                    if not all(valid_groups[key])
                    else ''
                )
                manifest.append(dict(**identity, **sample))
                reconstruction = base + values.sum(axis=1)
                check = pd.DataFrame(
                    dict(
                        subject_id=ids[available],
                        shap_sum=reconstruction[available],
                        reference_output=reference[available],
                        absolute_error=np.abs(reconstruction - reference)[available],
                    )
                )
                for k, v in {
                    **identity,
                    'reference_method': meta.get('reference_method', 'frozen_model'),
                }.items():
                    check[k] = v
                validation.append(check)
                if not use.any():
                    continue
                importance, relative = normalize_importance(values[use])
                g = pd.DataFrame(
                    dict(
                        feature=features,
                        mean_abs_shap=importance,
                        mean_shap=values[use].mean(axis=0),
                        relative_importance=relative,
                    )
                )
                for k, v in {**identity, **sample}.items():
                    g[k] = v
                global_rows.append(g)
                frame = pd.DataFrame(
                    dict(
                        subject_id=np.repeat(ids[use], len(features)),
                        feature=np.tile(features, use.sum()),
                        feature_value=x[use].ravel(),
                        shap_value=values[use].ravel(),
                        base_value=np.repeat(base[use], len(features)),
                        reference_output=np.repeat(reference[use], len(features)),
                        output=meta['positive_label'],
                        feature_value_space='model_transformed',
                    )
                )
                for k, v in identity.items():
                    frame[k] = v
                frame['additivity_passed'] = sample['additivity_passed']
                frame['comparison_eligible'] = sample['comparison_eligible']
                frame.to_csv(stream, index=False, header=first)
                first = False
    global_data = pd.concat(global_rows, ignore_index=True)
    summary = aggregate_folds(global_data[global_data.comparison_eligible])
    save(
        root,
        'shap_global',
        global_data,
        '阳性类、共同可预测样本的每折原始及归一化 SHAP；分母包含所有特征。',
    )
    save(
        root,
        'shap_summary',
        summary,
        '先折内归一化再等权汇总，SD 为折间总体标准差；training 单独汇总。',
    )
    save(
        root,
        'shap_validation',
        pd.concat(validation, ignore_index=True),
        '原尺度加和与独立预测核对；外层 XGBoost 由保存概率反算 logit。',
    )
    save(
        root,
        'shap_bundles',
        pd.DataFrame(manifest),
        '每包解释样本量、排除人数、共同样本、特征集合和来源。',
    )
    save(
        root,
        'shap_exclusions',
        pd.DataFrame(manifest).query('not comparison_eligible'),
        (
            '跨模型比较排除整组：特征集合不同，或规则输出与历史 SHAP 加和不一致。'
            '原始解释仍保留。'
        ),
    )
