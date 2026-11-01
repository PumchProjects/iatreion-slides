"""Collect schema-v2 predictions and explanations into D5 and D6."""

import argparse
import json
import tomllib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.metrics import calibration, curves, metric_table, paired_difference

KEYS = [
    'run_id',
    'setting',
    'model',
    'stage',
    'cohort',
    'subset',
    'train_cohort',
    'feature_scheme',
    'task',
]


def save(root, name, df, description):
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / (name + '.csv'), index=False, encoding='utf-8-sig')
    p = root / 'tables.json'
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta[name] = dict(description=description, rows=len(df))
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def predictions(source, runs, labels):
    rows = []
    for path in sorted((source / 'predictions').glob('**/predictions.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['run_id'] not in runs:
            continue
        with np.load(path, allow_pickle=False) as z:
            p = z['y_score'][:, meta['positive_index']]
            d = pd.DataFrame(
                dict(
                    subject_id=z['sample_id'].astype(str),
                    y_true=np.where(
                        z['y_true'] < 0,
                        np.nan,
                        (z['y_true'] == meta['positive_index']).astype(int),
                    ),
                    probability=p,
                    y_pred=(z['y_pred'] == meta['positive_index']).astype(int),
                    unavailable=z['y_mask'].astype(bool),
                    outer_fold=z['outer_fold'],
                    inner_fold=z['inner_fold'],
                    record_kind=z['kind'],
                    threshold=z['threshold'],
                )
            )
            if 'module_names' in z:
                for j, name in enumerate(z['module_names']):
                    d['p_raw__' + name] = z['module_raw'][:, j]
                    d['p_calibrated__' + name] = z['module_calibrated'][:, j]
                    d['missing__' + name] = z['module_mask'][:, j]
        for key in KEYS:
            d[key] = meta[key]
        d['tsh_variant'] = (
            'excluded'
            if '-no-tsh' in meta['subset']
            else (
                'included'
                if any(x in meta['subset'] for x in ['nonimaging', 'labdata'])
                else 'not_applicable'
            )
        )
        d['decision_rule'] = (
            'argmax' if meta['threshold_source'] == 'fixed_0.5' else '>='
        )
        d['model_label_version'] = meta['label_version']
        d['threshold_source'] = meta['threshold_source']
        d['source_prediction'] = str(path)
        d['configuration'] = '|'.join(str(meta[k]) for k in KEYS)
        d.loc[d.unavailable, ['probability', 'y_pred']] = np.nan
        rows.append(d)
    data = pd.concat(rows, ignore_index=True)
    assert not data.duplicated(['configuration', 'subject_id']).any()
    if labels is not None:
        labels = labels[labels.definition == 'main'][
            ['cohort', 'subject_id', 'label', 'label_version']
        ]
        labels = labels.rename(
            columns={
                'label': 'y_true_baseline',
                'label_version': 'baseline_label_version',
            }
        )
        data = data.merge(
            labels, on=['cohort', 'subject_id'], how='left', validate='many_to_one'
        )
        data.loc[data.task != 'MMSE', ['y_true_baseline', 'baseline_label_version']] = (
            np.nan
        )
    return data


def evaluate(data, root, resamples, seed):
    metrics = []
    cm = []
    cal = []
    calcurves = []
    rocrows = []
    prrows = []
    for configuration, g in data.groupby('configuration', sort=False):
        ident = {k: g.iloc[0][k] for k in KEYS}
        ident['configuration'] = configuration
        for label_column in ['y_true', 'y_true_baseline']:
            if label_column not in g:
                continue
            label_version = (
                g.iloc[0]['model_label_version']
                if label_column == 'y_true'
                else 'baseline_ols_le3_v1'
            )
            eligible = g[g[label_column].notna()]
            d = eligible[~eligible.unavailable & eligible.probability.notna()]
            if not len(d):
                continue
            y = d[label_column].to_numpy(dtype=int)
            p = d.probability.to_numpy()
            base = dict(
                **ident,
                evaluation_label_version=label_version,
                n_eligible=len(eligible),
                n_evaluated=len(d),
                events=int(y.sum()),
                coverage=len(d) / len(eligible),
            )
            for threshold_mode in ['frozen', '0.5']:
                pred = (
                    d.y_pred.to_numpy(dtype=int)
                    if threshold_mode == 'frozen'
                    else (p >= 0.5).astype(int)
                )
                table, counts = metric_table(y, p, pred, resamples, seed)
                for k, v in {**base, 'threshold_mode': threshold_mode}.items():
                    table[k] = v
                metrics.append(table)
                cm.append(dict(**base, threshold_mode=threshold_mode, **counts))
            if len(np.unique(y)) == 2:
                diagnostic, curve = calibration(y, p)
                cal.append(dict(**base, **diagnostic))
                for k, v in base.items():
                    curve[k] = v
                calcurves.append(curve)
                roc, pr = curves(y, p)
                for k, v in base.items():
                    roc[k] = v
                    pr[k] = v
                rocrows.append(roc)
                prrows.append(pr)
        print('Metrics:', configuration, flush=True)
    outputs = {
        'metrics': (
            metrics,
            '每配置、标签版本和阈值口径的点估计及受试者 bootstrap 95% 区间。',
        ),
        'confusion': (cm, 'TP/TN/FP/FN、标签分母、可预测分母及覆盖率。'),
        'calibration': (
            cal,
            '校准截距（固定斜率为 1）、校准斜率及联合截距；对冻结概率作诊断。',
        ),
        'calibration_curve': (
            calcurves,
            '按预测概率十分位分箱的实际事件率与平均预测风险。',
        ),
        'roc_curve': (rocrows, '完整 ROC 阈值点。'),
        'pr_curve': (prrows, '完整精确率—召回率阈值点。'),
    }
    for name, (parts, description) in outputs.items():
        df = (
            pd.concat(parts, ignore_index=True)
            if parts and isinstance(parts[0], pd.DataFrame)
            else pd.DataFrame(parts)
        )
        save(root, name, df, description)
    paired = []
    groupcols = ['run_id', 'setting', 'stage', 'cohort', 'subset', 'task']
    for keys, g in data.groupby(groupcols, sort=False):
        for a, b in combinations(sorted(g.model.unique()), 2):
            left = g[(g.model == a) & ~g.unavailable].set_index('subject_id')
            right = g[(g.model == b) & ~g.unavailable].set_index('subject_id')
            common = left.index.intersection(right.index).sort_values()
            for col in ['y_true', 'y_true_baseline']:
                if col not in g:
                    continue
                ids = common[
                    left.loc[common, col].notna() & right.loc[common, col].notna()
                ]
                if not len(ids):
                    continue
                x, z = left.loc[ids], right.loc[ids]
                assert np.array_equal(x[col], z[col])
                table = paired_difference(
                    x[col].astype(int),
                    x.probability,
                    x.y_pred.astype(int),
                    z.probability,
                    z.y_pred.astype(int),
                    resamples,
                    seed,
                )
                for k, v in {
                    **dict(zip(groupcols, keys, strict=True)),
                    'model_a': a,
                    'model_b': b,
                    'label_column': col,
                }.items():
                    table[k] = v
                paired.append(table)
    save(
        root,
        'paired_comparisons',
        pd.concat(paired, ignore_index=True)
        if paired
        else pd.DataFrame(columns=['model_a', 'model_b', 'metric', 'difference']),
        '同一配置、共同可预测受试者的配对 bootstrap 指标差（模型 A − 模型 B）。',
    )


def explanations(source, runs, root):
    long = []
    global_rows = []
    checks = []
    validation = []
    for path in sorted((source / 'shap').glob('**/*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['run_id'] not in runs:
            continue
        with np.load(path, allow_pickle=False) as z:
            values = z['values']
            base = z['base_values']
            ids = z['sample_id'].astype(str)
            features = z['feature_names']
            outputs = z['output_names']
            x = z['data']
            reference = z['reference_output'].reshape(len(ids), len(outputs))
            if values.ndim == 2:
                values = values[:, :, None]
            if base.ndim == 1:
                base = (
                    np.broadcast_to(base, (len(ids), len(outputs)))
                    if len(base) == len(outputs)
                    else base[:, None]
                )
            for k, output in enumerate(outputs):
                g = pd.DataFrame(
                    dict(
                        feature=features,
                        mean_abs_shap=np.nanmean(np.abs(values[:, :, k]), axis=0),
                        mean_shap=np.nanmean(values[:, :, k], axis=0),
                        output=output,
                        n_explained=len(ids),
                    )
                )
                for key, value in meta.items():
                    g[key] = value
                global_rows.append(g)
                frame = pd.DataFrame(
                    dict(
                        subject_id=np.repeat(ids, len(features)),
                        feature=np.tile(features, len(ids)),
                        feature_value=x.ravel(),
                        shap_value=values[:, :, k].ravel(),
                        base_value=np.repeat(base[:, k], len(features)),
                        output=output,
                    )
                )
                for key in [
                    'run_id',
                    'setting',
                    'model',
                    'module',
                    'cohort',
                    'task',
                    'label_version',
                    'training_visit_selection',
                    'data_scope',
                    'output_space',
                ]:
                    frame[key] = meta[key]
                frame['feature_value_space'] = 'model_transformed'
                long.append(frame)
                reconstructed = base[:, k] + np.sum(values[:, :, k], axis=1)
                for sid, p, q in zip(ids, reconstructed, reference[:, k], strict=True):
                    validation.append(
                        dict(
                            run_id=meta['run_id'],
                            setting=meta['setting'],
                            model=meta['model'],
                            module=meta['module'],
                            subject_id=sid,
                            output=output,
                            shap_sum=p,
                            reference_output=q,
                            absolute_error=abs(p - q),
                            reference_model='parsed_RRL'
                            if meta['model'] == 'rrl'
                            else 'frozen_tree',
                            output_space=meta['output_space'],
                        )
                    )
                for sid, p in zip(ids, reconstructed, strict=True):
                    checks.append(
                        dict(
                            run_id=meta['run_id'],
                            setting=meta['setting'],
                            model=meta['model'],
                            module=meta['module'],
                            subject_id=sid,
                            output=output,
                            reconstructed_output=p,
                            output_space=meta['output_space'],
                        )
                    )
    root.mkdir(parents=True, exist_ok=True)
    save(
        root,
        'shap_validation',
        pd.DataFrame(validation),
        'SHAP 加和输出与相同样本、相同尺度的单模态原始预测逐人核对。',
    )
    pd.concat(long, ignore_index=True).to_csv(
        root / 'shap_values.csv.gz', index=False, compression='gzip'
    )
    save(
        root,
        'shap_global',
        pd.concat(global_rows, ignore_index=True),
        '按训练解释样本计算的 SHAP 全局汇总。',
    )
    save(
        root,
        'shap_additivity',
        pd.DataFrame(checks),
        '基础值加 SHAP 贡献之和；用于与冻结模型同尺度预测核对。',
    )
    rule_tables = []
    for path in sorted((source / 'rules').glob('**/*.csv')):
        run, setting, cohort = path.relative_to(source / 'rules').parts[:3]
        if run not in runs:
            continue
        d = pd.read_csv(path)
        d['run_id'] = run
        d['setting'] = setting
        d['cohort'] = cohort
        d['subset'] = path.stem
        d['source_file'] = str(path)
        d['valid_rate'] = d['N Valid'] / d['N']
        d['active_rate_total'] = d['N Active'] / d['N']
        d['active_rate_valid'] = d['N Active'] / d['N Valid'].replace(0, np.nan)
        d['rule_length'] = d['Rule'].str.count(r' and | or ') + 1
        rule_tables.append(d)
    save(
        root,
        'rules',
        pd.concat(rule_tables, ignore_index=True),
        '冻结 RRL 规则在各外部队列的有效性、激活、事件和效应统计；保留原始规则表达。',
    )


def dictionary(root):
    descriptions = {
        'probability': '阳性类预测概率；融合配置为校准融合后的概率',
        'y_true': '训练运行采用的原始标签，1 为快进展',
        'y_true_baseline': '预测基线起算、OLS 斜率 ≤−3 的新标签',
        'threshold': '每名受试者实际使用的开发侧阈值',
        'unavailable': '没有可用预测；概率及类别留空',
        'record_kind': 'outer_test 为外层测试，external 为冻结模型外部评价',
        'value': '指标点估计，概率和比例采用 0–1，未乘 100',
        'ci_low': '95% bootstrap 区间下界',
        'ci_high': '95% bootstrap 区间上界',
        'bootstrap_valid': '该指标产生有限估计的重抽样次数',
        'configuration': '运行、处理设置、模型、验证阶段、队列和特征方案的组合',
        'feature_value_space': 'SHAP 特征值为模型变换后的空间；原始单位及变换见 D3',
        'data_scope': '解释样本来源，training 为最终模型训练人群',
        'output_space': '解释的模型输出尺度，如 probability 或 logit',
        'model_label_version': '开发模型时使用的标签版本',
        'evaluation_label_version': '计算本行指标采用的标签版本',
        'clinical_ad_confirmed': '经临床来源确认的 AD 诊断',
        'coverage': '可预测且标签有效的人数 / 标签有效人数',
        'difference': '模型 A 指标减模型 B 指标',
        'threshold_mode': 'frozen 为开发侧冻结阈值；0.5 为固定概率阈值',
    }
    rows = []
    for path in root.glob('*.csv*'):
        if path.name == 'dictionary.csv':
            continue
        d = pd.read_csv(path, nrows=5)
        for c in d:
            desc = descriptions.get(c, c.replace('_', ' '))
            if c.startswith('p_raw__'):
                desc = '单模态原始阳性概率：' + c.split('__')[1]
            elif c.startswith('p_calibrated__'):
                desc = '开发侧 Platt 校准后的单模态阳性概率：' + c.split('__')[1]
            elif c.startswith('missing__'):
                desc = '该模态不可用标记：' + c.split('__')[1]
            rows.append(
                dict(table=path.name, field=c, dtype=str(d[c].dtype), description=desc)
            )
    pd.DataFrame(rows).to_csv(
        root / 'dictionary.csv', index=False, encoding='utf-8-sig'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.toml'))
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--tables', nargs='+', default=['D5', 'D6'])
    parser.add_argument('--bootstrap', type=int)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    source = Path(config['exports']['local_root'])
    runs = args.runs or config['evaluation']['runs']
    output = args.output or Path(config['evaluation']['output'])
    labels = (
        pd.read_csv(args.labels, dtype={'subject_id': str}) if args.labels else None
    )
    if 'D5' in args.tables:
        data = predictions(source, runs, labels)
        save(
            output / 'D5',
            'predictions',
            data,
            '一名受试者、运行、模型和配置的内部 OOF 或外部冻结预测。',
        )
        evaluate(
            data,
            output / 'D5',
            args.bootstrap or config['evaluation']['bootstrap'],
            config['evaluation']['seed'],
        )
        dictionary(output / 'D5')
    if 'D6' in args.tables:
        explanations(source, runs, output / 'D6')
        dictionary(output / 'D6')


if __name__ == '__main__':
    main()
