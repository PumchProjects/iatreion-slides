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
    'inference_engine',
    'prediction_kind',
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
        d['training_visit_selection'] = meta.get('training_visit_selection', '')
        d['evaluation_visit_selection'] = meta.get('visit_selection', '')
        d['source_sha256'] = meta.get('source_sha256', '')
        d['prediction_implementation'] = meta['inference_engine']
        d['fusion_fit_engine'] = meta['fusion_fit_engine']
        d['fusion_artifacts'] = json.dumps(
            meta.get('fusion_artifacts', []), sort_keys=True
        )
        d['source_log'] = meta.get('source_log', '')
        d['source_log_sha256'] = meta.get('source_log_sha256', '')
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
            if (
                label_column == 'y_true_baseline'
                and g.model_label_version.eq('baseline_ols_le3_v1').all()
            ):
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
    groupcols = [
        'run_id',
        'setting',
        'stage',
        'cohort',
        'subset',
        'task',
        'prediction_kind',
    ]
    primary = data[(data.model != 'rrl') | (data.inference_engine == 'rrl_parser')]
    for keys, g in primary.groupby(groupcols, sort=False):
        for a, b in combinations(sorted(g.model.unique()), 2):
            left = g[(g.model == a) & ~g.unavailable].set_index('subject_id')
            right = g[(g.model == b) & ~g.unavailable].set_index('subject_id')
            common = left.index.intersection(right.index).sort_values()
            for col in ['y_true', 'y_true_baseline']:
                if col not in g:
                    continue
                if (
                    col == 'y_true_baseline'
                    and g.model_label_version.eq('baseline_ols_le3_v1').all()
                ):
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
                    'inference_engine_a': x.inference_engine.iloc[0],
                    'inference_engine_b': z.inference_engine.iloc[0],
                    'n_common': len(ids),
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

    compare_implementations(data, root, resamples, seed)


def compare_implementations(data, root, resamples, seed):
    keys = ['run_id', 'setting', 'stage', 'cohort', 'subset', 'task', 'prediction_kind']
    rows, differences = [], []
    for key, group in data[data.model == 'rrl'].groupby(keys, sort=False):
        left = group[
            (group.inference_engine == 'rrl_parser') & ~group.unavailable
        ].set_index('subject_id')
        right = group[
            (group.inference_engine == 'rrl_neural') & ~group.unavailable
        ].set_index('subject_id')
        common = left.index.intersection(right.index).sort_values()
        for column in ['y_true', 'y_true_baseline']:
            if column not in group:
                continue
            if (
                column == 'y_true_baseline'
                and group.model_label_version.eq('baseline_ols_le3_v1').all()
            ):
                continue
            ids = common[
                left.loc[common, column].notna() & right.loc[common, column].notna()
            ]
            if not len(ids):
                continue
            a, b = left.loc[ids], right.loc[ids]
            np.testing.assert_array_equal(a[column], b[column])
            np.testing.assert_array_equal(a.outer_fold, b.outer_fold)
            np.testing.assert_allclose(a.threshold, b.threshold)
            delta = a.probability.to_numpy() - b.probability.to_numpy()
            ident = dict(
                zip(keys, key, strict=True),
                label_column=column,
                n_common=len(ids),
                implementation_a='rrl_parser',
                implementation_b='rrl_neural',
            )
            rows.append(
                dict(
                    **ident,
                    mean_probability_difference=delta.mean(),
                    mean_absolute_probability_difference=np.abs(delta).mean(),
                    max_absolute_probability_difference=np.abs(delta).max(),
                    classification_agreement=np.mean(
                        a.y_pred.to_numpy() == b.y_pred.to_numpy()
                    ),
                )
            )
            table = paired_difference(
                a[column].astype(int),
                a.probability,
                a.y_pred.astype(int),
                b.probability,
                b.y_pred.astype(int),
                resamples,
                seed,
            )
            table = table[table.metric.isin(['AUROC', 'AP', 'Brier'])].copy()
            for k, value in ident.items():
                table[k] = value
            differences.append(table)
    save(
        root,
        'implementation_agreement',
        pd.DataFrame(
            rows,
            columns=keys
            + [
                'label_column',
                'n_common',
                'implementation_a',
                'implementation_b',
                'mean_probability_difference',
                'mean_absolute_probability_difference',
                'max_absolute_probability_difference',
                'classification_agreement',
            ],
        ),
        '相同受试者、标签、折分和冻结阈值下，parser 与 neural 的概率差和分类一致率。',
    )
    save(
        root,
        'implementation_differences',
        pd.concat(differences, ignore_index=True)
        if differences
        else pd.DataFrame(columns=keys + ['metric', 'difference']),
        '实现差值为 parser − neural；共同受试者配对 bootstrap。',
    )


def explanations(source, runs, root):
    from scripts.shap_summary import summarize

    summarize(source, runs, root, save)
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
        pd.concat(rule_tables, ignore_index=True)
        if rule_tables
        else pd.DataFrame(columns=['explanation_engine', 'Rule']),
        '冻结 RRL 规则在各外部队列的有效性、激活、事件和效应统计；保留原始规则表达。',
    )


def dictionary(root):
    descriptions = {
        'comparison_eligible': '该折跨模型比较组特征集合一致且原尺度加和通过核对',
        'comparison_exclusion': (
            '整组排除原因：特征集合不同，或模型输出与历史 SHAP 不一致'
        ),
        'additivity_passed': '可用记录的原尺度加和误差均不超过 1e-4',
        'inference_engine': (
            'rrl_parser 为部署规则解析器；rrl_neural 为神经网络；tree 为树模型'
        ),
        'prediction_kind': (
            'raw 为未校准单模态；calibrated_fusion 包括校准单模态和多模态融合'
        ),
        'fusion_fit_engine': '拟合校准、权重和阈值的实现；RRL 始终为 rrl_neural',
        'fusion_artifacts': '冻结参数的原路径、SHA-256 和导出副本路径',
        'explanation_engine': '解释所对应的实际模型实现',
        'reference_engine': 'SHAP 加和检查参考输出的实现；必须与解释一致',
        'prediction_implementation': '与 inference_engine 一致',
        'source_log': '与该预测数组配套的历史日志',
        'probability': '阳性类预测概率；融合配置为校准融合后的概率',
        'y_true': '训练运行采用的原始标签：1 为 A+、T+ 或快进展，见 task',
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
        'data_scope': 'held-out 为外层测试折；training 为最终训练人群，分别汇总',
        'relative_importance': '阳性类每折 mean-|SHAP| 除以该折全部特征之和',
        'relative_mean': '折内归一化份额的等权均值，随后才选展示特征',
        'relative_sd': '折间总体标准差（ddof=0），不是均值的置信区间',
        'n_folds': '参与该汇总的折数；最终训练模型为 1',
        'n_explained': '共同且模态可用的解释人数；汇总表为各折人数之和',
        'same_feature_set': '同组各模型的完整特征集合是否相同',
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
        pd.read_csv(args.labels, dtype={'subject_id': str, 'cohort': str})
        if args.labels
        else None
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
