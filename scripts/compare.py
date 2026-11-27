"""Compare two explicit frozen configurations on common external participants."""

import argparse
from pathlib import Path

import pandas as pd

from scripts.metrics import paired_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--left-run', required=True)
    p.add_argument('--right-run', required=True)
    p.add_argument('--left-subset', required=True)
    p.add_argument('--right-subset', required=True)
    p.add_argument('--task', choices=['A', 'T', 'MMSE'], required=True)
    p.add_argument(
        '--label-column', choices=['y_true', 'y_true_baseline'], default='y_true'
    )
    p.add_argument('--cohort', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    tables = []
    for file, run, subset in [
        (args.left, args.left_run, args.left_subset),
        (args.right, args.right_run, args.right_subset),
    ]:
        d = pd.read_csv(file, dtype={'subject_id': str, 'cohort': str})
        d = d[
            (d.run_id == run)
            & (d.subset == subset)
            & (d.task == args.task)
            & (d.cohort == args.cohort)
            & (d.stage == 'external')
            & ~d.unavailable
            & (d.prediction_kind == 'calibrated_fusion')
            & ((d.model != 'rrl') | (d.inference_engine == 'rrl_parser'))
        ].copy()
        tables.append(d)
    rows = []
    keys = ['setting', 'model', 'inference_engine']
    for key, a in tables[0].groupby(keys):
        b = tables[1]
        for k, v in zip(keys, key, strict=True):
            b = b[b[k] == v]
        a, b = a.set_index('subject_id'), b.set_index('subject_id')
        assert a.index.is_unique and b.index.is_unique
        common = a.index.intersection(b.index).sort_values()
        label = args.label_column
        ids = common[a.loc[common, label].notna() & b.loc[common, label].notna()]
        if not len(ids):
            continue
        x, z = a.loc[ids], b.loc[ids]
        assert (x[label] == z[label]).all()
        t = paired_difference(
            x[label].astype(int),
            x.probability,
            x.y_pred.astype(int),
            z.probability,
            z.y_pred.astype(int),
            args.bootstrap,
            42,
        )
        for k, v in {
            **dict(zip(keys, key, strict=True)),
            'left_run': args.left_run,
            'right_run': args.right_run,
            'left_subset': args.left_subset,
            'right_subset': args.right_subset,
            'task': args.task,
            'cohort': args.cohort,
            'label_column': label,
            'n_common': len(ids),
        }.items():
            t[k] = v
        rows.append(t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
