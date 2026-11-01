"""Compare two runs on matched external participants and MRI/settings/model."""

import argparse
from pathlib import Path

import pandas as pd

from scripts.metrics import paired_difference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--left-run', required=True)
    p.add_argument('--right-run', required=True)
    p.add_argument('--cohort', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    tables = []
    for file, run in [(args.left, args.left_run), (args.right, args.right_run)]:
        d = pd.read_csv(file, dtype={'subject_id': str})
        d = d[
            (d.run_id == run)
            & (d.cohort == args.cohort)
            & (d.stage == 'external')
            & ~d.unavailable
        ].copy()
        d['mri'] = d.subset.str.contains('h-mri-roi').map({True: 'ROI', False: 'fine'})
        tables.append(d)
    rows = []
    keys = ['setting', 'model', 'mri']
    for key, a in tables[0].groupby(keys):
        b = tables[1]
        for k, v in zip(keys, key, strict=True):
            b = b[b[k] == v]
        a = a.set_index('subject_id')
        b = b.set_index('subject_id')
        common = a.index.intersection(b.index).sort_values()
        for label in ['y_true', 'y_true_baseline']:
            if label not in a or label not in b:
                continue
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
                'cohort': args.cohort,
                'label_column': label,
            }.items():
                t[k] = v
            rows.append(t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
