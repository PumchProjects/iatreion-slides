"""Readable metric workbook from the long-form prediction evaluation tables."""

import argparse
from pathlib import Path

import pandas as pd
import xlsxwriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('outputs/paper_tables/D5'))
    p.add_argument(
        '--output', type=Path, default=Path('outputs/prediction_metrics.xlsx')
    )
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with xlsxwriter.Workbook(args.output, {'nan_inf_to_errors': False}) as book:
        header = book.add_format(
            {
                'bold': True,
                'bg_color': '#143A4A',
                'font_color': 'white',
                'text_wrap': True,
            }
        )
        number = book.add_format({'num_format': '0.0000'})
        for name in ['metrics', 'confusion', 'calibration', 'paired_comparisons']:
            d = pd.read_csv(args.input / (name + '.csv'))
            sheet = book.add_worksheet(name)
            sheet.freeze_panes(1, 0)
            sheet.hide_gridlines(2)
            sheet.set_row(0, 38)
            for j, c in enumerate(d):
                sheet.write(0, j, c, header)
                sheet.set_column(j, j, 24 if d[c].dtype == 'object' else 17)
            for i, row in enumerate(d.itertuples(index=False, name=None), 1):
                for j, value in enumerate(row):
                    if pd.isna(value):
                        continue
                    sheet.write(
                        i, j, value, number if isinstance(value, float) else None
                    )
            sheet.autofilter(0, 0, len(d), len(d.columns) - 1)
            if name == 'metrics':
                j = d.columns.get_loc('value')
                for direction, group in d.groupby('better'):
                    # Each comparison contains the same metric and target population.
                    keys = [
                        'run_id',
                        'setting',
                        'stage',
                        'cohort',
                        'subset',
                        'task',
                        'evaluation_label_version',
                        'threshold_mode',
                        'metric',
                    ]
                    for _, g in group.groupby([k for k in keys if k in d]):
                        best = g.value.min() if direction == 'lower' else g.value.max()
                        best_fmt = book.add_format(
                            {
                                'bold': True,
                                'bg_color': '#CCE9DF',
                                'num_format': '0.0000',
                            }
                        )
                        for ix in g.index[g.value.eq(best)]:
                            sheet.write_number(int(ix) + 1, j, float(best), best_fmt)
    print(args.output)


if __name__ == '__main__':
    main()
