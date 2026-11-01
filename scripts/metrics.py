"""Binary prediction metrics and paired participant bootstrap."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit

METRICS = [
    'AUROC',
    'PR_AUC',
    'AP',
    'ACC',
    'BAL_ACC',
    'F1_macro',
    'F1_positive',
    'SEN',
    'SPC',
    'PPV',
    'NPV',
    'Brier',
    'LogLoss',
]
LOWER_IS_BETTER = {'Brier', 'LogLoss'}


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b != 0)


class BinaryMetrics:
    """Pre-sort scores once; bootstrap counts preserve tied-score thresholds."""

    def __init__(self, y, probability, prediction):
        self.y = np.asarray(y, dtype=int)
        self.p = np.asarray(probability, dtype=float)
        self.pred = np.asarray(prediction, dtype=int)
        self.order = np.argsort(-self.p, kind='stable')
        self.group_starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order])) + 1]
        self.error = (self.p - self.y) ** 2
        q = np.clip(self.p, np.finfo(float).eps, 1 - np.finfo(float).eps)
        self.loss = -(self.y * np.log(q) + (1 - self.y) * np.log1p(-q))

    def calculate(self, weights=None):
        n = len(self.y)
        w = np.ones((1, n)) if weights is None else np.atleast_2d(weights)
        ys = self.y[self.order]
        sorted_w = w[:, self.order]
        positive = np.add.reduceat(sorted_w * ys, self.group_starts, axis=1)
        negative = np.add.reduceat(sorted_w * (1 - ys), self.group_starts, axis=1)
        tp_curve, fp_curve = positive.cumsum(1), negative.cumsum(1)
        positives, negatives = tp_curve[:, -1], fp_curve[:, -1]
        tpr = divide(tp_curve, positives[:, None])
        fpr = divide(fp_curve, negatives[:, None])
        precision = divide(tp_curve, tp_curve + fp_curve)
        # Empty bootstrap score groups carry the previous curve point.
        precision = np.where(np.isnan(precision), 1.0, precision)
        zero = np.zeros((len(w), 1))
        recall = np.column_stack([zero, tpr])
        auroc = np.trapezoid(
            np.column_stack([zero, tpr]), np.column_stack([zero, fpr]), axis=1
        )
        pr_auc = np.trapezoid(
            np.column_stack([np.ones(len(w)), precision]), recall, axis=1
        )
        ap = np.sum(np.diff(recall, axis=1) * precision, axis=1)
        # No negative observations yields an undefined discrimination estimate.
        auroc[(positives == 0) | (negatives == 0)] = np.nan
        pr_auc[positives == 0] = np.nan
        ap[positives == 0] = np.nan
        tp = w @ ((self.y == 1) & (self.pred == 1))
        tn = w @ ((self.y == 0) & (self.pred == 0))
        fp = w @ ((self.y == 0) & (self.pred == 1))
        fn = w @ ((self.y == 1) & (self.pred == 0))
        total = tp + tn + fp + fn
        sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
        fpos, fneg = divide(2 * tp, 2 * tp + fp + fn), divide(2 * tn, 2 * tn + fp + fn)
        return dict(
            AUROC=auroc,
            PR_AUC=pr_auc,
            AP=ap,
            ACC=divide(tp + tn, total),
            BAL_ACC=(sensitivity + specificity) / 2,
            F1_macro=np.nanmean(np.stack([fpos, fneg]), axis=0),
            F1_positive=fpos,
            SEN=sensitivity,
            SPC=specificity,
            PPV=divide(tp, tp + fp),
            NPV=divide(tn, tn + fn),
            Brier=divide(w @ self.error, total),
            LogLoss=divide(w @ self.loss, total),
            TP=tp,
            TN=tn,
            FP=fp,
            FN=fn,
        )

    def bootstrap(self, n_resamples=2000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(self.y)
        results = {name: [] for name in METRICS}
        for start in range(0, n_resamples, 128):
            weights = rng.multinomial(
                n, np.full(n, 1 / n), size=min(128, n_resamples - start)
            )
            calculated = self.calculate(weights)
            for name in METRICS:
                results[name].append(calculated[name])
        return {name: np.concatenate(values) for name, values in results.items()}


def metric_table(y, probability, prediction, n_resamples=2000, seed=42):
    engine = BinaryMetrics(y, probability, prediction)
    point = engine.calculate()
    sampled = engine.bootstrap(n_resamples, seed)
    rows = []
    for name in METRICS:
        finite = sampled[name][np.isfinite(sampled[name])]
        lo, hi = (
            np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        )
        rows.append(
            dict(
                metric=name,
                value=point[name][0],
                ci_low=lo,
                ci_high=hi,
                bootstrap_valid=len(finite),
                bootstrap_requested=n_resamples,
                better='lower' if name in LOWER_IS_BETTER else 'higher',
            )
        )
    counts = {k: int(point[k][0]) for k in ['TP', 'TN', 'FP', 'FN']}
    return pd.DataFrame(rows), counts


def calibration(y, probability):
    y = np.asarray(y, dtype=float)
    p = np.asarray(probability, dtype=float)
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def objective(beta, slope_fixed=False):
        z = beta[0] + (1 if slope_fixed else beta[1]) * x
        return np.mean(np.logaddexp(0, z) - y * z)

    joint = minimize(objective, [0.0, 1.0], method='BFGS')
    intercept = minimize(lambda b: objective(b, True), [0.0], method='BFGS')
    result = dict(
        calibration_intercept=float(intercept.x[0]),
        calibration_slope=float(joint.x[1]),
        joint_intercept=float(joint.x[0]),
        calibration_fit_ok=bool(joint.success and intercept.success),
    )
    bins = pd.qcut(p, q=10, duplicates='drop')
    d = pd.DataFrame({'bin': bins, 'probability': p, 'y': y})
    if len(bins.categories) == 0:
        d['bin'] = 'all'
    curve = (
        d.groupby('bin', observed=True)
        .agg(
            n=('y', 'size'),
            events=('y', 'sum'),
            predicted=('probability', 'mean'),
            observed=('y', 'mean'),
        )
        .reset_index()
    )
    curve['bin'] = curve['bin'].astype(str)
    return result, curve


def curves(y, probability):
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, rt = roc_curve(y, probability, drop_intermediate=False)
    precision, recall, pt = precision_recall_curve(
        y, probability, drop_intermediate=False
    )
    roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=rt))
    pr = pd.DataFrame(
        dict(precision=precision, recall=recall, threshold=np.r_[pt, np.nan])
    )
    return roc, pr


def paired_difference(y, pa, preda, pb, predb, n_resamples=2000, seed=42):
    a, b = BinaryMetrics(y, pa, preda), BinaryMetrics(y, pb, predb)
    point_a, point_b = a.calculate(), b.calculate()
    sa, sb = a.bootstrap(n_resamples, seed), b.bootstrap(n_resamples, seed)
    rows = []
    for name in ['AUROC', 'PR_AUC', 'AP', 'Brier', 'F1_macro', 'SEN', 'SPC']:
        delta = sa[name] - sb[name]
        finite = delta[np.isfinite(delta)]
        lo, hi = (
            np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        )
        rows.append(
            dict(
                metric=name,
                difference=point_a[name][0] - point_b[name][0],
                ci_low=lo,
                ci_high=hi,
                n_common=len(y),
            )
        )
    return pd.DataFrame(rows)
