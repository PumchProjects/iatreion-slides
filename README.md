# iatreion-slides

Evaluate frozen Iatreion predictions and explanations, and export publication tables.
Iatreion fits the models, Platt calibrators, fusion weights, and decision thresholds.
This repository computes metrics, confidence intervals, calibration diagnostics,
paired comparisons, and rule and SHAP summaries from the saved outputs.

## Setup

Install Python 3.12 or later and [uv](https://docs.astral.sh/uv/), then run these
commands from the repository root:

```bash
uv sync --locked
cp config.example.toml config.toml
```

Edit `config.toml` with your SSH host, experiment paths, and selected run IDs.
The `.python-version` file selects Python 3.12. `pyproject.toml` declares the
runtime dependencies, and `uv.lock` records their resolved versions. uv uses the
local `.venv`; the project runs directly from `scripts/` without a package build.

Ruff is managed separately on each machine. Its configuration is in
`pyproject.toml`; it is not a project dependency.

## Configuration and collection

| Setting | Purpose |
| --- | --- |
| `remote_host` | SSH host or alias used by SSH and rsync. |
| `[exports].remote_root` | Unified prediction and explanation export directory on the server. |
| `[exports].local_root` | Local copy of the unified exports. |
| `[evaluation].runs` | Run IDs to include in the evaluation. |
| `[evaluation].bootstrap` | Number of participant bootstrap samples; default 2,000. |
| `[evaluation].seed` | Random seed; default 42. |
| `[evaluation].output` | Parent directory for the D5 and D6 tables. |
| `[[runs]]` | Run ID and remote/local directories for collecting experiment metadata. |

Download uses SSH, rsync, and Python 3 on the server. The unified export directory
is synchronized without deleting local files. Metadata collection retrieves
manifests, preprocessing information, selected hyperparameters, and fusion
artifacts from the configured runs. Model checkpoints and ROC images are not
part of metadata collection.

```bash
# Inspect the planned synchronization.
scripts/extract.sh --dry-run

# Download exports and experiment metadata.
scripts/extract.sh

# Generate D5 and D6 from the selected frozen runs.
uv run --locked python -m scripts.collect

# Create a readable metrics workbook.
uv run --locked python -m scripts.to_excel \
  --input outputs/paper_tables/D5 \
  --output outputs/prediction_metrics.xlsx
```

The shell wrapper changes to the repository root and runs
`uv run --locked python -m scripts.download`. Each command accepts `--help`.

To evaluate updated MMSE labels alongside the model's original labels, supply a
CSV with `cohort`, `subject_id`, `definition`, `label`, and `label_version`:

```bash
uv run --locked python -m scripts.collect \
  --labels /path/to/D2/mmse_labels.csv
```

The collector uses rows with `definition = main`, matched by cohort and subject
ID. The current study's main MMSE definition is baseline-onward OLS slope
≤ −3 points/year (`baseline_ols_le3_v1`). Original model labels remain available
for comparison. Study-specific cohort tables, label construction, and feature
mapping belong to the paper's D1–D7 workflow, which can call this collector with
`--output`, `--labels`, and `--tables`.

## Exporting a new Iatreion run

Run the following command in the **Iatreion training repository** after training,
result replay, and external evaluation have produced their artifacts:

```bash
uv run --no-sync python scripts/export_predictions.py \
  --root /path/to/experiment \
  --run-id example_run \
  --train-cohort 4500 \
  --feature-scheme shared \
  --label-version baseline_ols_le3_v1 \
  --output /path/to/exports
```

Use `--feature-scheme clinical` for the full clinical modality scheme. Set the
label version to the definition used in that run. The exporter packages current
schema-v2 internal out-of-fold (OOF) predictions, external predictions, rules,
and SHAP artifacts. The collector consumes this format directly.

Point `[exports].remote_root` at the export directory and add the run ID to
`[evaluation].runs`. No retraining is needed to change offline metrics.

## Prediction format

Each `predictions.npz` has a companion `predictions.json` file.

| NPZ field | Meaning |
| --- | --- |
| `sample_id` | Participant identifier, preserved as a string. |
| `y_true` | Original class index; negative values indicate a missing label. |
| `y_score` | Class probabilities, ordered by the JSON `labels` array. |
| `y_pred` | Predicted class from the saved decision rule. |
| `y_mask` | `true` when a prediction is unavailable. |
| `outer_fold`, `inner_fold` | Saved cross-validation fold assignments. |
| `kind` | Full record type, such as `outer_test` or `external`. |
| `threshold` | Actual development-side decision threshold for each participant. |
| `module_names` | Ordered modalities in a fusion configuration. |
| `module_raw` | Raw positive-class probabilities by modality. |
| `module_calibrated` | Development-side calibrated probabilities by modality. |
| `module_mask` | Unavailable-modality mask aligned with `module_names`. |

Fusion fields accompany fused predictions. JSON metadata records class meaning,
positive-class index, task, run, cohort, feature scheme, label and visit
definitions, threshold source, and source file provenance.

D5 contains one row per configuration and participant. Internal OOF predictions
contain one outer-test record per participant and configuration. Unavailable
probabilities and predictions remain empty in the exported table.

Frozen-threshold evaluation uses the saved `y_pred`. A separate evaluation uses
`p >= 0.5`. Single-modality predictions recorded with `argmax` retain that
decision rule; a binary tie at 0.5 selects class index 0.

TSH inclusion and MRI representation are alternative replay branches:

- `h-nonimaging` / `h-nonimaging-no-tsh`, or
  `sh-apoe-labdata` / `sh-apoe-labdata-no-tsh`;
- `h-mri` / `h-mri-roi`.

Each replay configuration selects one TSH branch and one MRI branch, yielding
four combinations while reusing the other trained modalities.

## Metrics and calibration

The metric table reports AUROC, trapezoidal precision–recall AUC (`PR_AUC`),
average precision (`AP`), accuracy, balanced accuracy, macro F1, positive-class
F1, sensitivity, specificity, positive predictive value (PPV), negative
predictive value (NPV), Brier score, and log loss. Confusion tables include
TP/TN/FP/FN, event counts, eligible and evaluated sample sizes, and coverage.

`PR_AUC` integrates the complete precision–recall curve using trapezoids. `AP`
weights precision by each increase in recall. The historical training log field
named `AUPRC` was computed as AP. Undefined predictive values, such as PPV when
there are no positive predictions, are left empty.

The default 95% confidence intervals use 2,000 unstratified participant bootstrap
samples with seed 42. Non-finite estimates are excluded per metric, and the
number of valid samples is reported. Predictions, calibrators, and thresholds
remain frozen throughout this calculation.

Calibration outputs include an intercept fitted with slope fixed at 1, a jointly
fitted intercept and slope, optimization status, and a quantile-binned calibration
curve. Probabilities are clipped to `[1e-6, 1 - 1e-6]` for the logit fit.

Within a run, model comparisons use shared participants and paired bootstrap
samples. The metrics workbook highlights the best value within each comparable
metric and configuration; Brier score and log loss use the lower-is-better
direction.

For a comparison between two runs, collect their D5 tables separately and use:

```bash
uv run --locked python -m scripts.compare \
  --left outputs/run_a/D5/predictions.csv \
  --right outputs/run_b/D5/predictions.csv \
  --left-run run_a --right-run run_b --cohort AIBL \
  --output outputs/comparisons/run_a_vs_run_b.csv
```

This comparison matches external participants within each missing-data setting,
model, and MRI representation. Each input run should provide one feature scheme
per such combination. Differences are reported as the left run minus the right
run, using common participants with matching labels.

## Rule and SHAP outputs

D6 contains rule conditions, class weights, complexity, valid and active counts,
coverage rates, outcome counts, odds ratios, and associated uncertainty. It
preserves both the raw rule weights and the weights transformed by calibration
and fusion. The saved `Weight Space` identifies the scale; fusion weights are
reported before participant-specific normalization over available modalities.
`Outcome Label` identifies the class used for each rule's effect statistics.

SHAP bundles preserve participant IDs, transformed feature values, feature and
output names, base values, contributions, explanation population, output scale,
and same-scale `reference_output`. D6 exports global summaries, a compressed
individual contribution table, reconstructed model outputs, and participant-level
additivity checks. Feature units and transformations are supplied by the paper's
feature dictionary. Rule statistics retain their recorded label version.

## Outputs and repository contents

- `D5/`: individual predictions, metrics, confusion tables, calibration diagnostics,
  ROC/PR curve points, and paired model comparisons.
- `D6/`: rule summaries, SHAP summaries, `shap_values.csv.gz`, and additivity checks.
- Each table directory includes `dictionary.csv` and `tables.json`.
- `scripts/` and `tests/`: maintained collection and evaluation code.
- `config.example.toml`, `pyproject.toml`, `uv.lock`, and `.python-version`: shared
  configuration and environment definition.
- `config.toml`, `data/`, `outputs/`, `.venv/`, and caches: local working files
  excluded from version control.

## Development

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests
ruff check scripts tests
ruff format --check scripts tests
```

To format edited files, run `ruff format scripts tests`. Ruff uses the version
installed on the current machine, independently of `uv.lock`.
