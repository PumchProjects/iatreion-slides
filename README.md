# iatreion-slides

Generate slides and spreadsheets from configured experiment logs.

## Requirements

- Python 3.12+ with `xlsxwriter`
- SSH and `rsync` access to the configured remote host; `rsync` 3.x is recommended for aggregate progress
- [Tinymist](https://github.com/Myriad-Dreamin/tinymist) for compiling the Typst report

## Configuration

Create the local configuration before running the pipeline:

```bash
cp config.example.toml config.toml
```

Edit `config.toml` to define the remote host, experiment units, collection settings, spreadsheet output, labels, and report metadata. Unit roots are project-relative; other relative paths are also resolved from the project root.

## Pipeline

Run the following commands from the project root.

### 1. Download logs

```bash
scripts/extract.sh
```

Every unit is downloaded to its configured local root. The script excludes large or unnecessary training artifacts. Use `--dry-run` to inspect transfer statistics without writing files.

### 2. Collect derived results

```bash
python -m scripts.collect
```

This aggregates feature importance across folds, extracts final evaluation text, and writes modality weights and ranked rules for every configured unit.

### 3. Build the report

Open `report/main.typ` with Tinymist and export the document. The report reads its metadata and all units directly from `config.toml`; adding units does not require editing the Typst sources.

### 4. Export the spreadsheet

```bash
python -m scripts.to_excel
```

The configured metrics from every unit are written to the single workbook at `excel.output`.
