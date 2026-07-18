import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import xlsxwriter  # type: ignore[import]

from config import CONFIG, ProjectConfig, Unit

METRIC_ALIASES = {
    "AUC": "AUROC",
}
METRIC_PATTERN = re.compile(
    r"^\s*(?P<metric>[A-Za-z0-9_+-]+)\s+(?P<value>-?\d+(?:\.\d+)?)%",
    re.MULTILINE,
)


@dataclass
class TableRow:
    setting: str
    stage: str
    group: str
    item: str
    model: str
    values: dict[str, float]

    @property
    def best_key(self) -> tuple[str, str, str, str]:
        return (self.setting, self.stage, self.group, self.item)


def normalize_metric(metric: str) -> str:
    metric = metric.strip().upper()
    return METRIC_ALIASES.get(metric, metric)


def normalize_metrics(metrics: Iterable[str]) -> list[str]:
    result: list[str] = []
    for raw_metric in metrics:
        for metric in re.split(r"[,\s]+", raw_metric):
            if metric:
                result.append(normalize_metric(metric))
    return list(dict.fromkeys(result))


def read_metrics(log_path: Path, metrics: Sequence[str]) -> dict[str, float]:
    data = log_path.read_text(encoding="utf-8")
    found = {
        normalize_metric(match.group("metric")): float(match.group("value"))
        for match in METRIC_PATTERN.finditer(data)
    }
    missing = [metric for metric in metrics if metric not in found]
    if missing:
        raise ValueError(f"{log_path} is missing metrics: {', '.join(missing)}")
    return {metric: found[metric] for metric in metrics}


def mapped(mapping: dict[str, str], key: str) -> str:
    return mapping.get(key, key)


def group_label(config: ProjectConfig, group: str) -> str:
    return " : ".join(mapped(config.group_map, item) for item in group.split("_"))


def model_label(
    config: ProjectConfig, model: str, ref: str | None, refs: Sequence[str]
) -> str:
    label = mapped(config.model_map, model)
    if ref is not None and len(refs) > 1:
        label += f"（{mapped(config.ref_map, ref)}）"
    return label


def training_fusion_log(
    unit: Unit, setting: str, group: str, model: str, ref: str, name: str
) -> Path:
    ref_name = ref.replace("-", "_")
    return (
        unit.root
        / setting
        / "training"
        / unit.train_name
        / group
        / "result-replay"
        / model
        / ref
        / name
        / f"train_ci_all_{ref_name}_youden.log"
    )


def final_fusion_log(
    unit: Unit, setting: str, group: str, model: str, name: str
) -> Path:
    return unit.root / setting / "final" / group / model / "eval" / name / "results.log"


def modality_log(
    unit: Unit, setting: str, group: str, model: str, ref: str, modality: str
) -> Path:
    log_model = "rrl-parser" if model == "rrl" else model
    return (
        unit.root
        / setting
        / "training"
        / unit.train_name
        / group
        / log_model
        / ref
        / f"train_ci_{modality}.log"
    )


def collect_fusion_rows(
    config: ProjectConfig, unit: Unit, metrics: Sequence[str]
) -> list[TableRow]:
    rows: list[TableRow] = []
    for setting in unit.settings:
        setting_name = mapped(config.setting_map, setting)
        for name in unit.names:
            item_name = mapped(config.name_map, name)
            for group in unit.groups:
                group_name = group_label(config, group)
                for model in unit.models:
                    for ref in unit.refs:
                        rows.append(
                            TableRow(
                                setting=setting_name,
                                stage="交叉验证",
                                group=group_name,
                                item=item_name,
                                model=model_label(config, model, ref, unit.refs),
                                values=read_metrics(
                                    training_fusion_log(
                                        unit, setting, group, model, ref, name
                                    ),
                                    metrics,
                                ),
                            )
                        )
                for model in unit.models:
                    rows.append(
                        TableRow(
                            setting=setting_name,
                            stage="外部验证",
                            group=group_name,
                            item=item_name,
                            model=model_label(config, model, None, unit.refs),
                            values=read_metrics(
                                final_fusion_log(unit, setting, group, model, name),
                                metrics,
                            ),
                        )
                    )
    return rows


def collect_modality_rows(
    config: ProjectConfig, unit: Unit, metrics: Sequence[str]
) -> list[TableRow]:
    rows: list[TableRow] = []
    for setting in unit.settings:
        setting_name = mapped(config.setting_map, setting)
        for group in unit.groups:
            group_name = group_label(config, group)
            for modality in unit.modalities:
                item_name = mapped(config.modality_map, modality)
                for model in unit.models:
                    for ref in unit.refs:
                        rows.append(
                            TableRow(
                                setting=setting_name,
                                stage="交叉验证",
                                group=group_name,
                                item=item_name,
                                model=model_label(config, model, ref, unit.refs),
                                values=read_metrics(
                                    modality_log(
                                        unit, setting, group, model, ref, modality
                                    ),
                                    metrics,
                                ),
                            )
                        )
    return rows


def best_values(
    rows: Sequence[TableRow], metrics: Sequence[str]
) -> dict[tuple[tuple[str, str, str, str], str], float]:
    result: dict[tuple[tuple[str, str, str, str], str], float] = {}
    for row in rows:
        for metric in metrics:
            key = (row.best_key, metric)
            value = row.values[metric]
            if key not in result or value > result[key]:
                result[key] = value
    return result


def write_sheet(
    workbook: xlsxwriter.Workbook,
    sheet_name: str,
    rows: Sequence[TableRow],
    metrics: Sequence[str],
) -> None:
    worksheet = workbook.add_worksheet(sheet_name)
    headers = ["设置", "阶段", "分组", "项目", "模型", *metrics]

    header_format = workbook.add_format(
        {"bold": True, "font_size": 12, "align": "center", "valign": "vcenter"}
    )
    text_format = workbook.add_format({"font_size": 11, "valign": "vcenter"})
    number_format = workbook.add_format(
        {"font_size": 11, "num_format": "0.00", "valign": "vcenter"}
    )
    best_number_format = workbook.add_format(
        {"bold": True, "font_size": 11, "num_format": "0.00", "valign": "vcenter"}
    )

    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)

    bests = best_values(rows, metrics)
    for row_idx, row in enumerate(rows, start=1):
        for col, value_str in enumerate(
            [row.setting, row.stage, row.group, row.item, row.model]
        ):
            worksheet.write(row_idx, col, value_str, text_format)
        for metric_idx, metric in enumerate(metrics, start=5):
            value = row.values[metric]
            best = bests[(row.best_key, metric)]
            cell_format = (
                best_number_format if abs(value - best) < 1e-9 else number_format
            )
            worksheet.write_number(row_idx, metric_idx, value, cell_format)

    worksheet.freeze_panes(1, 0)
    if rows:
        worksheet.autofilter(0, 0, len(rows), len(headers) - 1)
    worksheet.set_column(0, 0, 34, text_format)
    worksheet.set_column(1, 1, 12, text_format)
    worksheet.set_column(2, 2, 14, text_format)
    worksheet.set_column(3, 3, 18, text_format)
    worksheet.set_column(4, 4, 14, text_format)
    worksheet.set_column(5, len(headers) - 1, 11, number_format)


def build_workbook(
    config: ProjectConfig, metrics: Sequence[str], output: Path
) -> None:
    fusion_rows: list[TableRow] = []
    modality_rows: list[TableRow] = []
    for unit in config.units:
        fusion_rows.extend(collect_fusion_rows(config, unit, metrics))
        modality_rows.extend(collect_modality_rows(config, unit, metrics))

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output))
    try:
        write_sheet(workbook, "校准融合", fusion_rows, metrics)
        write_sheet(workbook, "各模态", modality_rows, metrics)
    finally:
        workbook.close()


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit(f"Usage: {sys.argv[0]}")

    metrics = normalize_metrics(CONFIG.excel.metrics)
    build_workbook(CONFIG, metrics, CONFIG.excel.output)
    print(f"Wrote {CONFIG.excel.output}")


if __name__ == "__main__":
    main()
