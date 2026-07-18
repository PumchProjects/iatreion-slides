import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class CollectConfig:
    n_folds: int
    importance_models: tuple[str, ...]


@dataclass(frozen=True)
class ExcelConfig:
    metrics: tuple[str, ...]
    output: Path


@dataclass(frozen=True)
class ReportConfig:
    title: str
    subtitle: str
    author: str
    footer: str


@dataclass(frozen=True)
class Unit:
    remote_root: str
    root: Path
    settings: tuple[str, ...]
    train_name: str
    names: tuple[str, ...]
    groups: tuple[str, ...]
    models: tuple[str, ...]
    refs: tuple[str, ...]

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.train_name.split("_"))


@dataclass(frozen=True)
class ProjectConfig:
    remote_host: str
    collect: CollectConfig
    excel: ExcelConfig
    report: ReportConfig
    setting_map: dict[str, str]
    name_map: dict[str, str]
    modality_map: dict[str, str]
    group_map: dict[str, str]
    model_map: dict[str, str]
    ref_map: dict[str, str]
    units: tuple[Unit, ...]


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value


def _string(data: dict[str, Any], key: str, prefix: str = "") -> str:
    value = data.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strings(data: dict[str, Any], key: str, prefix: str = "") -> tuple[str, ...]:
    value = data.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty array of strings")
    return tuple(value)


def _mapping(data: dict[str, Any], key: str) -> dict[str, str]:
    value = _table(data, key)
    if any(
        not isinstance(name, str) or not isinstance(label, str)
        for name, label in value.items()
    ):
        raise ValueError(f"{key} must map strings to strings")
    return dict(value)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: str | Path = CONFIG_FILE) -> ProjectConfig:
    config_file = Path(path).expanduser().resolve()
    with config_file.open("rb") as file:
        data: dict[str, Any] = tomllib.load(file)

    remote_host = _string(data, "remote_host")

    collect_data = _table(data, "collect")
    n_folds = collect_data.get("n_folds")
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds <= 0:
        raise ValueError("collect.n_folds must be a positive integer")
    collect = CollectConfig(
        n_folds=n_folds,
        importance_models=_strings(collect_data, "importance_models", "collect"),
    )

    excel_data = _table(data, "excel")
    metrics = _strings(excel_data, "metrics", "excel")
    if len(set(metric.upper() for metric in metrics)) != len(metrics):
        raise ValueError("excel.metrics must not contain duplicates")
    output = _path(_string(excel_data, "output", "excel"), config_file.parent)
    if output.suffix.lower() != ".xlsx":
        raise ValueError("excel.output must end with .xlsx")
    excel = ExcelConfig(metrics=metrics, output=output)

    report_data = _table(data, "report")
    report = ReportConfig(
        title=_string(report_data, "title", "report"),
        subtitle=_string(report_data, "subtitle", "report"),
        author=_string(report_data, "author", "report"),
        footer=_string(report_data, "footer", "report"),
    )

    units_data = data.get("units")
    if not isinstance(units_data, list) or not units_data:
        raise ValueError("units must contain at least one unit")

    units: list[Unit] = []
    roots: set[Path] = set()
    for index, unit_data in enumerate(units_data):
        prefix = f"units[{index}]"
        if not isinstance(unit_data, dict):
            raise ValueError(f"{prefix} must be a TOML table")
        root_value = _string(unit_data, "root", prefix)
        if Path(root_value).expanduser().is_absolute():
            raise ValueError(f"{prefix}.root must be relative to the project root")
        root = _path(root_value, config_file.parent)
        if root in roots:
            raise ValueError(f"duplicate local unit root: {root}")
        roots.add(root)
        units.append(
            Unit(
                remote_root=_string(unit_data, "remote_root", prefix).rstrip("/"),
                root=root,
                settings=_strings(unit_data, "settings", prefix),
                train_name=_string(unit_data, "train_name", prefix),
                names=_strings(unit_data, "names", prefix),
                groups=_strings(unit_data, "groups", prefix),
                models=_strings(unit_data, "models", prefix),
                refs=_strings(unit_data, "refs", prefix),
            )
        )

    return ProjectConfig(
        remote_host=remote_host,
        collect=collect,
        excel=excel,
        report=report,
        setting_map=_mapping(data, "setting_map"),
        name_map=_mapping(data, "name_map"),
        modality_map=_mapping(data, "modality_map"),
        group_map=_mapping(data, "group_map"),
        model_map=_mapping(data, "model_map"),
        ref_map=_mapping(data, "ref_map"),
        units=tuple(units),
    )


CONFIG: ProjectConfig = load_config()


def _extract_roots(config: ProjectConfig) -> None:
    for unit in config.units:
        fields = (config.remote_host, unit.remote_root, str(unit.root))
        if any(any(char in field for char in "\t\r\n") for field in fields):
            raise ValueError(
                "extract paths and remote_host cannot contain tabs or newlines"
            )
        print("\t".join(fields))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the project configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "extract-roots",
        help="Print remote host, remote root, and local root for every unit.",
    )
    args = parser.parse_args()

    if args.command == "extract-roots":
        _extract_roots(CONFIG)


if __name__ == "__main__":
    main()
