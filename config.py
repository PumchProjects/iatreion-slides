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
class Download:
    id: str
    remote_root: str
    local_root: Path


@dataclass(frozen=True)
class Model:
    name: str
    download: str
    suffix: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}{self.suffix}"

    @property
    def parser_id(self) -> str:
        return f"{self.name}-parser{self.suffix}"


@dataclass(frozen=True)
class Unit:
    root: Path
    settings: tuple[str, ...]
    train_name: str
    names: tuple[str, ...]
    groups: tuple[str, ...]
    models: tuple[Model, ...]
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
    downloads: tuple[Download, ...]
    units: tuple[Unit, ...]

    @property
    def downloads_by_id(self) -> dict[str, Download]:
        return {download.id: download for download in self.downloads}


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


def _optional_string(
    data: dict[str, Any], key: str, prefix: str = "", default: str = ""
) -> str:
    value = data.get(key, default)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _path_component(value: str, label: str) -> str:
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be a single path component")
    return value


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _relative_path(data: dict[str, Any], key: str, prefix: str, base: Path) -> Path:
    value = _string(data, key, prefix)
    if Path(value).expanduser().is_absolute():
        raise ValueError(f"{prefix}.{key} must be relative to the project root")
    return _path(value, base)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


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

    downloads_data = data.get("downloads")
    if not isinstance(downloads_data, list) or not downloads_data:
        raise ValueError("downloads must contain at least one download")

    downloads: list[Download] = []
    download_ids: set[str] = set()
    download_roots: set[Path] = set()
    for index, download_data in enumerate(downloads_data):
        prefix = f"downloads[{index}]"
        if not isinstance(download_data, dict):
            raise ValueError(f"{prefix} must be a TOML table")
        download_id = _path_component(
            _string(download_data, "id", prefix), f"{prefix}.id"
        )
        if download_id in download_ids:
            raise ValueError(f"duplicate download id: {download_id}")
        local_root = _relative_path(
            download_data, "local_root", prefix, config_file.parent
        )
        if local_root in download_roots:
            raise ValueError(f"duplicate download local_root: {local_root}")
        download_ids.add(download_id)
        download_roots.add(local_root)
        downloads.append(
            Download(
                id=download_id,
                remote_root=_string(download_data, "remote_root", prefix).rstrip("/"),
                local_root=local_root,
            )
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
        root = _relative_path(unit_data, "root", prefix, config_file.parent)
        for other_root in roots:
            if _paths_overlap(root, other_root):
                raise ValueError(f"overlapping local unit roots: {root}, {other_root}")
        for download_root in download_roots:
            if _paths_overlap(root, download_root):
                raise ValueError(
                    f"{prefix}.root overlaps download local_root: {download_root}"
                )
        models_data = unit_data.get("models")
        if not isinstance(models_data, list) or not models_data:
            raise ValueError(f"{prefix}.models must be a non-empty array of tables")
        models: list[Model] = []
        model_ids: set[str] = set()
        for model_index, model_data in enumerate(models_data):
            model_prefix = f"{prefix}.models[{model_index}]"
            if not isinstance(model_data, dict):
                raise ValueError(f"{model_prefix} must be a TOML table")
            name = _path_component(
                _string(model_data, "name", model_prefix), f"{model_prefix}.name"
            )
            suffix = _optional_string(model_data, "suffix", model_prefix)
            model_id = _path_component(f"{name}{suffix}", f"{model_prefix} id")
            if model_id in model_ids:
                raise ValueError(f"duplicate model id in {prefix}: {model_id}")
            download_id = _string(model_data, "download", model_prefix)
            if download_id not in download_ids:
                raise ValueError(
                    f"{model_prefix}.download references unknown download: "
                    f"{download_id}"
                )
            model_ids.add(model_id)
            models.append(Model(name=name, download=download_id, suffix=suffix))
        roots.add(root)
        units.append(
            Unit(
                root=root,
                settings=_strings(unit_data, "settings", prefix),
                train_name=_string(unit_data, "train_name", prefix),
                names=_strings(unit_data, "names", prefix),
                groups=_strings(unit_data, "groups", prefix),
                models=tuple(models),
                refs=_strings(unit_data, "refs", prefix),
            )
        )

    model_map = _mapping(data, "model_map")
    missing_model_labels = sorted(
        model.id for unit in units for model in unit.models if model.id not in model_map
    )
    if missing_model_labels:
        raise ValueError(
            "model_map is missing labels for: " + ", ".join(missing_model_labels)
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
        model_map=model_map,
        ref_map=_mapping(data, "ref_map"),
        downloads=tuple(downloads),
        units=tuple(units),
    )


CONFIG: ProjectConfig = load_config()


def _extract_roots(config: ProjectConfig) -> None:
    for download in config.downloads:
        fields = (config.remote_host, download.remote_root, str(download.local_root))
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
        help="Print remote host, remote root, and local root for every download.",
    )
    args = parser.parse_args()

    if args.command == "extract-roots":
        _extract_roots(CONFIG)


if __name__ == "__main__":
    main()
