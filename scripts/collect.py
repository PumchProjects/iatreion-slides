import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from config import CONFIG, ProjectConfig, Unit


def get_scores(ref_root: Path, modality: str, n_folds: int, top_k: int = 10) -> str:
    result: defaultdict[str, float] = defaultdict(float)
    for fold in range(n_folds):
        score_file = ref_root / f"score_native_{modality}_{fold}_{n_folds}.json"
        with score_file.open("r", encoding="utf-8") as file:
            scores: dict[str, float] = json.load(file)
        for name, score in scores.items():
            result[name] += score

    rows = sorted(result.items(), key=lambda item: item[1], reverse=True)
    precision = 0 if all(score.is_integer() for _, score in rows) else 4
    return "\n".join(f"{name}\t{score:.{precision}f}" for name, score in rows[:top_k])


def write_scores(exp_root: Path, modalities: tuple[str, ...], n_folds: int) -> None:
    for modality in modalities:
        scores = get_scores(exp_root, modality, n_folds)
        (exp_root / f"scores_{modality}.tsv").write_text(scores, encoding="utf-8")

    weights = get_weights(exp_root / "available_fusion.toml")
    (exp_root / "weights.tsv").write_text(weights, encoding="utf-8")


def get_results(log_file: Path) -> str:
    data = log_file.read_text(encoding="utf-8")
    try:
        anchor = data.rindex("Confusion matrix")
    except ValueError as error:
        raise ValueError(f"{log_file} does not contain a confusion matrix") from error
    lines = data[anchor:].split("\n")
    return "\n".join([lines[0], *(line[33:] for line in lines[1:])])


def get_weights(fusion_file: Path) -> str:
    with fusion_file.open("rb") as file:
        data = tomllib.load(file)
    weights: dict[str, float] = data["weights"]
    return "\n".join(f"{name}\t{score:.4f}" for name, score in weights.items())


def get_rules(rule_file: Path) -> str:
    lines = rule_file.read_text(encoding="utf-8").strip().split("\n")
    for index in range(1, len(lines)):
        prefix, score, suffix = lines[index].rsplit("\t", maxsplit=2)
        lines[index] = f"{prefix}\t{float(score):.2f}\t{suffix}"
    return "\n".join(lines)


def write_results(exp_root: Path, names: tuple[str, ...], collect_rules: bool) -> None:
    for name in names:
        name_root = exp_root / "eval" / name
        modalities = "_".join(sorted(name.split("_")))
        results = get_results(name_root / "eval.log")
        (name_root / "results.log").write_text(results, encoding="utf-8")

        weights = get_weights(
            exp_root / "fusion" / modalities / "available_fusion.toml"
        )
        (name_root / "weights.tsv").write_text(weights, encoding="utf-8")

        if collect_rules:
            rules = get_rules(name_root / "rrl_ranked_rules.tsv")
            (name_root / "rules.tsv").write_text(rules, encoding="utf-8")


def collect_unit(config: ProjectConfig, unit: Unit) -> None:
    for setting in unit.settings:
        setting_root = unit.root / setting
        if "internal" in unit.validations:
            training_root = setting_root / "training" / unit.train_name
            print(f"{' Internal ':=^80}")
            for group in unit.groups:
                print(f"Processing {setting} / {group}...")
                group_root = training_root / group
                for model in unit.models:
                    if model.name not in config.collect.importance_models:
                        continue
                    for ref in unit.refs:
                        write_scores(
                            group_root / model.id / ref,
                            unit.modalities,
                            config.collect.n_folds,
                        )

        if "external" in unit.validations:
            print(f"{' External ':=^80}")
            for group in unit.groups:
                print(f"Processing {setting} / {group}...")
                group_root = setting_root / "final" / group
                for model in unit.models:
                    write_results(
                        group_root / model.id,
                        unit.names,
                        collect_rules=model.name == "rrl",
                    )


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit(f"Usage: {sys.argv[0]}")

    for index, unit in enumerate(CONFIG.units, start=1):
        print(f"{f' Unit {index} ':#^80}")
        collect_unit(CONFIG, unit)


if __name__ == "__main__":
    main()
