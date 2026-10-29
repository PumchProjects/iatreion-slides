import json
import os
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from config import CONFIG, Unit, load_config
from scripts.collect import collect_unit, write_results, write_scores
from scripts.merge import MARKER, merge_unit
from scripts.to_excel import (
    TableRow,
    best_values,
    build_workbook,
    collect_fusion_rows,
    collect_modality_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
TEST_CONFIG = replace(
    CONFIG,
    cohort_map={
        "training": "训练队列",
        "cohort-a": "队列 A",
        "cohort-b": "队列 B",
    },
)


class ConfigTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text(text, encoding="utf-8")
            return load_config(config_file)

    def assert_config_error(self, old: str, new: str, pattern: str) -> None:
        self.assertIn(old, EXAMPLE_CONFIG)
        with self.assertRaisesRegex(ValueError, pattern):
            self.load_text(EXAMPLE_CONFIG.replace(old, new))

    def test_validation_stages_and_labels_are_loaded(self) -> None:
        config = self.load_text(EXAMPLE_CONFIG)
        self.assertEqual(config.units[0].validations, ("internal", "external"))
        self.assertEqual(config.validation_map["internal"], "内部验证")

    def test_cohorts_and_labels_are_loaded(self) -> None:
        config = self.load_text(EXAMPLE_CONFIG)
        self.assertEqual(config.units[0].internal_cohort, "training")
        self.assertEqual(
            config.units[0].external_cohorts, ("cohort-a", "cohort-b")
        )
        self.assertEqual(config.cohort_map["cohort-a"], "队列 A")

    def test_missing_internal_cohort_is_rejected(self) -> None:
        self.assert_config_error(
            'internal_cohort = "training"\n',
            "",
            "internal_cohort must be a non-empty string",
        )

    def test_duplicate_external_cohort_is_rejected(self) -> None:
        self.assert_config_error(
            'external_cohorts = [ "cohort-a", "cohort-b" ]',
            'external_cohorts = [ "cohort-a", "cohort-a" ]',
            "external_cohorts must not contain duplicates",
        )

    def test_missing_cohort_label_is_rejected(self) -> None:
        self.assert_config_error(
            'cohort-b = "队列 B"\n',
            "",
            "cohort_map is missing labels for: cohort-b",
        )

    def test_empty_validations_are_rejected(self) -> None:
        self.assert_config_error(
            'validations = [ "internal", "external" ]',
            "validations = []",
            "validations must be a non-empty array",
        )

    def test_duplicate_validations_are_rejected(self) -> None:
        self.assert_config_error(
            'validations = [ "internal", "external" ]',
            'validations = [ "external", "external" ]',
            "validations must not contain duplicates",
        )

    def test_unknown_validation_is_rejected(self) -> None:
        self.assert_config_error(
            'validations = [ "internal", "external" ]',
            'validations = [ "unknown" ]',
            "validations contains unknown stages",
        )

    def test_missing_validation_label_is_rejected(self) -> None:
        self.assert_config_error(
            'external = "外部验证"\n',
            "",
            "validation_map is missing non-empty labels for: external",
        )

    def test_model_ids_include_configured_suffix(self) -> None:
        config = self.load_text(EXAMPLE_CONFIG)
        self.assertEqual(
            [model.id for model in config.units[0].models],
            ["xgboost", "random-forest", "rrl", "rrl-variant"],
        )

    def test_duplicate_download_id_is_rejected(self) -> None:
        self.assert_config_error(
            'id = "run-b"', 'id = "run-a"', "duplicate download id"
        )

    def test_unknown_download_is_rejected(self) -> None:
        self.assert_config_error(
            'download = "run-b"', 'download = "missing"', "unknown download"
        )

    def test_overlapping_download_and_unit_roots_are_rejected(self) -> None:
        self.assert_config_error(
            'root = "data/combined"',
            'root = "data/downloads"',
            "overlaps download local_root",
        )

    def test_duplicate_model_id_is_rejected(self) -> None:
        self.assert_config_error(
            '{ download = "run-b", name = "rrl", suffix = "-variant" }',
            '{ download = "run-b", name = "rrl" }',
            "duplicate model id",
        )


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        config_file = self.root / "config.toml"
        config_file.write_text(EXAMPLE_CONFIG, encoding="utf-8")
        self.config = load_config(config_file)
        self.unit = self.config.units[0]
        self.create_sources()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_sources(self) -> None:
        downloads = self.config.downloads_by_id
        for model in self.unit.models:
            download_root = downloads[model.download].local_root
            for setting in self.unit.settings:
                for group in self.unit.groups:
                    training_root = (
                        download_root
                        / setting
                        / "training"
                        / self.unit.train_name
                        / group
                    )
                    for path in (
                        download_root / setting / "final" / group / model.name,
                        training_root / model.name,
                        training_root / "result-replay" / model.name,
                    ):
                        path.mkdir(parents=True, exist_ok=True)
                    if model.name == "rrl":
                        (training_root / "rrl-parser").mkdir(
                            parents=True, exist_ok=True
                        )

    def test_merge_creates_relative_model_and_parser_links(self) -> None:
        merge_unit(self.config, self.unit)

        training_root = (
            self.unit.root
            / self.unit.settings[0]
            / "training"
            / self.unit.train_name
            / self.unit.groups[0]
        )
        model_link = training_root / "result-replay" / "rrl-variant"
        parser_link = training_root / "rrl-parser-variant"
        self.assertTrue((self.unit.root / MARKER).is_file())
        self.assertTrue(model_link.is_symlink())
        self.assertTrue(parser_link.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(model_link)))
        self.assertEqual(model_link.resolve().name, "rrl")
        self.assertEqual(parser_link.resolve().name, "rrl-parser")

    def test_external_only_does_not_require_or_link_training(self) -> None:
        for download in self.config.downloads:
            for setting in self.unit.settings:
                shutil.rmtree(download.local_root / setting / "training", ignore_errors=True)
        unit = replace(self.unit, validations=("external",))

        merge_unit(self.config, unit)

        self.assertFalse((unit.root / unit.settings[0] / "training").exists())
        self.assertTrue(
            (
                unit.root
                / unit.settings[0]
                / "final"
                / unit.groups[0]
                / unit.models[0].id
            ).is_symlink()
        )

    def test_internal_only_does_not_require_or_link_final(self) -> None:
        for download in self.config.downloads:
            for setting in self.unit.settings:
                shutil.rmtree(download.local_root / setting / "final", ignore_errors=True)
        unit = replace(self.unit, validations=("internal",))

        merge_unit(self.config, unit)

        self.assertFalse((unit.root / unit.settings[0] / "final").exists())
        self.assertTrue(
            (
                unit.root
                / unit.settings[0]
                / "training"
                / unit.train_name
                / unit.groups[0]
                / unit.models[0].id
            ).is_symlink()
        )

    def test_rebuild_removes_stale_content(self) -> None:
        merge_unit(self.config, self.unit)
        stale = self.unit.root / "stale"
        stale.write_text("old", encoding="utf-8")

        merge_unit(self.config, self.unit)

        self.assertFalse(stale.exists())

    def test_missing_source_does_not_replace_existing_root(self) -> None:
        merge_unit(self.config, self.unit)
        sentinel = self.unit.root / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        missing = next(
            link
            for link in (
                self.config.downloads_by_id["run-a"].local_root
                / self.unit.settings[0]
                / "final"
                / self.unit.groups[0]
            ).iterdir()
            if link.name == "xgboost"
        )
        missing.rmdir()

        with self.assertRaises(FileNotFoundError):
            merge_unit(self.config, self.unit)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


class CollectTests(unittest.TestCase):
    def test_internal_collection_writes_training_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "score_native_modality_0_1.json").write_text(
                json.dumps({"feature": 1.0}), encoding="utf-8"
            )
            (root / "available_fusion.toml").write_text(
                "[weights]\nmodality = 1.0\n", encoding="utf-8"
            )

            write_scores(root, ("modality",), 1)

            self.assertEqual(
                (root / "scores_modality.tsv").read_text(encoding="utf-8"),
                "feature\t1",
            )
            self.assertEqual(
                (root / "weights.tsv").read_text(encoding="utf-8"),
                "modality\t1.0000",
            )

    def test_external_results_are_written_per_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            name_root = root / "eval" / "modality"
            for cohort, value in (("cohort-a", 91), ("cohort-b", 82)):
                cohort_root = name_root / cohort
                cohort_root.mkdir(parents=True)
                (cohort_root / "eval.log").write_text(
                    "Confusion matrix:\n"
                    + " " * 33
                    + "[[1 0]\n"
                    + " " * 33
                    + " [0 1]]\n"
                    + " " * 33
                    + f"AUROC {value}%\n",
                    encoding="utf-8",
                )
            fusion_root = root / "fusion" / "modality"
            fusion_root.mkdir(parents=True)
            (fusion_root / "available_fusion.toml").write_text(
                "[weights]\nmodality = 1.0\n", encoding="utf-8"
            )

            write_results(
                root, ("modality",), ("cohort-a", "cohort-b"), collect_rules=False
            )

            self.assertIn(
                "AUROC 91%",
                (name_root / "cohort-a" / "results.log").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertIn(
                "AUROC 82%",
                (name_root / "cohort-b" / "results.log").read_text(
                    encoding="utf-8"
                ),
            )

    def test_external_only_skips_internal_collection(self) -> None:
        unit = replace(
            CONFIG.units[0],
            root=Path("unused"),
            validations=("external",),
            settings=("setting",),
            groups=("group",),
        )
        with patch("scripts.collect.write_scores") as write_scores, patch(
            "scripts.collect.write_results"
        ) as write_results:
            collect_unit(CONFIG, unit)

        write_scores.assert_not_called()
        self.assertEqual(write_results.call_count, len(unit.models))

    def test_internal_only_skips_external_collection(self) -> None:
        unit = replace(
            CONFIG.units[0],
            root=Path("unused"),
            validations=("internal",),
            settings=("setting",),
            groups=("group",),
        )
        importance_count = sum(
            model.name in CONFIG.collect.importance_models for model in unit.models
        )
        with patch("scripts.collect.write_scores") as write_scores, patch(
            "scripts.collect.write_results"
        ) as write_results:
            collect_unit(CONFIG, unit)

        self.assertEqual(write_scores.call_count, importance_count * len(unit.refs))
        write_results.assert_not_called()


class ExcelTests(unittest.TestCase):
    def test_best_values_are_isolated_by_unit_cohort_and_ref(self) -> None:
        def row(unit: int, cohort: str, ref: str, value: float) -> TableRow:
            return TableRow(
                comparison=(
                    unit,
                    "setting",
                    "external",
                    cohort,
                    "group",
                    "item",
                    ref,
                ),
                setting="same display label",
                stage=f"外部验证（{cohort}）",
                group="group",
                item="item",
                model="model",
                values={"AUROC": value},
            )

        rows = [
            row(0, "a", "ref", 90),
            row(0, "a", "ref", 80),
            row(0, "b", "ref", 70),
            row(1, "a", "ref", 60),
        ]
        bests = best_values(rows, ["AUROC"])

        self.assertEqual(bests[(rows[0].best_key, "AUROC")], 90)
        self.assertEqual(bests[(rows[2].best_key, "AUROC")], 70)
        self.assertEqual(bests[(rows[3].best_key, "AUROC")], 60)

    def test_fusion_rows_keep_configured_models_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            unit = Unit(
                root=Path(temp_dir),
                validations=("internal", "external"),
                internal_cohort="training",
                external_cohorts=("cohort-a", "cohort-b"),
                settings=("setting",),
                train_name="modality",
                names=("name",),
                groups=("group",),
                models=CONFIG.units[0].models,
                refs=("ref",),
            )
            for index, model in enumerate(unit.models):
                value = 80 + index
                training_log = (
                    unit.root
                    / "setting"
                    / "training"
                    / unit.train_name
                    / "group"
                    / "result-replay"
                    / model.id
                    / "ref"
                    / "name"
                    / "train_ci_all_ref_youden.log"
                )
                training_log.parent.mkdir(parents=True, exist_ok=True)
                training_log.write_text(f"AUROC {value}%\n", encoding="utf-8")
                for cohort in unit.external_cohorts:
                    final_log = (
                        unit.root
                        / "setting"
                        / "final"
                        / "group"
                        / model.id
                        / "eval"
                        / "name"
                        / cohort
                        / "results.log"
                    )
                    final_log.parent.mkdir(parents=True, exist_ok=True)
                    final_log.write_text(f"AUROC {value}%\n", encoding="utf-8")

            rows = collect_fusion_rows(TEST_CONFIG, unit, 0, ["AUROC"])
            expected = [TEST_CONFIG.model_map[model.id] for model in unit.models]
            count = len(unit.models)
            self.assertEqual([row.model for row in rows[:count]], expected)
            self.assertEqual(
                [row.stage for row in rows[:count]],
                ["内部验证（训练队列）"] * count,
            )
            self.assertEqual(
                [row.model for row in rows[count : count * 2]], expected
            )
            self.assertEqual(
                [row.stage for row in rows[count : count * 2]],
                ["外部验证（队列 A）"] * count,
            )
            self.assertEqual(
                [row.model for row in rows[count * 2 :]], expected
            )
            self.assertEqual(
                [row.stage for row in rows[count * 2 :]],
                ["外部验证（队列 B）"] * count,
            )

    def test_external_only_rows_do_not_read_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            unit = Unit(
                root=Path(temp_dir),
                validations=("external",),
                internal_cohort="training",
                external_cohorts=("cohort-a",),
                settings=("setting",),
                train_name="modality",
                names=("name",),
                groups=("group",),
                models=(CONFIG.units[0].models[0],),
                refs=("ref",),
            )
            final_log = (
                unit.root
                / "setting"
                / "final"
                / "group"
                / unit.models[0].id
                / "eval"
                / "name"
                / "cohort-a"
                / "results.log"
            )
            final_log.parent.mkdir(parents=True)
            final_log.write_text("AUROC 91%\n", encoding="utf-8")

            rows = collect_fusion_rows(TEST_CONFIG, unit, 0, ["AUROC"])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].stage, "外部验证（队列 A）")
            self.assertEqual(
                collect_modality_rows(TEST_CONFIG, unit, 0, ["AUROC"]), []
            )

    def test_external_only_workbook_omits_modality_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unit = Unit(
                root=root,
                validations=("external",),
                internal_cohort="training",
                external_cohorts=("cohort-a",),
                settings=("setting",),
                train_name="modality",
                names=("name",),
                groups=("group",),
                models=(CONFIG.units[0].models[0],),
                refs=("ref",),
            )
            final_log = (
                root
                / "setting"
                / "final"
                / "group"
                / unit.models[0].id
                / "eval"
                / "name"
                / "cohort-a"
                / "results.log"
            )
            final_log.parent.mkdir(parents=True)
            final_log.write_text("AUROC 91%\n", encoding="utf-8")
            output = root / "results.xlsx"

            build_workbook(replace(TEST_CONFIG, units=(unit,)), ["AUROC"], output)

            with zipfile.ZipFile(output) as workbook:
                xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
            namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            names = [
                sheet.attrib["name"] for sheet in xml.findall("x:sheets/x:sheet", namespace)
            ]
            self.assertEqual(names, ["校准融合"])


if __name__ == "__main__":
    unittest.main()
