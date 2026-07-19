import os
import tempfile
import unittest
from pathlib import Path

from config import CONFIG, Unit, load_config
from scripts.merge import MARKER, merge_unit
from scripts.to_excel import TableRow, best_values, collect_fusion_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")


class ConfigTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text(text, encoding="utf-8")
            return load_config(config_file)

    def assert_config_error(self, old: str, new: str, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            self.load_text(EXAMPLE_CONFIG.replace(old, new))

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
            '{ name = "rrl", suffix = "-variant", download = "run-b" }',
            '{ name = "rrl", download = "run-b" }',
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


class ExcelTests(unittest.TestCase):
    def test_best_values_are_isolated_by_unit_and_ref(self) -> None:
        def row(unit: int, ref: str, value: float) -> TableRow:
            return TableRow(
                comparison=(unit, "setting", "training", "group", "item", ref),
                setting="same display label",
                stage="交叉验证",
                group="group",
                item="item",
                model="model",
                values={"AUROC": value},
            )

        rows = [row(0, "a", 90), row(0, "a", 80), row(1, "a", 70), row(0, "b", 60)]
        bests = best_values(rows, ["AUROC"])

        self.assertEqual(bests[(rows[0].best_key, "AUROC")], 90)
        self.assertEqual(bests[(rows[2].best_key, "AUROC")], 70)
        self.assertEqual(bests[(rows[3].best_key, "AUROC")], 60)

    def test_fusion_rows_keep_five_models_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            unit = Unit(
                root=Path(temp_dir),
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
                final_log = (
                    unit.root
                    / "setting"
                    / "final"
                    / "group"
                    / model.id
                    / "eval"
                    / "name"
                    / "results.log"
                )
                for log in (training_log, final_log):
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(f"AUROC {value}%\n", encoding="utf-8")

            rows = collect_fusion_rows(CONFIG, unit, 0, ["AUROC"])
            expected = [CONFIG.model_map[model.id] for model in unit.models]
            self.assertEqual([row.model for row in rows[:5]], expected)
            self.assertEqual([row.stage for row in rows[:5]], ["交叉验证"] * 5)
            self.assertEqual([row.model for row in rows[5:]], expected)
            self.assertEqual([row.stage for row in rows[5:]], ["外部验证"] * 5)


if __name__ == "__main__":
    unittest.main()
