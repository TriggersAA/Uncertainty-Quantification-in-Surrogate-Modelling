import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import REPO_ROOT, repo_path


class RepoSmokeTests(unittest.TestCase):
    def test_repo_root_is_stable(self):
        self.assertEqual(REPO_ROOT, ROOT)
        self.assertTrue(repo_path("README.md").exists())

    def test_core_metadata_exists(self):
        required = [
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "requirements.txt",
            ".gitignore",
            ".env.example",
            "project_paths.py",
        ]
        for rel in required:
            with self.subTest(path=rel):
                self.assertTrue(repo_path(rel).exists(), f"Missing {rel}")

    def test_quickstart_scripts_exist(self):
        required = [
            "01_samplying/01_generate_ihs_samples.py",
            "01_samplying/02_visualize_samples.py",
            "04_PCA/01_pca_reduction.py",
            "05_autoencoder_gpr/01_preprocess_data.py",
            "07_processing/run_uq_pipeline.py",
        ]
        for rel in required:
            with self.subTest(path=rel):
                self.assertTrue(repo_path(*rel.split("/")).exists(), f"Missing {rel}")

    def test_active_source_has_no_live_machine_specific_paths(self):
        source_roots = [
            repo_path("01_samplying"),
            repo_path("02_abaqus"),
            repo_path("03_postprocess"),
            repo_path("04_PCA"),
            repo_path("05_autoencoder_gpr"),
            repo_path("06_shape_scale_gpr"),
            repo_path("07_processing"),
            repo_path("augmentation_physics_fixed"),
            repo_path("Plottings"),
        ]
        skip_parts = {"__pycache__", "data_preprocessed", "01_extracted_data", "10_final_outputs"}
        bad_hits = []
        for root in source_roots:
            for path in root.rglob("*.py"):
                if any(part in skip_parts for part in path.parts):
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if line.lstrip().startswith("#"):
                        continue
                    if r"C:\Users\jidro" in line or r"C:/Users/jidro" in line:
                        bad_hits.append(f"{path.relative_to(ROOT)}:{lineno}")
        self.assertEqual(bad_hits, [], f"Found machine-specific paths: {bad_hits}")


if __name__ == "__main__":
    unittest.main()
