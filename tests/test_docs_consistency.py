import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationConsistencyTests(unittest.TestCase):
    def test_no_known_stale_doc_references(self):
        docs = [
            ROOT / "README.md",
            ROOT / "00_README" / "README.md",
            ROOT / "01_samplying" / "README.md",
            ROOT / "02_abaqus" / "README.md",
            ROOT / "04_PCA" / "README.md",
            ROOT / "05_autoencoder_gpr" / "README.md",
            ROOT / "06_shape_scale_gpr" / "README.md",
        ]
        forbidden = [
            "01_generate_lhs_samples.py",
            "processed_inputs_final.csv",
            "step1_pca_reduction.py",
            "step2_train_surrogate.py",
            "step3_validate_reconstruction.py",
            "step4_interactive_gui.py",
            "Scripts_2_0",
        ]
        for doc in docs:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                with self.subTest(document=doc.name, token=token):
                    self.assertNotIn(token, text)

    def test_no_common_mojibake_in_main_docs(self):
        docs = [
            ROOT / "README.md",
            ROOT / "00_README" / "README.md",
            ROOT / "02_abaqus" / "README.md",
            ROOT / "04_PCA" / "README.md",
            ROOT / "05_autoencoder_gpr" / "README.md",
            ROOT / "06_shape_scale_gpr" / "README.md",
        ]
        bad_tokens = ["â€", "â†", "âœ", "â€”", "â€“", "RÂ²", "MatÃ", "Ã—", "Ï", "ðŸ"]
        for doc in docs:
            text = doc.read_text(encoding="utf-8", errors="ignore")
            for token in bad_tokens:
                with self.subTest(document=doc.name, token=token):
                    self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
