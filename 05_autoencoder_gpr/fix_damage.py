#!/usr/bin/env python3
"""
Helper script for re-running the improved AE+GPR damage workflow.
"""

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import REPO_ROOT


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and report status."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"{'=' * 70}")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True, capture_output=False, text=True)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] {description} failed with exit code {exc.returncode}")
        return False
    except Exception as exc:
        print(f"[ERROR] {description} failed: {exc}")
        return False


def main() -> None:
    base = REPO_ROOT / "05_autoencoder_gpr"

    print("=" * 70)
    print("AE+GPR DAMAGE WORKFLOW HELPER")
    print("=" * 70)
    print("\nThis helper will:")
    print("  1. Retrain the improved autoencoders")
    print("  2. Prompt you to re-encode curves")
    print("  3. Prompt you to retrain the latent GPR models")
    print("  4. Run the evaluation script")
    print(f"\nBase directory: {base}")

    response = input("\nProceed? (y/n): ")
    if response.lower() != "y":
        print("Aborted.")
        return

    script1 = base / "02_train_autoencoders.py"
    if not script1.exists():
        print(f"\n[ERROR] Script not found: {script1}")
        return

    if not run_script(script1, "Step 1: Retrain improved autoencoders"):
        print("\n[WARN] Autoencoder training failed. Please check the logs above.")
        return

    print("\n" + "=" * 70)
    print("Step 2: Re-encode curves")
    print("=" * 70)
    print("\nRun this next:")
    print("  python 05_autoencoder_gpr/03_encode_curves.py")
    response = input("\nHave you run 03_encode_curves.py? (y/n): ")
    if response.lower() != "y":
        print("\nPlease run it, then restart this helper.")
        return

    print("\n" + "=" * 70)
    print("Step 3: Retrain latent GPR models")
    print("=" * 70)
    print("\nRun this next:")
    print("  python 05_autoencoder_gpr/04_train_gpr.py")
    response = input("\nHave you run 04_train_gpr.py? (y/n): ")
    if response.lower() != "y":
        print("\nPlease run it, then restart this helper.")
        return

    script4 = base / "05_evaluate_model.py"
    if script4.exists():
        run_script(script4, "Step 4: Evaluate the updated surrogate")
    else:
        print(f"\n[WARN] Evaluation script not found: {script4}")

    print("\n" + "=" * 70)
    print("PROCESS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review outputs in 05_autoencoder_gpr/output_evaluation/")
    print("  2. Check training summaries in the improved output directories")
    print("  3. Re-run downstream UQ if the updated metrics look good")
    print("\nTo use the improved model in downstream scripts:")
    print("  from ae_surrogate_model import ImprovedAESurrogateModel")
    print("  model = ImprovedAESurrogateModel(base, use_improved=True)")


if __name__ == "__main__":
    main()
