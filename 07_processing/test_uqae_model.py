from uncertainty_quantification_07 import UQAESurrogateModel
import traceback
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import REPO_ROOT

try:
    m = UQAESurrogateModel(str(REPO_ROOT))
    print("UQ_AE_MODEL_OK")
except Exception as e:
    print("UQ_AE_MODEL_ERROR", type(e).__name__, str(e))
    traceback.print_exc()
