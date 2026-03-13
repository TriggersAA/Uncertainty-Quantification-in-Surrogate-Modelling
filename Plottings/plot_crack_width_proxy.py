# Crack width proxy
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import repo_path

df = pd.read_csv(repo_path("03_postprocess", "01_extracted_data", "damage_evolution_full.csv"))
out_path = repo_path("Plottings", "results", "Crack_Width_Proxy_vs_Displacement.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7, 5))

for job, g in df.groupby("job"):
    plt.plot(g["U2"], g["PEEQ_max"], alpha=0.3)

plt.xlabel("Displacement U2 [mm]")
plt.ylabel("Max PEEQ (crack width proxy)")
plt.title("Crack Width Proxy vs Displacement")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()
