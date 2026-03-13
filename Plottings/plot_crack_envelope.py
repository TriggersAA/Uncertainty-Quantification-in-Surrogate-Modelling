# Mean crack evolution + uncertainty
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import repo_path

df = pd.read_csv(repo_path("03_postprocess", "01_extracted_data", "damage_evolution_full.csv"))
out_path = repo_path("Plottings", "results", "Probabilistic_Damage_Evolution.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

u_grid = np.linspace(df["U2"].min(), df["U2"].max(), 200)

def envelope(var):
    curves = []
    for job, g in df.groupby("job"):
        curves.append(np.interp(u_grid, g["U2"], g[var]))
    curves = np.array(curves)
    return curves.mean(axis=0), np.percentile(curves, 95, axis=0)

plt.figure(figsize=(7, 5))

for var, label in [
    ("DAMAGET_max", "Tension Damage"),
    ("SDEG_max", "Stiffness Degradation"),
]:
    mean, p95 = envelope(var)
    plt.plot(u_grid, mean, label=f"{label} (mean)")
    plt.plot(u_grid, p95, "--", label=f"{label} (95%)")

plt.xlabel("Displacement U2 [mm]")
plt.ylabel("Damage Index")
plt.title("Probabilistic Damage Evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.show()
