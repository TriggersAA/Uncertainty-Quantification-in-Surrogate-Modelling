# # Probabilistic envelope (5–95%)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv(r"C:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env\Scripts\03_postprocess\results_4\load_displacement_full.csv")

# # Interpolate to common displacement grid
# u_grid = np.linspace(df["U2"].min(), df["U2"].max(), 200)

# responses = []

# for job, g in df.groupby("job"):
#     responses.append(np.interp(u_grid, g["U2"], g["RF2"]))

# responses = np.array(responses)

# mean = responses.mean(axis=0)
# p05 = np.percentile(responses, 5, axis=0)
# p95 = np.percentile(responses, 95, axis=0)

# plt.figure(figsize=(7, 5))
# plt.plot(u_grid, mean, lw=2, label="Mean")
# plt.fill_between(u_grid, p05, p95, alpha=0.3, label="5–95% Envelope")

# plt.xlabel("Displacement U2 [mm]")
# plt.ylabel("Reaction Force RF2 [N]")
# plt.title("Probabilistic Load–Displacement Envelope")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("results/postprocessed_4/Probabilistic Load–Displacement Envelope.png", dpi=300)
# plt.show()






# ============================================================
# Probabilistic Load–Displacement Envelope (Refined Plot)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import repo_path

plt.rcParams["font.family"] = "Times New Roman"

# Load data
df = pd.read_csv(repo_path("03_postprocess", "01_extracted_data", "load_displacement_full.csv"))

# Create save directory
SAVE_DIR = repo_path("Plottings", "results", "postprocessed_5")
os.makedirs(SAVE_DIR, exist_ok=True)

# Interpolate to common displacement grid
u_grid = np.linspace(df["U2"].min(), df["U2"].max(), 300)

responses = []
for job, g in df.groupby("job"):
    responses.append(np.interp(u_grid, g["U2"], g["RF2"]))

responses = np.array(responses)

# Statistics
mean_curve = responses.mean(axis=0)
p05 = np.percentile(responses, 5, axis=0)
p95 = np.percentile(responses, 95, axis=0)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Envelope shading
ax.fill_between(
    u_grid, p05, p95,
    color="steelblue",
    alpha=0.25,
    label="5–95% Envelope"
)

# Mean curve
ax.plot(
    u_grid, mean_curve,
    color="darkred",
    linewidth=2.2,
    label="Mean Response"
)

# Axis labels
ax.set_xlabel("Displacement $U_2$ [mm]", fontsize=13)
ax.set_ylabel("Reaction Force $RF_2$ [N]", fontsize=13)

# Title
ax.set_title("Probabilistic Load–Displacement Envelope", fontsize=15)

# Grid and legend
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=12)

plt.tight_layout()

# Save
save_path = os.path.join(str(SAVE_DIR), "probabilistic_load_displacement_envelope.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved refined plot to: {save_path}")
