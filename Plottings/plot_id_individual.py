# # Plot load–displacement for EACH SAMPLE (individual curves)

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# DATA = Path(r"C:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env\Scripts\03_postprocess\results_5\load_displacement_full.csv")
# OUT = Path("results/plots_4/ld_individual")
# OUT.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(DATA)

# for job, g in df.groupby("job"):
#     plt.figure(figsize=(6, 4))
#     plt.plot(g["U2"], g["RF2"], lw=2)
#     plt.xlabel("Displacement U2 [mm]")
#     plt.ylabel("Reaction Force RF2 [N]")
#     plt.title(f"Load–Displacement: {job}")
#     plt.grid(True)

#     plt.savefig(OUT / f"{job}_ld.png", dpi=300)
#     plt.close()

# print("Individual load–displacement plots saved.")





















# # Plot load–displacement for EACH SAMPLE (individual curves)

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# DATA = Path(r"C:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env\Scripts\augmentation_physics_fixed\load_displacement_full_aug.csv")
# OUT = Path("results/plots_6/ld_individual")
# OUT.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(DATA)

# for job, g in df.groupby("job_aug"):
#     plt.figure(figsize=(6, 4))
#     plt.plot(g["U2"], g["RF2"], lw=2)
#     plt.xlabel("Displacement U2 [mm]")
#     plt.ylabel("Reaction Force RF2 [N]")
#     plt.title(f"Load–Displacement: {job}")
#     plt.grid(True)

#     plt.savefig(OUT / f"{job}_ld.png", dpi=300)
#     plt.close()

# print("Individual load–displacement plots saved.")






























# ============================================================
# Individual Load–Displacement Curves (Refined Plot Style)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import repo_path

plt.rcParams["font.family"] = "Times New Roman"

# Paths
DATA = repo_path("03_postprocess", "01_extracted_data", "load_displacement_full.csv")
OUT = repo_path("Plottings", "results", "plots_6", "ld_individual")
OUT.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA)

# Loop through each augmented job
for job, g in df.groupby("job"):

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot curve
    ax.plot(
        g["U2"], g["RF2"],
        color="darkred",
        linewidth=2.0,
        label="Load–Displacement Curve"
    )

    # Labels
    ax.set_xlabel("Displacement $U_2$ [mm]", fontsize=12)
    ax.set_ylabel("Reaction Force $RF_2$ [N]", fontsize=12)

    # Title
    ax.set_title(f"Load–Displacement Curve — Sample {job}", fontsize=14)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legend
    ax.legend(fontsize=11)

    # Layout + Save
    plt.tight_layout()
    plt.savefig(OUT / f"{job}_ld.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Individual load–displacement plots saved.")
