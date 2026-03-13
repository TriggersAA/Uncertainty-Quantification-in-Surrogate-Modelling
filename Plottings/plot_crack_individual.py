# # Crack evolution vs displacement (per sample)
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# df = pd.read_csv(r"C:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env\Scripts\03_postprocess\results_4\crack_evolution_full.csv")


# OUT = Path("results/plots_4s/crack_individual")
# OUT.mkdir(parents=True, exist_ok=True)

# for job, g in df.groupby("job"):
#     plt.figure(figsize=(6, 4))
#     plt.plot(g["U2"], g["DAMAGET_max"], label="Tension Damage")
#     plt.plot(g["U2"], g["DAMAGEC_max"], label="Compression Damage")
#     plt.plot(g["U2"], g["SDEG_max"], label="Stiffness Degradation")

#     plt.xlabel("Displacement U2 [mm]")
#     plt.ylabel("Damage Index")
#     plt.title(f"Crack Evolution: {job}")
#     plt.legend()
#     plt.grid(True)

#     plt.savefig(OUT / f"{job}_crack.png", dpi=300)
    
#     plt.close()

# print("[OK] Individual crack evolution plots saved.")































# # ============================================================
# # Crack Evolution vs Displacement (Refined Plot Style)
# # ============================================================

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# plt.rcParams["font.family"] = "Times New Roman"

# # Load data
# df = pd.read_csv(
#     r"C:\Users\jidro\Documents\Elijah\RUB\Third Semester\Uncertainty FEM\Project\ufem_env\Scripts\03_postprocess\results_4\crack_evolution_full.csv"
# )

# # Output directory
# OUT = Path("results/plots_4s/crack_individual")
# OUT.mkdir(parents=True, exist_ok=True)

# # Loop through each job
# for job, g in df.groupby("job"):

#     fig, ax = plt.subplots(figsize=(7, 4.5))

#     # Plot each damage metric
#     ax.plot(
#         g["U2"], g["DAMAGET_max"],
#         color="darkred",
#         linewidth=2.0,
#         label="Tension Damage"
#     )

#     ax.plot(
#         g["U2"], g["DAMAGEC_max"],
#         color="steelblue",
#         linewidth=2.0,
#         label="Compression Damage"
#     )

#     ax.plot(
#         g["U2"], g["SDEG_max"],
#         color="darkgreen",
#         linewidth=2.0,
#         label="Stiffness Degradation"
#     )

#     # Labels
#     ax.set_xlabel("Displacement $U_2$ [mm]", fontsize=12)
#     ax.set_ylabel("Damage Index [-]", fontsize=12)

#     # Title
#     ax.set_title(f"Crack Evolution — Sample {job}", fontsize=14)

#     # Grid
#     ax.grid(True, linestyle="--", alpha=0.6)

#     # Legend
#     ax.legend(fontsize=11)

#     # Layout + Save
#     plt.tight_layout()
#     plt.savefig(OUT / f"{job}_crack.png", dpi=300, bbox_inches="tight")
#     plt.close()

# print("[OK] Individual crack evolution plots saved.")




# =============================================
# ============================================================
# Compression Damage vs Displacement (Refined Plot Style)
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

# Load data
df = pd.read_csv(repo_path("03_postprocess", "01_extracted_data", "damage_evolution_full.csv"))

# Output directory
OUT = repo_path("Plottings", "results", "plots_4s", "crack_individual", "DAMAGEC")
OUT.mkdir(parents=True, exist_ok=True)

# Loop through each job
for job, g in df.groupby("job"):

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Compression damage only
    ax.plot(
        g["U2"], g["DAMAGEC_max"],
        color="steelblue",
        linewidth=2.2,
        label="Compression Damage"
    )

    # Labels
    ax.set_xlabel("Displacement $U_2$ [mm]", fontsize=12)
    ax.set_ylabel("Compression Damage Index [-]", fontsize=12)

    # Title
    ax.set_title(f"Compression Damage Evolution — Sample {job}", fontsize=14)

    # Grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # Legend
    ax.legend(fontsize=11)

    # Layout + Save
    plt.tight_layout()
    plt.savefig(OUT / f"{job}_compression_damage.png", dpi=300, bbox_inches="tight")
    plt.close()

print("[OK] Compression damage plots saved.")
