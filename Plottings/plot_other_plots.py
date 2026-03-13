import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import repo_path

# 1. Setup paths and directories
csv_path = repo_path("03_postprocess", "01_extracted_data", "load_displacement_full.csv")
output_dir = repo_path("Plottings", "results", "postprocessed")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Load and clean data
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# 3. Peak Load Distribution
plt.figure()
peak = df.groupby("job")["RF2"].max()
peak.hist(bins=15, edgecolor='black')
plt.xlabel("Peak Load [N]")
plt.ylabel("Frequency")
plt.title("Distribution of Peak Load Capacity")
plt.savefig(output_dir / "Distribution_of_Peak_Load_Capacity.png", dpi=300)
plt.show()

# 4. Failure Displacement Distribution
plt.figure()
fail_disp = df.groupby("job")["U2"].max()
fail_disp.hist(bins=15, color='orange', edgecolor='black')
plt.xlabel("Failure Displacement [mm]")
plt.ylabel("Frequency")
plt.title("Displacement at Failure Distribution")
plt.savefig(output_dir / "Displacement_at_Failure_Distribution.png", dpi=300)
plt.show()
