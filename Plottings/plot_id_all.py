# Combined load–displacement (all samples)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\jidro\\Documents\\Elijah\\RUB\\Third Semester\\Uncertainty FEM\\Project\\ufem_env\\Scripts\\03_postprocess\\results_4\\load_displacement_full.csv")

plt.figure(figsize=(7, 5))

for job, g in df.groupby("job"):
    plt.plot(g["U2"], g["RF2"], alpha=0.3)

plt.xlabel("Displacement U2 [mm]")
plt.ylabel("Reaction Force RF2 [N]")
plt.title("Load–Displacement Curves (All Samples)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/postprocessed_4/load_displacement_plot.png", dpi=300)
plt.show() # Show always comes AFTER savefig