# Probabilistic Crack Width Evolution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\jidro\\Documents\\Elijah\\RUB\\Third Semester\\Uncertainty FEM\\Project\\ufem_env\\Scripts\\03_postprocess\\results_3\\crack_proxy_vs_displacement.csv")

ELEMENT_SIZE = 0.025  # meters
df["crack_width"] = df["max_PEEQ"] * ELEMENT_SIZE

# Interpolate onto common displacement grid
disp_grid = np.linspace(0, df["disp"].max(), 100)

curves = []

for job, g in df.groupby("job"):
    curves.append(np.interp(disp_grid, g["disp"], g["crack_width"]))

curves = np.array(curves)

plt.fill_between(
    disp_grid,
    np.percentile(curves, 5, axis=0),
    np.percentile(curves, 95, axis=0),
    alpha=0.3,
    label="5–95%"
)

plt.plot(disp_grid, curves.mean(axis=0), label="Mean")
plt.xlabel("Displacement (mm)")
plt.ylabel("Crack width proxy (m)")
plt.legend()
plt.grid(True)
plt.savefig("results/postprocessed/Probabilistic Crack Width Evolution.png", dpi=300)
plt.show()
