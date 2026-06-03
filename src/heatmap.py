##Figure 1 敏感区域热力图

import numpy as np
import matplotlib.pyplot as plt

grid_size = 10

heatmap = np.full(
    (grid_size, grid_size),
    0.2
)

center = grid_size // 2

heatmap[center-1:center+1,
        center-1:center+1] = 0.9

plt.figure(figsize=(8,6))

plt.imshow(
    heatmap,
    cmap='hot'
)

plt.colorbar(
    label='Sensitivity'
)

plt.title(
    'Sensitive Region Heatmap'
)

plt.savefig(
    '../figures/heatmap.png',
    dpi=300
)

plt.show()