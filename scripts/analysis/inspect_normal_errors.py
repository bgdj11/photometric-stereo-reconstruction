from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.algorithms.ls import angular_error

DATASET = 'complex/terrain/baseline'
INTERACTIVE = True

def draw_errors_3d(folder, show=False):
    points = np.load(folder/'surface_points.npy')
    truth = np.load(folder/'reference_normals.npy')
    estimated = np.load(folder/'estimated_normals.npy')
    mask = np.linalg.norm(truth, axis=-1) > 0

    for _ in range(2):
        pad = np.pad(mask, 1, constant_values=False)
        mask = np.logical_and.reduce([pad[i:i+mask.shape[0], j:j+mask.shape[1]]
                                      for i in range(3) for j in range(3)])
    mask &= np.isfinite(points).all(axis=-1) & np.isfinite(estimated).all(axis=-1)

    error = angular_error(estimated, truth)
    xyz = np.where(mask[..., None], points, np.nan)
    color_scale = Normalize(0, max(0.01, float(error[mask].max())))
    colors = plt.get_cmap('magma')(color_scale(np.where(mask, error, np.nan)))


    # 3 random pixels not close to each other
    pixels = np.argwhere(mask)
    order = np.argsort(error[mask])[::-1]
    selected = []

    for index in order:
        pixel = pixels[index]
        if all(np.linalg.norm(pixel-other) > min(mask.shape)*0.2 for other in selected):
            selected.append(pixel)
        if len(selected) == 3:
            break

    fig = plt.figure(figsize=(15, 6))
    left = fig.add_subplot(121, projection='3d', computed_zorder=False)
    right = fig.add_subplot(122, projection='3d', computed_zorder=False)


    left.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], facecolors=colors,
                      rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False, zorder=1)
    right.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], color='lightgray',
                       rstride=4, cstride=4, alpha=0.65, linewidth=0, zorder=1)
    colorbar = plt.cm.ScalarMappable(norm=color_scale, cmap='magma')
    fig.colorbar(colorbar, ax=left, shrink=0.5, pad=0.16, label='normal error (degrees)')
    low, high = points[mask].min(axis=0), points[mask].max(axis=0)
    arrow_length = 0.12 * float(np.max(high-low))
    labels = []

    for number, (row, col) in enumerate(selected, start=1):
        point = points[row, col]

        for normal, color in [(truth[row, col], 'tab:green'), (estimated[row, col], 'tab:red')]:
            right.quiver(*point, *normal, length=arrow_length, normalize=True,
                         color=color, linewidth=2, arrow_length_ratio=0.2, zorder=3)
            
        for ax in (left, right):
            ax.scatter(*point, color='deepskyblue', s=20, depthshade=False, zorder=4)
            ax.text(*point, str(number), color='deepskyblue', zorder=5)
        labels.append(f'{number}: pixel ({row}, {col}), error {error[row, col]:.2f} deg')
    right.legend(handles=[Line2D([0], [0], color='tab:green', label='Known normal'),
                          Line2D([0], [0], color='tab:red', label='Estimated normal')], loc='upper left')
    left.set_title('Error mapped onto the original surface')
    right.set_title('Known and estimated normals at marked points')

    for ax in (left, right):
        ax.set(xlabel='world X', ylabel='world Y', zlabel='world Z',
               xlim=(low[0]-arrow_length, high[0]+arrow_length),
               ylim=(low[1]-arrow_length, high[1]+arrow_length),
               zlim=(low[2]-arrow_length*0.2, high[2]+arrow_length*1.2))
        ax.set_box_aspect((high-low) + [2*arrow_length, 2*arrow_length, 1.4*arrow_length])
        ax.view_init(elev=32, azim=-55)


    fig.suptitle(f'{folder.parent.name} / {folder.name}: original geometry, not reconstructed height')
    fig.text(0.5, 0.025, ' | '.join(labels), ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(folder/'normal_errors_3d.png', dpi=140)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    draw_errors_3d(PROJECT/'data'/DATASET, show=INTERACTIVE)
