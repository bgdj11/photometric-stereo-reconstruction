from pathlib import Path
import sys
import numpy as np
import OpenEXR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.algorithms.ls import estimate_normals, angular_error
from scripts.analysis.inspect_normal_errors import draw_errors_3d

DATA = PROJECT / 'data'
DATASET = None  # None = all or 'simple/pyramid/baseline'.


def load_images(folder, count):
    images = []
    for index in range(count):
        with OpenEXR.File(str(folder/'images'/f'light_{index:02d}.exr'), separate_channels=True) as file:
            rgb = np.stack([file.channels()[c].pixels for c in 'RGB'], axis=-1)
            images.append(rgb.mean(axis=-1))
    return np.array(images, dtype=np.float64)


def interior_mask(normals):
    mask = np.linalg.norm(normals, axis=-1) > 0
    for _ in range(2):
        pad = np.pad(mask, 1, constant_values=False)
        mask = np.logical_and.reduce([pad[i:i+mask.shape[0], j:j+mask.shape[1]]
                                      for i in range(3) for j in range(3)])
    return mask


def plot_results(folder, images, truth, estimated, mask, error):
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)

    for ax in axes.flat:
        ax.axis('off')

    for index, (ax, photo) in enumerate(zip(axes.flat, images)):
        ax.imshow(photo, cmap='gray', vmin=0, vmax=float(images.max()))
        ax.set_title(f'Light {index}')

    fig.suptitle('Input images: fixed camera and surface, changing light direction')
    fig.tight_layout()
    fig.savefig(folder/'images.png', dpi=130)
    plt.close(fig)


    def colors(normals):
        return np.where(mask[..., None], np.nan_to_num((normals+1)/2), 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(colors(truth))
    axes[0].set_title('Known normals (evaluation only)')
    axes[1].imshow(colors(estimated))
    axes[1].set_title('Normals estimated by LS')
    plot = axes[2].imshow(np.where(mask, error, np.nan), cmap='magma',
                          vmin=0, vmax=max(0.01, float(error[mask].max())))
    axes[2].set_title('Angular error')
    fig.colorbar(plot, ax=axes[2], label='degrees', shrink=0.75)
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(folder/'normals.png', dpi=130)
    plt.close(fig)


def estimate_dataset_normals(folder):
    L = np.loadtxt(folder/'lights.txt', ndmin=2)
    images = load_images(folder, len(L))

    estimated, scale, residual, valid = estimate_normals(images, L)
    np.save(folder/'estimated_normals.npy', estimated)


    truth = np.load(folder/'reference_normals.npy')
   
    interior = interior_mask(truth)
    mask = interior & valid

    error = angular_error(estimated, truth)
    errors = error[mask]
    min_dot = float(np.min(truth[interior] @ L.T))
    positive = bool(images[:, interior].min() > 0)

    report = (f'Mean angular error: {errors.mean():.6f} deg\n'
              f'95th percentile: {np.percentile(errors, 95):.6f} deg\n'
              f'Maximum: {errors.max():.6f} deg\n'
              f'Evaluated pixels: {mask.sum()}\nInvalid interior pixels: {(interior & ~valid).sum()}\n'
              f'Light matrix condition: {np.linalg.cond(L):.6f}\n'
              f'Mean intensity residual: {residual[mask].mean():.8f}\n'
              f'Minimum reference normal dot light: {min_dot:.6f}\n'
              f'All interior observations positive: {positive}\n'
              'Method: ordinary LS; no IRLS, no height reconstruction.\n')
    
    (folder/'results.txt').write_text(report, encoding='utf-8')
    plot_results(folder, images, truth, estimated, mask, error)

    if (folder/'surface_points.npy').exists():
        draw_errors_3d(folder, show=False)

    print(f'{folder.relative_to(DATA)}: mean error {errors.mean():.6f} deg')


if __name__ == '__main__':
    folders = ([DATA/DATASET] if DATASET else sorted(p.parent for p in DATA.glob('*/*/*/lights.txt')))
    for folder in folders:
        estimate_dataset_normals(folder)
