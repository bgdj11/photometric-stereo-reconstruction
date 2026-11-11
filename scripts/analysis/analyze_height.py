from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.algorithms.integrate_normals import reconstruct_height

DATASET = 'complex/terrain/baseline'
IMAGE_WIDTH = 1.0
INTERACTIVE = True


def analyze_height(folder, show=False):

    normals = np.load(folder / 'estimated_normals.npy')
    heights = reconstruct_height(normals, IMAGE_WIDTH)
    np.save(folder / 'estimated_height.npy', heights)


    reference_file = folder / 'surface_points.npy'
    if reference_file.exists():
        points = np.load(reference_file)
        if points.shape != normals.shape:
            raise ValueError('not same shape')

        
        original_step = float(np.nanmedian(np.diff(points[..., 0], axis=1)))
        if not np.isfinite(original_step) or original_step <= 0:
            raise ValueError('Poredjenje ocekuje ortografsku kameru odozgo, poravnatu sa X/Y.')
        scale = (IMAGE_WIDTH / heights.shape[1]) / original_step
        reference = points[..., 2] * scale

        # dont use edge pixels for check - problem with background from blender
        mask = binary_erosion(np.isfinite(reference), structure=np.ones((3, 3)), iterations=2)
        mask &= np.isfinite(heights)
        if not mask.any():
            raise ValueError('Nema zajednickih unutrasnjih piksela za poredjenje.')
        reference = reference - reference[mask].min()


        # translation for allignement
        aligned = heights + np.mean(reference[mask] - heights[mask])
        error = np.abs(aligned - reference)
        maps = [reference, aligned, error]
        titles = ['Original height', 'Estimated height (offset aligned)', 'Absolute height error']
        rmse = np.sqrt(np.mean(error[mask]**2))
        note = f'Independent color scales: compare the numbers. 2-pixel rim excluded; offset aligned; RMSE = {rmse:.6g}'
        print(f'{folder.parent.name}: height RMSE = {rmse:.8f} (frame width = {IMAGE_WIDTH})')
    else:
        mask = np.isfinite(heights)
        maps, titles = [heights], ['Estimated height']
        note = 'No reference available'

    fig, axes = plt.subplots(1, len(maps), figsize=(5*len(maps), 4.8), squeeze=False)
    for index, (ax, values, title) in enumerate(zip(axes.flat, maps, titles)):
        is_error = index == 2
        # Svaka mapa ima svoju skalu: losa procena ne menja boje originala.
        vmin = 0 if is_error else float(values[mask].min())
        vmax = float(values[mask].max())
        plot = ax.imshow(np.where(mask, values, np.nan), origin='upper',
                         interpolation='nearest', cmap='magma' if is_error else 'terrain',
                         vmin=vmin, vmax=max(vmax, vmin+1e-12))
        ax.set_title(f'{title}\nrange: {vmin:.4f} to {vmax:.4f}', fontsize=11)
        ax.set_axis_off()
        fig.colorbar(plot, ax=ax, shrink=0.75, label='relative height units')
    fig.suptitle(f'{folder.parent.name} / {folder.name} | frame width = {IMAGE_WIDTH}')
    fig.text(0.5, 0.02, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(folder / 'height_comparison.png', dpi=140)
    
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    analyze_height(PROJECT / 'data' / DATASET, show=INTERACTIVE)
