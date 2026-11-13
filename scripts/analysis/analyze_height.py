from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.algorithms.integrate_normals import reconstruct_height

DATASET = 'complex/terrain/baseline'
INTERACTIVE = True


def analyze_height(folder, show=False):
    settings = dict(line.split(': ', 1) for line in (folder/'settings.txt').read_text().splitlines() if ': ' in line)
    image_width = float(settings['Ortho scale'])
    normals = np.load(folder / 'estimated_normals.npy')
    heights = reconstruct_height(normals, image_width)
    np.save(folder / 'estimated_height.npy', heights)


    reference_file = folder / 'surface_points.npy'
    if reference_file.exists():
        points = np.load(reference_file)
        if points.shape != normals.shape:
            raise ValueError('not same shape')

        reference = points[..., 2]

        # dont use edge pixels for check - problem with background from blender
        mask = binary_erosion(np.isfinite(reference), structure=np.ones((3, 3)), iterations=2)
        mask &= np.isfinite(heights)
        if not mask.any():
            raise ValueError('Nema zajednickih unutrasnjih piksela za poredjenje.')


        aligned = heights + np.mean(reference[mask] - heights[mask])
        error = np.abs(aligned - reference)
        maps = [reference, aligned, error]
        titles = ['Original height', 'Estimated height (offset aligned)', 'Absolute height error']
        rmse = np.sqrt(np.mean(error[mask]**2))
        note = f'Independent color scales: compare the numbers. 2-pixel rim excluded; offset aligned; RMSE = {rmse:.6g}'
        print(f'{folder.parent.name}: height RMSE = {rmse:.8f} (frame width = {image_width}, Blender units)')
    else:
        mask = np.isfinite(heights)
        maps, titles = [heights], ['Estimated height']
        note = 'No reference available'

    fig, axes = plt.subplots(1, len(maps), figsize=(5*len(maps), 4.8), squeeze=False)
    for index, (ax, values, title) in enumerate(zip(axes.flat, maps, titles)):
        is_error = index == 2

        vmin = 0 if is_error else float(values[mask].min())
        vmax = float(values[mask].max())
        plot = ax.imshow(np.where(mask, values, np.nan), origin='upper',
                         interpolation='nearest', cmap='magma' if is_error else 'terrain',
                         vmin=vmin, vmax=max(vmax, vmin+1e-12))
        ax.set_title(f'{title}\nrange: {vmin:.4f} to {vmax:.4f}', fontsize=11)
        ax.set_axis_off()
        fig.colorbar(plot, ax=ax, shrink=0.75, label='Blender units')
    fig.suptitle(f'{folder.parent.name} / {folder.name} | frame width = {image_width:g}')
    fig.text(0.5, 0.02, note, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(folder / 'height_comparison.png', dpi=140)
    
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    analyze_height(PROJECT / 'data' / DATASET, show=INTERACTIVE)
