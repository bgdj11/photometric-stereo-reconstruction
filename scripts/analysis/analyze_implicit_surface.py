from pathlib import Path
import ast
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.algorithms.implicit_surface import fit_surface
from scripts.algorithms import ls, irls
from scripts.algorithms.integrate_normals import reconstruct_height
from scripts.analysis.estimate_normals import load_images, plot_results

DATASET = 'complex/bump/spot_baseline'
STEPS = 600
LEARNING_RATE = .002
SEED = 7
DEVICE = 'auto'
PRECISION = 'float32'


def main(dataset=DATASET):

    folder = PROJECT/'data'/dataset
    output = folder/'mlp_b1'
    if not (folder/'spot.txt').exists():
        raise FileNotFoundError(f'Prvo renderuj jedan Spot primer: {folder}')
    settings = dict(line.split(': ', 1) for line in (folder/'settings.txt').read_text().splitlines() if ': ' in line)
    width = float(settings['Ortho scale'])
    camera = np.array(ast.literal_eval(settings['Camera matrix']))
    if not np.allclose(camera[:3, :3], np.eye(3)) or not np.allclose(camera[:2, 3], 0):
        raise ValueError('B1 ocekuje centriranu, poravnatu ortografsku kameru odozgo.')
    positions = np.loadtxt(folder/'light_positions.txt', ndmin=2)
    spot = np.loadtxt(folder/'spot.txt', ndmin=2)
    images = load_images(folder, len(positions))

    mask = binary_erosion(images.max(axis=0) > images.max()*1e-4,
                          structure=np.ones((3, 3)), iterations=2)
    start = time.perf_counter()
    result = fit_surface(images, positions, spot, mask, width, STEPS, LEARNING_RATE, SEED,
                         device=DEVICE, precision=PRECISION)
    seconds = time.perf_counter()-start

    
    truth = np.load(folder/'reference_normals.npy')
    original = np.load(folder/'surface_points.npy')[..., 2]
    evaluate = mask & np.isfinite(original) & (np.linalg.norm(truth, axis=-1) > 0)
    normals, height = result['normals'], result['height']
    error = ls.angular_error(normals, truth)
    height_offset = float(np.mean(original[evaluate]-height[evaluate]))
    aligned = height+height_offset
    height_error = np.abs(aligned-original)
    rows = []

    for name, method in [('LS', ls), ('IRLS', irls)]:
        estimated, _, _, valid = method.estimate_normals(images, np.loadtxt(folder/'lights.txt'))
        usable = evaluate & valid & (estimated[..., 2] > 1e-6)
        estimated[~usable] = np.nan
        baseline_height = reconstruct_height(estimated, width)
        difference = baseline_height[usable]-original[usable]
        difference -= difference.mean()
        rows.append((name, float(ls.angular_error(estimated, truth)[usable].mean()),
                     float(np.sqrt(np.mean(difference**2))), int(usable.sum())))
    rows.append(('MLP B1', float(error[evaluate].mean()),
                 float(np.sqrt(np.mean(height_error[evaluate]**2))), int(evaluate.sum())))

    output.mkdir(exist_ok=True)
    np.save(output/'estimated_normals.npy', np.where(mask[..., None], normals, np.nan))
    np.save(output/'estimated_height.npy', np.where(mask, height, np.nan))
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in result['model'].state_dict().items()},
                'frame_width': width, 'device': result['device'], 'precision': result['precision'],
                'steps': STEPS, 'seed': SEED, 'image_mse_history': result['history'].tolist()}, output/'model.pt')
    plot_results(output, images, truth, normals, evaluate, error, method='MLP B1 (known lights)')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    low = min(float(original[evaluate].min()), float(aligned[evaluate].min()))
    high = max(float(original[evaluate].max()), float(aligned[evaluate].max()))

    for index, (values, title) in enumerate(zip([original, aligned, height_error],
            ['Original height (evaluation only)', 'MLP height (offset aligned)', 'Absolute height error'])):
        plot = axes[index].imshow(np.where(evaluate, values, np.nan), interpolation='nearest',
                                 cmap='magma' if index == 2 else 'terrain',
                                 vmin=0 if index == 2 else low,
                                 vmax=None if index == 2 else high)
        axes[index].set_title(title)
        axes[index].axis('off')
        fig.colorbar(plot, ax=axes[index], shrink=.75, label='Blender units')
        
    fig.suptitle(f'{dataset} | B1: known Spot lamps, unknown surface')
    fig.text(.5, .015, 'Original and estimate share a color scale. Only Z offset aligned for evaluation.', ha='center')
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output/'height_comparison.png', dpi=130)
    plt.close(fig)

    lines = [f'Dataset: {dataset}', f'Resolution: {images.shape[2]} x {images.shape[1]}; images: {len(images)}',
             f'Known: camera, frame width {width}, lamp positions, axes, cone profiles, equal relative powers.',
             'Unknown: height represented by MLP; per-pixel reflectance fitted from all images.',
             'No true heights or normals in training. Flat initialization. No pretrained network.',
             f'MLP: 2 -> 48 -> 48 -> 1, Tanh; steps={STEPS}; lr={LEARNING_RATE}; seed={SEED}',
             f'Training seconds: {seconds:.2f}',
             f'Device: {result["device"]}; precision: {result["precision"]}; torch: {torch.__version__}',
             f'Initial image MSE: {result["history"][0]:.10g}',
             f'Best image MSE: {result["history"].min():.10g}',
             f'MLP normal error p95: {np.percentile(error[evaluate], 95):.6f} deg',
             f'MLP raw height RMSE (no alignment): {np.sqrt(np.mean((height[evaluate]-original[evaluate])**2)):.8f}',
             f'Height offset added ONLY for evaluation: {height_offset:.8f}',
             f'Evaluation pixels available: {evaluate.sum()}',
             '', 'Method | mean angular error (deg) | offset-aligned height RMSE (Blender units) | pixels']
    lines.extend(f'{name} | {angle:.6f} | {rmse:.8f} | {pixels}' for name, angle, rmse, pixels in rows)
    lines += ['', 'LS/IRLS use approximate center directions, not the calibrated local Spot model.',
              'This compares full pipelines; it does not prove MLP is better than classical near-light PS.',
              'B2 (unknown lamp positions/powers) has NOT been implemented here.',
              '', 'Files: images.png=input preview; normals.png=normal comparison;',
              'height_comparison.png=height comparison; estimated_*.npy=numeric predictions;',
              'model.pt=trained MLP weights and loss history; results.txt=this report.']
    report = '\n'.join(lines)+'\n'
    (output/'results.txt').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
