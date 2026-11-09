from pathlib import Path
import numpy as np
import OpenEXR

DATASET = 'simple/pyramid/baseline'
ROW = 128
COLUMN = 170
FOLDER = Path(__file__).resolve().parents[2] / 'data' / DATASET

# known directions of light
L = np.loadtxt(FOLDER/'lights.txt', ndmin=2)

# intinsity of pixel in each image
intensities = []
for index in range(len(L)):
    path = FOLDER/'images'/f'light_{index:02d}.exr'
    with OpenEXR.File(str(path), separate_channels=True) as image:
        red = float(image.channels()['R'].pixels[ROW, COLUMN])
        green = float(image.channels()['G'].pixels[ROW, COLUMN])
        blue = float(image.channels()['B'].pixels[ROW, COLUMN])
    intensities.append((red + green + blue) / 3) #avg
I = np.array(intensities)
if np.any(I <= 0):
    raise ValueError('shadow on all images')

# ls
g = np.linalg.lstsq(L, I, rcond=None)[0]
normal = g / np.linalg.norm(g)
np.set_printoptions(precision=6, suppress=True)
print('experiment:', DATASET, '| pixel:', ROW, COLUMN)
print('light directions L:\n', L)
print('intinsity I:', I)
print('estimated intensities:', L @ g)
print('solution g:', g)
print('calculated normal:', normal)

# check
true_normal = np.load(FOLDER/'reference_normals.npy')[ROW, COLUMN].astype(float)
true_normal /= np.linalg.norm(true_normal)
error = np.degrees(np.arccos(np.clip(normal @ true_normal, -1, 1)))
print('Poznata normala:', true_normal)
print('Greska:', round(float(error), 6), 'deg')
