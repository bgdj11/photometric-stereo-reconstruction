import numpy as np

def estimate_normals(images, lights, iterations=10):

    images = np.asarray(images, dtype=np.float64)
    lights = np.asarray(lights, dtype=np.float64)   

    # each column holds the observations of ONE pixel.
    intensities = images.reshape(len(lights), -1)

    g = np.linalg.lstsq(lights, intensities, rcond=None)[0]

    # Huber weights
    for _ in range(iterations):
        error = np.abs(lights @ g - intensities)
        threshold = np.maximum(1.345 * np.median(error, axis=0) / 0.6745, 1e-6)
        weights = np.minimum(1, threshold / np.maximum(error, 1e-10))
        root = np.sqrt(weights.T)
        weighted_lights = root[..., None] * lights
        weighted_images = root * intensities.T
        g = (np.linalg.pinv(weighted_lights) @ weighted_images[..., None])[..., 0].T

    length = np.linalg.norm(g, axis=0)

    # ignore dark pixels
    valid = (length > 1e-10) & (intensities.max(axis=0) > 1e-6)
    normals = np.full_like(g, np.nan)
    normals[:, valid] = g[:, valid] / length[valid]

    residual = np.sqrt(np.mean((lights @ g - intensities) ** 2, axis=0))
    shape = images.shape[1:]

    return (normals.T.reshape(*shape, 3), length.reshape(shape),
            residual.reshape(shape), valid.reshape(shape))