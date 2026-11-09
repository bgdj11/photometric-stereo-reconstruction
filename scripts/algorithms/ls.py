import numpy as np

def estimate_normals(images, lights):

    images = np.asarray(images, dtype=np.float64)
    lights = np.asarray(lights, dtype=np.float64)


    # each column now holds the six observations of ONE pixel.
    intensities = images.reshape(len(lights), -1)

    #solve
    g = np.linalg.lstsq(lights, intensities, rcond=None)[0]

    length = np.linalg.norm(g, axis=0)
    valid = length > 1e-10
    normals = np.full_like(g, np.nan)
    normals[:, valid] = g[:, valid] / length[valid]

    residual = np.sqrt(np.mean((lights @ g - intensities) ** 2, axis=0))
    shape = images.shape[1:]

    return (normals.T.reshape(*shape, 3), length.reshape(shape),
            residual.reshape(shape), valid.reshape(shape))


def angular_error(estimated, truth):
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    lengths = np.linalg.norm(estimated, axis=-1) * np.linalg.norm(truth, axis=-1)
    dots = np.sum(estimated * truth, axis=-1) / np.maximum(lengths, 1e-15)
    return np.degrees(np.arccos(np.clip(dots, -1, 1)))
