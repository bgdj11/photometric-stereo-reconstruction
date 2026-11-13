import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


# p = -nx/nz
# dh = p*dx
# dij = ((pij + pij+1) / 2) * dx --> for each pair of neighb pixels calc 
# hij+1 - hij = dij
# minimize E = sum of this above ofc x and y
# Ah = d
# h* = argmin ||Ah - d||^2


def reconstruct_height(normals, image_width):

    normals = np.asarray(normals, dtype=float)

    mask = np.isfinite(normals).all(axis=-1) & (normals[..., 2] > 1e-6)

    p, q = np.zeros(mask.shape), np.zeros(mask.shape)
    p[mask] = -normals[..., 0][mask] / normals[..., 2][mask]
    q[mask] = -normals[..., 1][mask] / normals[..., 2][mask]
    step = image_width / normals.shape[1]

    ids = np.full(mask.shape, -1, dtype=int)
    ids[mask] = np.arange(mask.sum())
    horizontal = mask[:, :-1] & mask[:, 1:]
    vertical = mask[:-1, :] & mask[1:, :]
    first = np.concatenate([ids[:, :-1][horizontal], ids[:-1, :][vertical]])
    second = np.concatenate([ids[:, 1:][horizontal], ids[1:, :][vertical]])


    differences = np.concatenate([
        (0.5 * (p[:, :-1] + p[:, 1:]) * step)[horizontal],
        (-0.5 * (q[:-1, :] + q[1:, :]) * step)[vertical],
    ])
    equations = np.arange(len(differences))

    A = coo_matrix((np.concatenate([-np.ones(len(first)), np.ones(len(second))]),
                    (np.concatenate([equations, equations]),
                     np.concatenate([first, second]))),
                   shape=(len(differences), int(mask.sum()))).tocsr()

    result = lsmr(A, differences, atol=1e-8, btol=1e-8, maxiter=5000)
    
    heights = np.full(mask.shape, np.nan)

    heights[mask] = result[0] - result[0].mean()

    return heights

