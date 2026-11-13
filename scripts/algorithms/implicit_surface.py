import numpy as np
import torch
from torch import nn


class HeightMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 48), nn.Tanh(),
                                    nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))
        
        # start with flat surface
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, xy):
        return self.layers(xy).squeeze(-1)

# add +.5 for pixel center
def camera_xy(shape, width):
    rows, cols = np.indices(shape)
    step = width/shape[1]
    return np.stack([(cols+.5-shape[1]/2)*step,
                     (shape[0]/2-rows-.5)*step], axis=-1)


def surface_normals(height, step):
    # -dx is because x axis goes right, but normal points left for positive slope.
    dy, dx = torch.gradient(height, spacing=step)
    normals = torch.stack([-dx, dy, torch.ones_like(height)], dim=-1)
    return normals/torch.linalg.vector_norm(normals, dim=-1, keepdim=True)


def spot_shading(height, normals, xy, positions, spot):

    # each point own direction to each lamp
    points = torch.cat([xy, height[..., None]], dim=-1)
    vectors = positions-points[..., None, :]
    distance = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1e-8)
    directions = vectors/distance[..., None]
    incidence = (directions*normals[..., None, :]).sum(dim=-1).clamp_min(0)

    # smoothing like ideal blender spotlight
    cosine = (-directions*spot[:, :3]).sum(dim=-1)
    edge = torch.cos(torch.deg2rad(spot[:, 3]/2))
    t = ((cosine-edge)/((1-edge)*spot[:, 4]).clamp_min(1e-12)).clamp(0, 1)
    beam = t*t*(3-2*t)
    return incidence*beam/distance**2


def fit_surface(images, positions, spot, mask, width=2.4, steps=600, learning_rate=.002, seed=7,
                device='auto', precision='float32'):

   
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)

    dtype = torch.float32 if precision == 'float32' else torch.float64
    print(f'Device: {device}; precision: {precision}', flush=True)


    xy = torch.as_tensor(camera_xy(mask.shape, width), dtype=dtype, device=device)
    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    spot = torch.as_tensor(spot, dtype=dtype, device=device)
    valid = torch.as_tensor(mask, dtype=torch.bool, device=device)

    observed = torch.as_tensor(images.transpose(1, 2, 0), dtype=dtype, device=device)
    observed = observed/observed[valid].max().clamp_min(1e-12)

    network = HeightMLP().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    history, best_loss, best_state = [], float('inf'), None

    def predict():
        height = network(xy)
        normals = surface_normals(height, width/mask.shape[1])
        shading = spot_shading(height, normals, xy, positions, spot)

        # I_j = reflectance * shading_j.
        reflectance = (shading*observed).sum(dim=-1)/(shading**2).sum(dim=-1).clamp_min(1e-12)
        predicted = shading*reflectance[..., None]
        return height, normals, reflectance, predicted

    for iteration in range(steps):
        optimizer.zero_grad()
        height, normals, reflectance, predicted = predict()
        loss = ((predicted-observed)[valid]**2).mean()

        value = float(loss.detach())
        history.append(value)

        if value < best_loss:
            best_loss = value
            best_state = {key: weight.detach().clone() for key, weight in network.state_dict().items()}

        loss.backward()
        optimizer.step()
        if iteration % 100 == 0 or iteration == steps-1:
            print(f'B1 step {iteration+1}/{steps}: image MSE={value:.8g}', flush=True)

    network.load_state_dict(best_state)
    with torch.no_grad():
        height, normals, reflectance, predicted = predict()

    return dict(height=height.cpu().numpy(), normals=normals.cpu().numpy(),
                reflectance=reflectance.cpu().numpy(), predicted=predicted.cpu().numpy(),
                history=np.array(history), model=network, device=str(device), precision=precision)
