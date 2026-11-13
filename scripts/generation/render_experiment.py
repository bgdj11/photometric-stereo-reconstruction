from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SCENES = ['complex/terrain', 'simple/pyramid']

EXPERIMENT = 'spot_baseline'
LIGHT_TYPE = 'SPOT'  # 'SUN', 'POINT' or 'SPOT'
LIGHT_COUNT = 6
TILT_DEGREES = 40
CUSTOM_ANGLES = [] # option (az, nagib)
SUN_STRENGTH = 2.0

# additional params
LAMP_POWER = 100.0  # W, only  POINT/SPOT sun is different
LIGHT_DISTANCE = 2.0  # distance from origin to light, only POINT/SPOT
SPOT_ANGLE = 90.0  # angle of cone for SPOT
SPOT_BLEND = 0.7  # blend factor for SPOT, 0=hard edge, 1=soft edge
RESOLUTION = 128
SAMPLES = 16
DATA = Path(__file__).resolve().parents[2] / 'data'


def reference_surface(scene, surface):
    evaluated = surface.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    if any(face.use_smooth for face in mesh.polygons):
        evaluated.to_mesh_clear()
        raise ValueError('use  Shade Flat, not Shade Smooth.')
    vertices = [evaluated.matrix_world @ v.co for v in mesh.vertices]
    faces = [list(face.vertices) for face in mesh.polygons]
    tree = BVHTree.FromPolygons(vertices, faces)
    evaluated.to_mesh_clear()
    camera = scene.camera
    camera_matrix = camera.matrix_world.normalized()
    direction = (camera_matrix.to_3x3() @ Vector((0, 0, -1))).normalized()


    frame = camera.data.view_frame(scene=scene)
    left, right = min(v.x for v in frame), max(v.x for v in frame)
    bottom, top = min(v.y for v in frame), max(v.y for v in frame)
    normals = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.float32)
    points = np.full((RESOLUTION, RESOLUTION, 3), np.nan, dtype=np.float32)
    for row in range(RESOLUTION):
        y = top - (row+0.5)/RESOLUTION * (top-bottom)
        for col in range(RESOLUTION):
            x = left + (col+0.5)/RESOLUTION * (right-left)
            origin = camera_matrix @ Vector((x, y, -camera.data.clip_start))
            hit, normal, _, _ = tree.ray_cast(origin, direction, camera.data.clip_end-camera.data.clip_start)
            if hit is not None:
                normals[row, col] = normal
                points[row, col] = hit
    return normals, points 


def render_experiment(name):
    source = DATA / name / 'scene.blend'
    folder = DATA / name / EXPERIMENT
    if folder.exists():
        raise FileExistsError(f'{folder} already exist')
    bpy.ops.wm.open_mainfile(filepath=str(source))

    scene = bpy.context.scene
    surfaces = [obj for obj in scene.objects if obj.type == 'MESH' and not obj.hide_render]
    lights = [obj for obj in scene.objects if obj.type == 'LIGHT' and not obj.hide_render]

    lamp = lights[0]
    angles = CUSTOM_ANGLES or [(float(a), TILT_DEGREES)
                              for a in np.linspace(0, 360, LIGHT_COUNT, endpoint=False)]
    directions = []
    for azimuth, tilt in angles:
        a, t = math.radians(azimuth), math.radians(tilt)
        directions.append([math.sin(t)*math.cos(a), math.sin(t)*math.sin(a), math.cos(t)])
    L = np.array(directions)
    if len(L) < 3 or np.linalg.matrix_rank(L) < 3:
        raise ValueError('at least 3 lights')

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples, scene.cycles.seed = SAMPLES, 42
    scene.cycles.use_denoising = scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces, scene.cycles.diffuse_bounces = 1, 0
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
    scene.render.resolution_x = scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.use_border = scene.render.use_motion_blur = False
    scene.render.use_compositing = scene.render.use_sequencer = False
    scene.camera.data.dof.use_dof = False

    # image format size ...
    scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.color_mode = 'RGB'
    lamp.data.type = LIGHT_TYPE
    lamp.rotation_mode = 'XYZ'
    if LIGHT_TYPE == 'SUN':
        lamp.data.energy, lamp.data.angle = SUN_STRENGTH, 0
    else:
        lamp.data.energy = LAMP_POWER
        lamp.data.shadow_soft_size = 0
        if LIGHT_TYPE == 'SPOT':
            lamp.data.spot_size = math.radians(SPOT_ANGLE)
            lamp.data.spot_blend = SPOT_BLEND
    bpy.context.view_layer.update()
    truth, points = reference_surface(scene, surfaces[0])
    (folder / 'images').mkdir(parents=True)

  
    for index, (azimuth, tilt) in enumerate(angles):
        if LIGHT_TYPE == 'SUN':
            lamp.rotation_euler = (0, math.radians(tilt), math.radians(azimuth))
        else:
            lamp.location = (LIGHT_DISTANCE * L[index]).tolist()

            # for POINT/SPOT, rotate lamp to point at world origin
            lamp.rotation_euler = Vector((-L[index]).tolist()).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()

        scene.render.filepath = str(folder / 'images' / f'light_{index:02d}.exr')
        bpy.ops.render.render(write_still=True)


    # save
    description = ('SUN directions, same at every point' if LIGHT_TYPE == 'SUN' else
                   'APPROXIMATE directions at world origin; LS/IRLS ignore local variation')
    np.savetxt(folder/'lights.txt', L, fmt='%.10f', header='lx ly lz; surface -> source; '+description)
    if LIGHT_TYPE != 'SUN':
        np.savetxt(folder/'light_positions.txt', LIGHT_DISTANCE*L, header='world X Y Z; one row per image')
    if LIGHT_TYPE == 'SPOT':
        np.savetxt(folder/'spot.txt', np.column_stack([
            -L, np.full(len(L), SPOT_ANGLE), np.full(len(L), SPOT_BLEND)]),
            header='axis X Y Z (lamp -> scene), full cone angle degrees, blend')
    np.save(folder/'reference_normals.npy', truth)
    np.save(folder/'surface_points.npy', points)
    settings = (f'Scene: {name}/scene.blend\nBlender: {bpy.app.version_string}\n'
                f'Images: {len(L)}\nResolution: {RESOLUTION} x {RESOLUTION}\nSamples: {SAMPLES}\n'
                f'Light type: {LIGHT_TYPE}\n'
                f'Strength: {SUN_STRENGTH if LIGHT_TYPE == "SUN" else LAMP_POWER}; '
                f'units: {"Sun strength" if LIGHT_TYPE == "SUN" else "W"}\n'
                f'Angles (azimuth, tilt): {angles}\n'
                f'Camera matrix: {[list(row) for row in scene.camera.matrix_world]}\n'
                f'Ortho scale: {scene.camera.data.ortho_scale}\n'
                'Images: scene-linear float32 EXR; fixed camera; world XYZ normals.\n'
                'Diffuse bounces: 0; denoising: off; seed: 42.\n')
    
    if LIGHT_TYPE != 'SUN':
        settings += f'Distance from origin: {LIGHT_DISTANCE}; source radius: 0\n{description}\n'
    if LIGHT_TYPE == 'SPOT':
        settings += f'Spot angle: {SPOT_ANGLE}; blend: {SPOT_BLEND}; aimed at world origin\n'
    (folder/'settings.txt').write_text(settings, encoding='utf-8')
    print('saved:', folder)


if __name__ == '__main__':
    for name in SCENES:
        render_experiment(name)
