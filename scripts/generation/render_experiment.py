from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SCENES = ['simple/plane', 'simple/tilted', 'simple/pyramid',
          'complex/bump', 'complex/terrain', 'complex/waves']
EXPERIMENT = 'baseline' # just name
LIGHT_COUNT = 6
TILT_DEGREES = 40
CUSTOM_ANGLES = [] # option (az, nagib)
SUN_STRENGTH = 2.0
RESOLUTION = 256
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

    sun = lights[0]
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
    sun.data.energy, sun.data.angle = SUN_STRENGTH, 0
    sun.rotation_mode = 'XYZ'
    bpy.context.view_layer.update()
    truth, points = reference_surface(scene, surfaces[0])
    (folder / 'images').mkdir(parents=True)

  
    for index, (azimuth, tilt) in enumerate(angles):
        sun.rotation_euler = (0, math.radians(tilt), math.radians(azimuth))
        bpy.context.view_layer.update()
        actual = np.array(sun.matrix_world.to_3x3() @ Vector((0, 0, 1)))

        scene.render.filepath = str(folder / 'images' / f'light_{index:02d}.exr')
        bpy.ops.render.render(write_still=True)


    # save
    np.savetxt(folder/'lights.txt', L, fmt='%.10f', header='lx ly lz; one row per image, surface -> source')
    np.save(folder/'reference_normals.npy', truth)
    np.save(folder/'surface_points.npy', points)
    settings = (f'Scene: {name}/scene.blend\nBlender: {bpy.app.version_string}\n'
                f'Images: {len(L)}\nResolution: {RESOLUTION} x {RESOLUTION}\nSamples: {SAMPLES}\n'
                f'Sun strength: {SUN_STRENGTH}\nAngles (azimuth, tilt): {angles}\n'
                f'Camera matrix: {[list(row) for row in scene.camera.matrix_world]}\n'
                f'Ortho scale: {scene.camera.data.ortho_scale}\n'
                'Images: scene-linear float32 EXR; fixed camera; world XYZ normals.\n'
                'Diffuse bounces: 0; denoising: off; seed: 42.\n')
    (folder/'settings.txt').write_text(settings, encoding='utf-8')
    print('saved:', folder)


if __name__ == '__main__':
    for name in SCENES:
        render_experiment(name)
