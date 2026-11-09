from pathlib import Path
import math
import bpy
import numpy as np

# set scene u want
SCENES = ['simple/plane', 'simple/tilted', 'simple/pyramid',
          'complex/bump', 'complex/terrain', 'complex/waves']
SIZE = 2.0
PYRAMID_HEIGHT = 0.45
BUMP_HEIGHT = 0.18
BUMP_WIDTH = 0.35
GRID_CELLS = 128
CAMERA_POSITION = (0, 0, 4)
CAMERA_ROTATION = (0, 0, 0)
CAMERA_SCALE = 2.4
DATA = Path(__file__).resolve().parents[2] / 'data'


def height_at(kind, x, y):
    if kind == 'bump':
        return BUMP_HEIGHT * np.exp(-(x*x + y*y) / (2*BUMP_WIDTH**2))
    if kind == 'waves':
        return 0.05*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) + 0.03*np.sin(3*y)
    
    # center x,y height, width 
    hills = [(-0.65, -0.45, .12, .20), (-0.15, -0.55, .15, .24),
             (0.50, -0.55, .11, .20), (-0.50, 0.25, .18, .24),
             (0.15, 0.20, .16, .25), (0.65, 0.55, .13, .20),
             (0.40, -0.05, -.07, .16), (-0.10, 0.65, -.035, .22)]
    z = 0.03
    for cx, cy, height, width in hills:
        z += height * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*width**2))
    return z + 0.012*np.sin(14*x + 3*y)*np.cos(12*y - 2*x)


def geometry(kind):
    half = SIZE / 2
    corners = [(-half, -half, 0), (half, -half, 0),
               (half, half, 0), (-half, half, 0)]
    if kind in ('plane', 'tilted'):
        return corners, [(0, 1, 2, 3)]
    if kind == 'pyramid':
        return corners + [(0, 0, PYRAMID_HEIGHT)], [
            (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (3, 2, 1, 0)]
    if kind not in ('bump', 'terrain', 'waves'):
        raise ValueError('Nepoznat oblik: ' + kind)
    vertices, faces = [], []
    axis = np.linspace(-half, half, GRID_CELLS + 1)
    for y in axis:
        for x in axis:
            vertices.append((float(x), float(y), float(height_at(kind, x, y))))
    for row in range(GRID_CELLS):
        for col in range(GRID_CELLS):
            a = row*(GRID_CELLS + 1) + col
            b, c, d = a+1, a+GRID_CELLS+2, a+GRID_CELLS+1
            faces.extend([(a, b, c), (a, c, d)])
    return vertices, faces


def create_scene(name):
    destination = DATA / name / 'scene.blend'
    if destination.exists():
        print('already exists ', destination)
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.file_preview_type = 'NONE'
    scene = bpy.context.scene

    # surface
    kind = name.split('/')[-1]
    vertices, faces = geometry(kind)
    mesh = bpy.data.meshes.new(kind)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    surface = bpy.data.objects.new('Surface', mesh)
    scene.collection.objects.link(surface)
    if kind == 'tilted':
        surface.rotation_euler.y = math.radians(15)

    # material
    material = bpy.data.materials.new('Gray diffuse')
    material.use_nodes = True
    material.node_tree.nodes.clear()
    diffuse = material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)
    diffuse.inputs['Roughness'].default_value = 0
    output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
    material.node_tree.links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    surface.data.materials.append(material)

    # camera
    camera = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    scene.collection.objects.link(camera)
    camera.location = CAMERA_POSITION
    camera.rotation_euler = [math.radians(angle) for angle in CAMERA_ROTATION]
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = CAMERA_SCALE
    scene.camera = camera

    # sun
    sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN'))
    scene.collection.objects.link(sun)
    sun.data.energy, sun.data.angle = 2, 0
    sun.rotation_euler.y = math.radians(40)
    scene.world = bpy.data.worlds.new('Black world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0

    # render
    scene.render.engine = 'CYCLES'
    scene.cycles.samples, scene.cycles.diffuse_bounces = 16, 0
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'Standard'
    scene.render.filepath = '//manual.png'
    surface.select_set(True)
    bpy.context.view_layer.objects.active = surface
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))


if __name__ == '__main__':
    for name in SCENES:
        create_scene(name)
