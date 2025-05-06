import numpy as np
import bpy
import os

scene_name = "my_scene"
path_pc = "scene_colors.txt"

# Delete existing object.
bpy.ops.object.select_all(action='DESELECT')
bpy.context.collection.objects['scene0081_00'].select_set(True)
bpy.ops.object.delete() 

# Load new scene
scene = np.loadtxt(path_pc)
pts = scene[:, :3]
pts = pts - np.mean(pts, 0)
colors = scene[:, 3:]
mesh = bpy.data.meshes.new(name=scene_name)
mesh.vertices.add(pts.shape[0])
mesh.vertices.foreach_set("co", [a for v in pts for a in (v[0], v[1], v[2])])

# Create our new object here
for ob in bpy.context.selected_objects:
    ob.select_set(False)
obj = bpy.data.objects.new(scene_name, mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Create new Attribute 'Col' to hold the color data
bpy.context.active_object.data.attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
newcolor = bpy.context.active_object.data
for i, col in enumerate(colors):
    newcolor.attributes['Col'].data[i].color[0] = col[0]
    newcolor.attributes['Col'].data[i].color[1] = col[1]
    newcolor.attributes['Col'].data[i].color[2] = col[2]
    newcolor.attributes['Col'].data[i].color[3] = 1
mesh.update()
mesh.validate()

# Set material.
bpy.context.active_object.data.materials.append(bpy.data.materials.get("Mat_feats"))
bpy.context.active_object.active_material_index = len(obj.data.materials) - 1

# Set geometric node.
modifier = bpy.context.active_object.modifiers.new("GN", "NODES")
replacement = bpy.data.node_groups["Points_feats"]
modifier.node_group = replacement