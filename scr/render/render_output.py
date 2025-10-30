import bpy, math, mathutils

# === 可按需修改 ===
SCALE = 0.001                 # 你导入等值线时用的 scale
R_E   = 6371.0                # 地球半径 km
H     = 10.0                  # 等值线所在高度 km
CURVE_THICK = 0.02 * SCALE    # 曲线厚度（约 20 m）
OUT_PATH = "/media/user/Elements/Ye_Liu/software/RadiantNeF4HAPS/out/contours_preview.png"

# 1) 让所有 iso_* 曲线可渲染：加厚 + 发光材质
for o in bpy.data.objects:
    if o.type == 'CURVE' and o.name.startswith("iso_"):
        o.data.bevel_depth = CURVE_THICK
        # 给曲线统一使用发光材质（若已存在材质则复用其颜色）
        if o.data.materials:
            for m in o.data.materials:
                m.use_nodes = True
                nt = m.node_tree; nt.nodes.clear()
                out = nt.nodes.new("ShaderNodeOutputMaterial")
                emi = nt.nodes.new("ShaderNodeEmission")
                emi.inputs["Strength"].default_value = 3.0
                nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
        else:
            m = bpy.data.materials.new(name="IsoEmission"); m.use_nodes = True
            nt = m.node_tree; nt.nodes.clear()
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            emi = nt.nodes.new("ShaderNodeEmission")
            emi.inputs["Color"].default_value = (1.0, 0.9, 0.2, 1.0)  # 淡黄
            emi.inputs["Strength"].default_value = 3.0
            nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
            o.data.materials.append(m)

# 2) 地球球体（深色材质）
bpy.ops.mesh.primitive_uv_sphere_add(radius=(R_E+H)*SCALE, segments=64, ring_count=32, location=(0,0,0))
earth = bpy.context.active_object
earth.name = "Earth10km"
mat = bpy.data.materials.new("EarthDark"); mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.02, 0.03, 0.05, 1.0)
bsdf.inputs["Roughness"].default_value   = 0.8
earth.data.materials.append(mat)

# 3) 太阳光
bpy.ops.object.light_add(type='SUN', location=(0, 0, (R_E+H)*SCALE + 1.0))
sun = bpy.context.active_object
sun.data.energy = 3.0

# 4) 相机（拉远三倍半径，从南侧看向地心）
R = (R_E+H)*SCALE
bpy.ops.object.camera_add(location=(0, -3.2*R, 1.2*R))
cam = bpy.context.active_object
cam.data.lens = 60.0
direction = mathutils.Vector((0,0,0)) - cam.location
cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
bpy.context.scene.camera = cam

# 5) 渲染设置（Eevee 快速出图；若无 Eevee，可手动改为 'CYCLES'）
scene = bpy.context.scene
try:
    scene.render.engine = 'BLENDER_EEVEE'
except:
    scene.render.engine = 'CYCLES'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
vl = bpy.context.view_layer if bpy.context.view_layer else bpy.context.scene.view_layers[0]
# 如果你的 Blender 版本没有 AO 这个开关，也不会报错
if hasattr(vl, "use_pass_ambient_occlusion"):
    vl.use_pass_ambient_occlusion = True


# 世界背景稍微提亮
world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg: bg.inputs["Strength"].default_value = 0.5

# 6) 渲染并保存
scene.render.filepath = OUT_PATH
bpy.ops.render.render(write_still=True)
print("Saved:", OUT_PATH)
