import sys
import argparse
from pathlib import Path

import bpy
import numpy as np


def parse_blender_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    print("SCRIPT_ARGV:", argv)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(fbx_path):
    fbx_path = Path(fbx_path).resolve()
    if not fbx_path.exists():
        raise FileNotFoundError(f"FBX not found: {fbx_path}")

    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))


def find_main_mesh():
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if not meshes:
        raise RuntimeError(
            "No mesh found in FBX. For Mixamo, download with Skin = With Skin."
        )

    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)

    print("Found mesh objects:")
    for obj in meshes:
        print(f"  {obj.name}: {len(obj.data.vertices)} vertices")

    return meshes[0]


def find_armature():
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if arms:
        print("Found armature:", arms[0].name)
    else:
        print("No armature found.")
    return arms[0] if arms else None


def get_frame_range(start_arg, end_arg):
    scene = bpy.context.scene

    start_frame = int(scene.frame_start)
    end_frame = int(scene.frame_end)

    if start_arg is not None:
        start_frame = int(start_arg)

    if end_arg is not None:
        end_frame = int(end_arg)

    if end_frame < start_frame:
        raise ValueError(f"Invalid frame range: {start_frame} -> {end_frame}")

    return start_frame, end_frame


def extract_faces_and_uvs(mesh_obj):
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()

    faces = []
    face_uvs = []

    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None

    for tri in mesh.loop_triangles:
        face = []
        tri_uv = []

        for loop_idx in tri.loops:
            vertex_idx = mesh.loops[loop_idx].vertex_index
            face.append(vertex_idx)

            if uv_layer is not None:
                uv = uv_layer[loop_idx].uv
                tri_uv.append([uv.x, uv.y])
            else:
                tri_uv.append([0.0, 0.0])

        faces.append(face)
        face_uvs.append(tri_uv)

    return (
        np.asarray(faces, dtype=np.int32),
        np.asarray(face_uvs, dtype=np.float32),
    )


def get_texture_paths(mesh_obj):
    texture_paths = []

    for mat_slot in mesh_obj.material_slots:
        mat = mat_slot.material

        if mat is None:
            continue

        if mat.node_tree is None:
            continue

        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                image_path = bpy.path.abspath(node.image.filepath)
                texture_paths.append(image_path)

    texture_paths = list(dict.fromkeys(texture_paths))

    print("Texture paths:")
    if texture_paths:
        for p in texture_paths:
            print(" ", p)
    else:
        print("  None found")

    return texture_paths


def bake_vertices_and_normals(mesh_obj, start_frame, end_frame):
    vertices_all = []
    normals_all = []

    for frame in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = mesh_obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()

        verts = np.asarray([v.co[:] for v in mesh_eval.vertices], dtype=np.float32)
        normals = np.asarray([v.normal[:] for v in mesh_eval.vertices], dtype=np.float32)

        vertices_all.append(verts)
        normals_all.append(normals)

        obj_eval.to_mesh_clear()

        if frame % 10 == 0:
            print(f"Baked frame {frame}/{end_frame}")

    vertices_all = np.stack(vertices_all, axis=0)
    normals_all = np.stack(normals_all, axis=0)

    return vertices_all, normals_all


def compute_bbox(vertices):
    bbox_min = vertices.reshape(-1, 3).min(axis=0).astype(np.float32)
    bbox_max = vertices.reshape(-1, 3).max(axis=0).astype(np.float32)
    return bbox_min, bbox_max


def main():
    args = parse_blender_args()

    clear_scene()
    import_fbx(args.fbx)

    bpy.context.scene.render.fps = args.fps

    mesh_obj = find_main_mesh()
    find_armature()

    start_frame, end_frame = get_frame_range(args.start, args.end)

    print("")
    print("Bake settings:")
    print("  mesh:", mesh_obj.name)
    print("  fps:", args.fps)
    print(f"  frames: {start_frame} -> {end_frame}")
    print("")

    faces, face_uvs = extract_faces_and_uvs(mesh_obj)
    texture_paths = get_texture_paths(mesh_obj)

    vertices, normals = bake_vertices_and_normals(
        mesh_obj=mesh_obj,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    bbox_min, bbox_max = compute_bbox(vertices)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        vertices=vertices,
        normals=normals,
        faces=faces,
        face_uvs=face_uvs,
        fps=np.array([args.fps], dtype=np.int32),
        start_frame=np.array([start_frame], dtype=np.int32),
        end_frame=np.array([end_frame], dtype=np.int32),
        mesh_name=np.array([mesh_obj.name]),
        texture_paths=np.array(texture_paths),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    print("")
    print("Saved raw baked mesh:")
    print("  path:", out_path)
    print("  vertices:", vertices.shape)
    print("  normals:", normals.shape)
    print("  faces:", faces.shape)
    print("  face_uvs:", face_uvs.shape)
    print("  bbox_min:", bbox_min)
    print("  bbox_max:", bbox_max)


if __name__ == "__main__":
    main()
