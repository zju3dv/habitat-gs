import numpy as np
from plyfile import PlyData

gaussian_path = "/mnt/data/home/ziyuan/gaussian-splatting/output/d6858747-c/point_cloud/iteration_30000/point_cloud.ply"

try:
    # 读取 PLY 文件
    plydata = PlyData.read(gaussian_path)

    # 'plydata' 包含了文件中的所有元素，我们关心的是 'vertex' (顶点)
    vertex_element = plydata['vertex']

    count = 0
    for vertex in vertex_element.data:
        print(f"x: {vertex['x']}, y: {vertex['y']}, z: {vertex['z']}, nx: {vertex['nx']}, ny: {vertex['ny']}, nz: {vertex['nz']}, f_dc_0: {vertex['f_dc_0']}, f_dc_1: {vertex['f_dc_1']}, f_dc_2: {vertex['f_dc_2']}, f_rest_0: {vertex['f_rest_0']}, f_rest_1: {vertex['f_rest_1']}, f_rest_2: {vertex['f_rest_2']}, opacity: {vertex['opacity']}, scale_0: {vertex['scale_0']}, scale_1: {vertex['scale_1']}, scale_2: {vertex['scale_2']}, rot_0: {vertex['rot_0']}, rot_1: {vertex['rot_1']}, rot_2: {vertex['rot_2']}, rot_3: {vertex['rot_3']}")
        if count > 10:
            break
        count += 1

except Exception as e:
    print(f"读取文件时出错: {e}")