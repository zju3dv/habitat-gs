# !/usr/bin/env python

"""
Script to convert all double precision position components to float32 in the PLY file in order to match the format requirements.
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement

def downgrade_dtype(props, target=np.float32):
    arr = np.column_stack([props[name] for name in props.dtype.names])
    arr = arr.astype(target)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PLY file path")
    parser.add_argument("--output", required=True, help="Output PLY file path")
    args = parser.parse_args()

    ply = PlyData.read(args.input)

    new_elements = []
    for el in ply.elements:
        if el.name == 'vertex':
            new_props = {}
            for name in el.data.dtype.names:
                dt = el.data.dtype[name]
                # Only convert double precision position components to float32, other components (e.g., uint8 color, faces) remain unchanged.
                if dt.kind == 'f' and dt.itemsize == 8:
                    new_props[name] = el.data[name].astype(np.float32)
                else:
                    new_props[name] = el.data[name]
            vertex_dtype = [(name, new_props[name].dtype) for name in el.data.dtype.names]
            vertex_arr = np.empty(len(el.data), dtype=vertex_dtype)
            for name in vertex_arr.dtype.names:
                vertex_arr[name] = new_props[name]
            new_el = PlyElement.describe(vertex_arr, 'vertex')
            new_elements.append(new_el)
        else:
            new_elements.append(el)

    new_ply = PlyData(new_elements, text=ply.text)
    new_ply.write(args.output)
    print(f"Wrote float32 ply: {args.output}")

if __name__ == "__main__":
    main()