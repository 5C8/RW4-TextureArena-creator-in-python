import struct
import os
import re

# Align constants with geo.py enums
class AptShapeType:
    GEOMETRY = 1
    TEXTURE = 2

class AptRenderType:
    LINE = 0
    SOLID = 1
    TEXTURE_CLAMPED = 2
    TEXTURE_WRAPPED = 3

def pack_apt_matrix(a=1.0, b=0.0, c=0.0, d=1.0, tx=0.0, ty=0.0):
    return struct.pack(">6f", a, b, c, d, tx, ty)

def parse_ru_to_shape(filepath):
    """Parses one .ru file into one AptShape structure."""
    if not os.path.exists(filepath):
        return None

    # Extract ID from filename (e.g., "15.ru" -> 15)
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    shape_id = int(match.group(1)) if match else 0

    shape_data = {
        "id": shape_id,
        "type": AptShapeType.GEOMETRY,
        "matrix": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        "elements": []
    }
    
    current_color = (1.0, 1.0, 1.0, 1.0)
    current_texture_id = 0
    current_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    current_render_type = AptRenderType.SOLID 

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cmd = parts[0]

            if cmd == 's':
                data = parts[1].split(':')
                prefix = data[0]
                current_color = tuple(float(x)/255.0 for x in data[1:5])
                
                if prefix == 's': 
                    current_render_type = AptRenderType.SOLID
                    shape_data["type"] = AptShapeType.GEOMETRY
                elif prefix == 'tc': 
                    current_render_type = AptRenderType.TEXTURE_CLAMPED
                    current_texture_id = int(data[5])
                    current_matrix = tuple(float(x) for x in data[6:])
                    shape_data["matrix"] = current_matrix
                    shape_data["type"] = AptShapeType.TEXTURE
                elif prefix == 'tw': 
                    current_render_type = AptRenderType.TEXTURE_WRAPPED
                    current_texture_id = int(data[5])
                    current_matrix = tuple(float(x) for x in data[6:])
                    shape_data["matrix"] = current_matrix
                    shape_data["type"] = AptShapeType.TEXTURE

            elif cmd in ('t', 'l'):
                active_type = AptRenderType.LINE if cmd == 'l' else current_render_type
                coords = [float(x) for x in parts[1].split(':')]
                
                elem = struct.pack(">I", active_type)           # 0x00: Type
                elem += struct.pack(">4f", *current_color)      # 0x04: RGBA
                elem += struct.pack(">I", current_texture_id)   # 0x14: ID
                elem += pack_apt_matrix(*current_matrix)        # 0x18: Matrix
                elem += struct.pack(">I", 1)                    # 0x30: NumPrims
                
                if cmd == 't':
                    elem += struct.pack(">6f", *coords)
                else:
                    elem += struct.pack(">4f", *coords) + (b'\x00' * 8)
                
                shape_data["elements"].append(elem)
                
    return shape_data if shape_data["elements"] else None

def build_combined_geo(ru_files, output_path):
    shapes = []
    for ru in ru_files:
        res = parse_ru_to_shape(ru)
        if res:
            shapes.append(res)

    with open(output_path, 'wb') as f:
        # 1. AptGeometryInfo Header
        # Write magic bytes 53 6B 38 00 (Sk8\0) instead of 0
        f.write(b'\x53\x6B\x38\x00') 
        f.write(struct.pack(">I", len(shapes))) 
        
        # 2. Global Shape Pointer Table
        shape_ptr_table_pos = f.tell()
        f.write(b'\x00' * (len(shapes) * 4))
        
        shape_offsets = []
        for s in shapes:
            shape_offsets.append(f.tell())
            f.write(struct.pack(">I I", s["id"], len(s["elements"])))
            
            units_ptr_table_pos = f.tell()
            f.write(b'\x00' * (len(s["elements"]) * 4))
            
            element_offsets = []
            for elem_bin in s["elements"]:
                element_offsets.append(f.tell())
                f.write(elem_bin)
            
            curr_pos = f.tell()
            f.seek(units_ptr_table_pos)
            for off in element_offsets:
                f.write(struct.pack(">I", off))
            f.seek(curr_pos)

        f.seek(shape_ptr_table_pos)
        for s_off in shape_offsets:
            f.write(struct.pack(">I", s_off))

    print(f"Created {output_path} with magic header and {len(shapes)} shapes.")

# Execute
# ru_list = [f for f in os.listdir('.') if f.endswith('.ru')]
# build_combined_geo(ru_list, "combined.geo")