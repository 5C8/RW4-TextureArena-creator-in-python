import os
import shutil
import subprocess
import sys
from PIL import Image

# Import logic from your existing scripts
from build_arena import process_folder
from rw4_writer import RW4Writer
from combine_to_geo import parse_ru_to_shape

def convert_to_dds_dxt5(folder_path):
    """Converts images to DXT5 DDS using texconv.exe."""
    print("Converting textures to DDS using texconv...")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tga')):
            img_path = os.path.join(folder_path, filename)
            base_name_no_ext = os.path.splitext(filename)[0]
            
            cmd = [
                "texconv.exe", 
                "-f", "DXT5", 
                "-y", 
                "-m", "1", 
                "-o", folder_path, 
                img_path
            ]
            
            # Run and capture output so we can see errors
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # texconv often outputs .DDS (uppercase). We need .dds (lowercase)
            expected_dds = os.path.join(folder_path, base_name_no_ext + ".dds")
            upper_dds = os.path.join(folder_path, base_name_no_ext + ".DDS")

            # 1. Rename .DDS to .dds if necessary
            if os.path.exists(upper_dds) and not os.path.exists(expected_dds):
                os.rename(upper_dds, expected_dds)

            # 2. Check if the file actually exists now
            if os.path.exists(expected_dds):
                os.remove(img_path)
                print(f"  Successfully converted: {filename} -> {base_name_no_ext}.dds")
            else:
                print(f"  !!! ERROR converting {filename} !!!")
                print(result.stdout) # This will show you exactly why texconv failed

def process_geometry(base_name, folder_path):
    """Aggregates .ru files into a .geo file using combine_to_geo logic."""
    print(f"Building {base_name}.geo...")
    ru_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.ru')]
    # Sort numerically if possible
    ru_files.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))) or 0))
    
    shapes = []
    for ru in ru_files:
        res = parse_ru_to_shape(ru)
        if res: shapes.append(res)

    import struct
    output_path = f"{base_name}.geo"
    with open(output_path, 'wb') as f:
        f.write(b'Sk8\x00') 
        f.write(struct.pack(">I", len(shapes))) 
        
        ptr_pos = f.tell()
        f.write(b'\x00' * (len(shapes) * 4))
        
        offsets = []
        for s in shapes:
            offsets.append(f.tell())
            f.write(struct.pack(">I I", s["id"], len(s["elements"])))
            u_ptr_pos = f.tell()
            f.write(b'\x00' * (len(s["elements"]) * 4))
            
            e_offsets = []
            for elem_bin in s["elements"]:
                e_offsets.append(f.tell())
                f.write(elem_bin)
            
            back = f.tell()
            f.seek(u_ptr_pos)
            for off in e_offsets: f.write(struct.pack(">I", off))
            f.seek(back)
            
        f.seek(ptr_pos)
        for off in offsets: f.write(struct.pack(">I", off))
    print(f"  Created {output_path}")

def main(swf_file):
    base_name = os.path.splitext(os.path.basename(swf_file))[0]
    swfc_path = os.path.join("Apt", "swfc.exe")
    
    # 1. Run swfc.exe
    print(f"Running swfc.exe on {swf_file}...")
    cmd = [swfc_path, "--endian=big", "-o", f"{base_name}.apt", swf_file]
    subprocess.run(cmd, check=True)

    # 2. Cleanup unused folders
    for suffix in ["_sounds", "_videos"]:
        dir_path = f"{base_name}{suffix}"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Deleted {dir_path}")

    # 3. Convert Textures to DDS
    tex_dir = f"{base_name}_textures"
    if os.path.exists(tex_dir):
        convert_to_dds_dxt5(tex_dir)
        
        # 4. Build Arena Files (.rx2 and .rps3)
        print(f"Building Arena files for {base_name}...")
        
        # XBOX
        xbox_entries = process_folder(tex_dir, platform="XBOX")
        if xbox_entries:
            writer = RW4Writer(platform="XBOX", arena_id=0x343534)
            with open(f"{base_name}.rx2", "wb") as f:
                f.write(writer.build(xbox_entries))
            print(f"  Created {base_name}.rx2")
            
        # PS3
        ps3_entries = process_folder(tex_dir, platform="PS3")
        if ps3_entries:
            writer = RW4Writer(platform="PS3", arena_id=0x343534)
            with open(f"{base_name}.rps3", "wb") as f:
                f.write(writer.build(ps3_entries))
            print(f"  Created {base_name}.rps3")

    # 5. Process Geometry
    geo_dir = f"{base_name}_geometry"
    if os.path.exists(geo_dir):
        process_geometry(base_name, geo_dir)

    # Delete temporary folders
    for suffix in ["_geometry", "_textures"]:
        dir_path = f"{base_name}{suffix}"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    print("\nWorkflow Complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pack_apt.py <filename>.swf")
    else:
        main(sys.argv[1])
