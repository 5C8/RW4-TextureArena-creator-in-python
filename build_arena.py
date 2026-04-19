import os
import sys
import struct
import glob
from rw4_writer import TextureEntry, PS3Format, write_rps3, write_rx2
from dds_to_x360 import tile_dds_for_xbox

def get_dds_info(dds_path):
    """Basic DDS header parser."""
    # Correctly use 'with' to open the file handle, not the bytes result
    with open(dds_path, 'rb') as f:
        raw_file = f.read()
        
    magic = raw_file[:4]
    if magic != b'DDS ': 
        raise Exception(f"File {dds_path} is not a valid DDS")
    
    # DDS Header is 124 bytes after magic (starts at index 4)
    header = raw_file[4:128]
    height = struct.unpack("<I", header[8:12])[0]
    width  = struct.unpack("<I", header[12:16])[0]
    mips   = struct.unpack("<I", header[24:28])[0]
    fourcc = header[80:84]
    data   = raw_file[128:] # Image data starts after the 128-byte header
        
    return width, height, mips, fourcc, data

def process_folder(input_folder, platform="XBOX"):
    """
    Scans a folder for .dds files and prepares them for the RW4 Arena.
    Internal Name Format: input_folder\\dds_name_no_ext.Texture
    """
    entries = []
    # Normalize folder name for the internal path
    folder_base = os.path.basename(input_folder.strip("\\/"))
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".dds"):
            continue
            
        full_path = os.path.join(input_folder, filename)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Internal name requirement: folder\\ddsname.Texture
        internal_name = f"{folder_base}\\{name_no_ext}.Texture"
        
        w, h, mips, fourcc, raw_data = get_dds_info(full_path)

        if platform == "XBOX":
            fetch_constant, chunks  = tile_dds_for_xbox(full_path)
            tiled_data = b"".join(chunks)
            
            entries.append(TextureEntry(
                name=internal_name,
                data=tiled_data,
                width=w, height=h,
                fmt=0, # Xbox uses fetch_constant instead
                fetch_constant=fetch_constant
            ))
            
        else: # PS3
            # PS3 uses original DDS data directly
            ps3_fmt_map = {b'DXT1': PS3Format.DXT1, b'DXT5': PS3Format.DXT5}
            fmt = ps3_fmt_map.get(fourcc, PS3Format.ARGB8888)
            
            entries.append(TextureEntry(
                name=internal_name,
                data=raw_data,
                width=w, height=h,
                fmt=fmt,
                mipmap=mips
            ))
            
    return entries

def process_single_texture(full_path, mode, platform, folder_base):
    """Prepares a single TextureEntry based on platform and mode."""
    name_no_ext = os.path.splitext(os.path.basename(full_path))[0]
    
    if name_no_ext.endswith(".Texture"):
        name_no_ext = name_no_ext[:-8]
    if mode == "multi":
        internal_name = f"{folder_base}\\{name_no_ext}.Texture"
    else:
        internal_name = f"{name_no_ext}.Texture"

    w, h, mips, fourcc, raw_data = get_dds_info(full_path)

    if platform == "xbox":
        fetch_constant, chunks  = tile_dds_for_xbox(full_path)
        tiled_data = b"".join(chunks)
        
        return TextureEntry(
            name=internal_name,
            data=tiled_data,
            width=w, height=h,
            fmt=0,
            fetch_constant=fetch_constant
        )
    else:
        # PS3 uses original DDS data
        ps3_fmt_map = {b'DXT1': PS3Format.DXT1, b'DXT5': PS3Format.DXT5}
        fmt = ps3_fmt_map.get(fourcc, PS3Format.ARGB8888)
        
        return TextureEntry(
            name=internal_name,
            data=raw_data,
            width=w, height=h,
            fmt=fmt,
            mipmap=mips
        )

def main():
    if len(sys.argv) < 4:
        print("Usage: python build_arena.py <single|multi> <ps3|xbox> <folder_path>")
        return

    mode = sys.argv[1].lower()
    platform = sys.argv[2].lower()
    folder_path = sys.argv[3].rstrip('\\/')
    folder_base = os.path.basename(folder_path)

    dds_files = glob.glob(os.path.join(folder_path, "*.dds"))
    if not dds_files:
        print(f"No DDS files found in {folder_path}")
        return

    write_func = write_rx2 if platform == "xbox" else write_rps3

    if mode == "multi":
        extension = ".rx2" if platform == "xbox" else ".rps3"
        print(f"Building multi-texture arena for {platform}...")
        entries = [process_single_texture(f, mode, platform, folder_base) for f in dds_files]
        arena_data = write_func(entries)
        
        output_name = f"{folder_base}{extension}"
        with open(output_name, "wb") as f:
            f.write(arena_data)
        print(f"Created {output_name}")

    elif mode == "single":
        extension = ".rx2" if platform == "xbox" else ".psg"
        print(f"Building single-texture arenas for {platform}...")
        for f in dds_files:
            entry = process_single_texture(f, mode, platform, folder_base)
            arena_data = write_func([entry])
            
            clean_name = os.path.splitext(os.path.basename(f))[0]
            if clean_name.endswith(".Texture"):
                clean_name = clean_name[:-8]
            output_name = f"{clean_name}{extension}"

            with open(output_name, "wb") as out:
                out.write(arena_data)
            print(f"Created {output_name}")

if __name__ == "__main__":
    main()