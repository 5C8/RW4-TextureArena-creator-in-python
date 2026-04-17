import struct
import math

# --- BIT MANIPULATION HELPERS ---

def breverse(val, size):
    """Reverses bits within a specific field size."""
    return int('{:0{width}b}'.format(val, width=size)[::-1], 2)

def finalize_xenos_dw(dw):
    binary_str = '{:032b}'.format(dw)
    reversed_dw = int(binary_str[::-1], 2)
    return struct.pack(">I", reversed_dw)

def generate_header(Width, Height, Mips, DataFormat, MipAddress):
    # --- DWORD 0 ---
    Tiled = 1
    Pitch = 16
    Padding = 0
    MultiSample = 0
    ClampZ = 0
    ClampY = 0
    ClampX = 0
    SignW = 0
    SignZ = 0
    SignY = 0
    SignX = 0
    Type = 2
    dw0 = (breverse(Tiled, 1)        << 0)  | \
          (breverse(Pitch, 9)        << 1)  | \
          (breverse(Padding, 1)      << 10) | \
          (breverse(MultiSample, 2)  << 11) | \
          (breverse(ClampZ, 3)       << 13) | \
          (breverse(ClampY, 3)       << 16) | \
          (breverse(ClampX, 3)       << 19) | \
          (breverse(SignW, 2)        << 22) | \
          (breverse(SignZ, 2)        << 24) | \
          (breverse(SignY, 2)        << 26) | \
          (breverse(SignX, 2)        << 28) | \
          (breverse(Type, 2)         << 30)

    # --- DWORD 1 ---
    BaseAddress = 0
    ClampPolicy = 0
    Stacked     = 0
    RequestSize = 0
    Endian      = 1
    dw1 = (breverse(BaseAddress, 20) << 0)  | \
          (breverse(ClampPolicy, 1)  << 20) | \
          (breverse(Stacked, 1)      << 21) | \
          (breverse(RequestSize, 2)  << 22) | \
          (breverse(Endian, 2)       << 24) | \
          (breverse(DataFormat, 6)   << 26)

    # --- DWORD 2 ---
    dw2_val = ((Width - 1) & 0x1FFF) | (((Height - 1) & 0x1FFF) << 13)

    # --- DWORD 3 ---
    AnisoFilter = 0
    MipFilter = 0
    MinFilter = 0
    MagFilter = 0
    SwizzleW = 0
    SwizzleZ = 3
    SwizzleY = 2
    SwizzleX = 1
    dw3 = (breverse(AnisoFilter, 3) << 4)  | \
          (breverse(MipFilter, 2) << 7)  | \
          (breverse(MinFilter, 2) << 9)  | \
          (breverse(MagFilter, 2) << 11) | \
          (breverse(SwizzleW, 3) << 16) | \
          (breverse(SwizzleZ, 3) << 19) | \
          (breverse(SwizzleY, 3) << 22) | \
          (breverse(SwizzleX, 3) << 25)

    # --- DWORD 4 ---
    GradExpAdjustV = 0
    GradExpAdjustH = 0
    LODBias        = 0
    MinAnisoWalk   = 0
    MagAnisoWalk   = 0
    MaxMipLevel    = max(0, Mips - 1)
    MinMipLevel    = 0
    VolMinFilter   = 0
    VolMagFilter   = 0
    dw4 = (breverse(GradExpAdjustV, 5) << 0)  | \
          (breverse(GradExpAdjustH, 5) << 5)  | \
          (breverse(LODBias, 10)       << 10) | \
          (breverse(MinAnisoWalk, 1)   << 20) | \
          (breverse(MagAnisoWalk, 1)   << 21) | \
          (breverse(MaxMipLevel, 4)    << 22) | \
          (breverse(MinMipLevel, 4)    << 26) | \
          (breverse(VolMinFilter, 1)   << 30) | \
          (breverse(VolMagFilter, 1)   << 31)

    # --- DWORD 5 ---
    PackedMips = 1
    Dimension = 1
    AnisoBias = 0
    TriClamp = 0
    ForceBCWtoMax = 0
    BorderColor = 0
    dw5 = (breverse(MipAddress, 20) << 0)  | \
          (breverse(PackedMips, 1)  << 20) | \
          (breverse(Dimension, 2)  << 21) | \
          (breverse(AnisoBias, 4)  << 23) | \
          (breverse(TriClamp, 2)  << 27) | \
          (breverse(ForceBCWtoMax, 1)  << 29) | \
          (breverse(BorderColor, 2)  << 30)
    
    res = bytearray()
    res.extend(finalize_xenos_dw(dw0))
    res.extend(finalize_xenos_dw(dw1))
    res.extend(struct.pack(">I", dw2_val))
    res.extend(finalize_xenos_dw(dw3))
    res.extend(finalize_xenos_dw(dw4))
    res.extend(finalize_xenos_dw(dw5))
    
    return bytes(res)

def align(val, alignment):
    return (val + alignment - 1) & ~(alignment - 1)

def app_log2(n):
    return n.bit_length() - 1 if n > 0 else -1

# --- XBOX 360 TILING ---

def get_xbox360_tiled_offset(x, y, width, log_bpb):
    aligned_width = align(width, 32)
    macro = ((x >> 5) + (y >> 5) * (aligned_width >> 5)) << (log_bpb + 7)
    micro = ((x & 7) + ((y & 0xE) << 2)) << log_bpb
    offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    address = (
        ((offset & ~0x1FF) << 3) +
        ((y & 16) << 7) +
        ((offset & 0x1C0) << 2) +
        (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) +
        (offset & 0x3F)
    )
    return address >> log_bpb

def tile_level(src_data, width, height, pitch, is_compressed,
               sx_offset=0, sy_offset=0,
               tiled_block_width=None, tiled_block_height=None,
               dst=None):
    """
    Tiles one mip level into dst (or a new buffer if dst is None).
    sx_offset/sy_offset: block-space offset within the tiled region (for packed mips).
    tiled_block_width/height: override the tile grid size (for packed mips sharing a chunk).
    """
    block_size = 4 if is_compressed else 1
    orig_bw = max(1, width  // block_size)
    orig_bh = max(1, height // block_size)

    if tiled_block_width is None:
        tw = align(width,  128) if is_compressed else align(width,  32)
        th = align(height, 128) if is_compressed else align(height, 32)
        tiled_block_width  = tw // block_size
        tiled_block_height = th // block_size

    out_size = tiled_block_width * tiled_block_height * pitch
    if dst is None:
        dst = bytearray(out_size)

    log_bpb = app_log2(pitch)

    for dy in range(orig_bh):
        for dx in range(orig_bw):
            swz_addr = get_xbox360_tiled_offset(
                dx + sx_offset, dy + sy_offset, tiled_block_width, log_bpb)
            dst_idx = swz_addr * pitch
            src_idx = (dy * orig_bw + dx) * pitch

            if src_idx + pitch <= len(src_data) and dst_idx + pitch <= len(dst):
                block = bytearray(src_data[src_idx : src_idx + pitch])
                if is_compressed:
                    for i in range(0, len(block), 2):
                        block[i], block[i+1] = block[i+1], block[i]
                else:
                    block.reverse()
                dst[dst_idx : dst_idx + pitch] = block

    return dst

# --- FORMAT MAP ---

FORMAT_MAP = {
    # pitch = data bytes per block (used for tiling)
    b'DXT1': {'idx': 18, 'pitch': 8, 'comp': True},
    b'DXT5': {'idx': 20, 'pitch': 16, 'comp': True},
    None:    {'idx': 6,  'pitch': 4,  'comp': False},
}

def compute_packed_mip_offsets_OLD(mip_width, mip_height, orig_width, orig_height,
                           pitch, block_size, is_wider):
    """
    Mirror the C++ packed-mip sx/sy offset logic.
    Returns (sx_offset, sy_offset) in block units.
    """
    orig_bw = max(1, mip_width  // block_size)
    orig_bh = max(1, mip_height // block_size)

    if is_wider:
        # width > height
        threshold = 2 / max(1, 16 // orig_height)
        if orig_bh > threshold:
            sy = orig_bh * max(1, 16 // orig_height)
            sx = 0
        else:
            sx = 4 * mip_width // block_size
            sy = 0
    else:
        # height >= width (square falls here)
        threshold = 2 / max(1, 16 // orig_width)
        if orig_bw > threshold:
            sx = orig_bw * max(1, 16 // orig_width)
            sy = 0
        else:
            sx = 0
            sy = 4 * mip_height // block_size

    return sx, sy

def compute_packed_mip_offsets(mip_infos, is_wider):
    packed = [m for m in mip_infos if m['is_packed']]
    if not packed:
        return {}

    first_bw   = packed[0]['bw']
    first_bh   = packed[0]['bh']
    num_packed = len(packed)
    log2_bh    = app_log2(first_bh) if first_bh > 0 else 0

    offsets = {}
    if is_wider:
        for pl, m in enumerate(packed):
            sy = first_bh >> pl
            sx = (first_bw >> (pl - log2_bh)) if sy == 0 else 0
            offsets[m['level']] = (sx, sy)
    else:
        for pl, m in enumerate(packed):
            sx = first_bw >> pl
            sy = (num_packed - pl) if sx == 0 else 0
            offsets[m['level']] = (sx, sy)

    return offsets

# --- MAIN CONVERTER ---

def convert_dds_to_x360(input_path, output_path):
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'DDS ':
            raise Exception("Not a valid DDS file")

        header = f.read(124)
        height  = struct.unpack("<I", header[8:12])[0]
        width   = struct.unpack("<I", header[12:16])[0]
        mips    = struct.unpack("<I", header[24:28])[0]
        fourcc  = header[80:84]

        raw_data = f.read()

    f_info     = FORMAT_MAP.get(fourcc, FORMAT_MAP[None])
    pitch      = f_info['pitch']
    is_comp    = f_info['comp']
    block_size = 4 if is_comp else 1
    is_wider   = width > height

    # chunksize: smallest aligned tiled mip size
    chunk_size = 4096 * pitch // block_size

    # MipAddress = mip0 tiled size / 4096 (GPU address units)
    if is_comp:
        mip0_tw = align(width, 128); mip0_th = align(height, 128)
        mip0_tiled = (mip0_tw // block_size) * (mip0_th // block_size) * pitch
    else:
        mip0_tw = align(width, 32); mip0_th = align(height, 32)
        mip0_tiled = mip0_tw * mip0_th * pitch
    mip_address = mip0_tiled // 4096

    gpu_header = generate_header(width, height, mips, f_info['idx'], mip_address)

    output_chunks = []
    src_offset    = 0

    mip_infos = []
    for level in range(mips):
        mw = max(1, width  >> level)
        mh = max(1, height >> level)
        bw = max(1, (mw + block_size - 1) // block_size)
        bh = max(1, (mh + block_size - 1) // block_size)
        raw_size = bw * bh * pitch

        if is_comp:
            tw = align(mw, 128)
            th = align(mh, 128)
        else:
            tw = align(mw, 32)
            th = align(mh, 32)
        tbw = tw // block_size
        tbh = th // block_size
        tiled_size = tbw * tbh * pitch

        is_packed = (mw <= 16 or mh <= 16) and (mips > 1)
        mip_infos.append({
            'level': level,
            'mw': mw, 'mh': mh,
            'bw': bw, 'bh': bh,
            'raw_size': raw_size,
            'tbw': tbw, 'tbh': tbh,
            'tiled_size': tiled_size,
            'is_packed': is_packed,
        })

    first_packed = next((m for m in mip_infos if m['is_packed']), None)
    packed_offsets = compute_packed_mip_offsets(mip_infos, is_wider)

    packed_chunk = None
    if first_packed is not None:
        packed_chunk = bytearray(first_packed['tiled_size'])

    for m in mip_infos:
        level    = m['level']
        mw, mh   = m['mw'], m['mh']
        raw_size = m['raw_size']

        src_slice = raw_data[src_offset : src_offset + raw_size]
        src_offset += raw_size

        if not m['is_packed']:
            tiled = tile_level(src_slice, mw, mh, pitch, is_comp)
            if len(tiled) % chunk_size != 0:
                tiled = tiled + b'\x00' * (align(len(tiled), chunk_size) - len(tiled))
            output_chunks.append(bytes(tiled))
        else:
            sx, sy = packed_offsets[level]
            tile_level(src_slice, mw, mh, pitch, is_comp,
                       sx_offset=sx, sy_offset=sy,
                       tiled_block_width=first_packed['tbw'],
                       tiled_block_height=first_packed['tbh'],
                       dst=packed_chunk)

    if packed_chunk is not None:
        output_chunks.append(bytes(packed_chunk))

    with open(output_path, 'wb') as out:
        out.write(gpu_header)
        for chunk in output_chunks:
            out.write(chunk)

    total = sum(len(c) for c in output_chunks)
    print(f"Converted {input_path} -> {output_path}")
    print(f"  {width}x{height}, {mips} mips, fourcc={fourcc}")
    print(f"  Header: {len(gpu_header)} bytes | Data: {hex(total)} bytes | Total: {hex(len(gpu_header)+total)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dds_to_x360.py <input.dds> <output.x360>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        convert_dds_to_x360(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
