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

def generate_header(Width, Height, Mips, DataFormat, HdrPitch, MipAddress, Endian=1, SwizzleY=2):
    # --- DWORD 0 ---
    Tiled = 1
    Pitch = HdrPitch
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

# FourCC-identified compressed formats
# pitch     = data bytes per block (used for tiling)
# hdr_pitch = GPU register Pitch field value (16 for all BCn)
# endian    = Xenos endian swap mode (1 = swap 16-bit pairs, as used by block-compressed)
FOURCC_FORMAT_MAP = {
    b'DXT1': {'idx': 18, 'pitch': 8,  'hdr_pitch': 16, 'comp': True,  'endian': 1},
    b'DXT3': {'idx': 19, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},
    b'DXT5': {'idx': 20, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},
    b'ATI1': {'idx': 58, 'pitch': 8,  'hdr_pitch': 16, 'comp': True,  'endian': 1},  # DXT3A
    b'ATI2': {'idx': 49, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},  # DXN/BC5
    b'BC4U': {'idx': 58, 'pitch': 8,  'hdr_pitch': 16, 'comp': True,  'endian': 1},  # DXT3A
    b'BC5U': {'idx': 49, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},  # DXN
}

# DX10 DXGI format codes (when FourCC == b'DX10', read from extended header)
DXGI_FORMAT_MAP = {
    71:  {'idx': 18, 'pitch': 8,  'hdr_pitch': 16, 'comp': True,  'endian': 1},  # BC1_UNORM (DXT1)
    74:  {'idx': 19, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},  # BC2_UNORM (DXT3)
    77:  {'idx': 20, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},  # BC3_UNORM (DXT5)
    80:  {'idx': 58, 'pitch': 8,  'hdr_pitch': 16, 'comp': True,  'endian': 1},  # BC4_UNORM
    83:  {'idx': 49, 'pitch': 16, 'hdr_pitch': 16, 'comp': True,  'endian': 1},  # BC5_UNORM
    28:  {'idx': 6,  'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # R8G8B8A8_UNORM
    87:  {'idx': 6,  'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # B8G8R8A8_UNORM
    56:  {'idx': 4,  'pitch': 2,  'hdr_pitch': 2,  'comp': False, 'endian': 1},  # B5G6R5_UNORM
    57:  {'idx': 3,  'pitch': 2,  'hdr_pitch': 2,  'comp': False, 'endian': 1},  # B5G5R5A1_UNORM
    35:  {'idx': 26, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # R16G16B16A16_UNORM
    10:  {'idx': 25, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # R16G16_UNORM
    56:  {'idx': 24, 'pitch': 2,  'hdr_pitch': 2,  'comp': False, 'endian': 1},  # R16_UNORM
    54:  {'idx': 32, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # R16G16B16A16_FLOAT
    34:  {'idx': 31, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # R16G16_FLOAT
    54:  {'idx': 30, 'pitch': 2,  'hdr_pitch': 2,  'comp': False, 'endian': 1},  # R16_FLOAT
    2:   {'idx': 38, 'pitch': 16, 'hdr_pitch': 16, 'comp': False, 'endian': 2},  # R32G32B32A32_FLOAT
    16:  {'idx': 37, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # R32G32_FLOAT
    41:  {'idx': 36, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # R32_FLOAT
}

def identify_uncompressed_format(dds_header):
    """
    Identifies an uncompressed DDS pixel format from the pixel format block.
    Returns a format info dict or None if unrecognized.

    DDS pixel format (DDSPF) layout within the 124-byte header (offsets relative to start of header):
      offset 76: dwSize (4)
      offset 80: dwFlags (4)
      offset 84: dwFourCC (4)
      offset 88: dwRGBBitCount (4)
      offset 92: dwRBitMask (4)
      offset 96: dwGBitMask (4)
      offset 100: dwBBitMask (4)
      offset 104: dwABitMask (4)
    """
    import struct
    pf_flags    = struct.unpack("<I", dds_header[76:80])[0]
    bit_count   = struct.unpack("<I", dds_header[84:88])[0]
    r_mask      = struct.unpack("<I", dds_header[88:92])[0]
    g_mask      = struct.unpack("<I", dds_header[92:96])[0]
    b_mask      = struct.unpack("<I", dds_header[96:100])[0]
    a_mask      = struct.unpack("<I", dds_header[100:104])[0]

    DDPF_RGB       = 0x40
    DDPF_ALPHA     = 0x02
    DDPF_LUMINANCE = 0x20000

    bpp = bit_count // 8  # bytes per pixel

    # Match by bit masks
    # 32-bit RGBA/RGBX
    if bit_count == 32 and b_mask == 0xFF and g_mask == 0xFF00 and r_mask == 0xFF0000:
        # ARGB8 or XRGB8  -> 8_8_8_8
        return {'idx': 6,  'pitch': 4, 'hdr_pitch': 4,  'comp': False, 'endian': 2}
    if bit_count == 32 and r_mask == 0xFF and g_mask == 0xFF00 and b_mask == 0xFF0000:
        # ABGR8 -> 8_8_8_8
        return {'idx': 6,  'pitch': 4, 'hdr_pitch': 4,  'comp': False, 'endian': 2}
    # 16-bit colour
    if bit_count == 16 and r_mask == 0xF800 and g_mask == 0x07E0 and b_mask == 0x001F:
        return {'idx': 4,  'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}  # R5G6B5
    if bit_count == 16 and r_mask == 0x7C00 and g_mask == 0x03E0 and b_mask == 0x001F:
        return {'idx': 3,  'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}  # A1R5G5B5
    if bit_count == 16 and r_mask == 0x0F00 and g_mask == 0x00F0 and b_mask == 0x000F:
        return {'idx': 15, 'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}  # A4R4G4B4
    # Luminance / alpha
    if bit_count == 8  and (pf_flags & (DDPF_LUMINANCE | DDPF_ALPHA | DDPF_RGB)):
        return {'idx': 2,  'pitch': 1, 'hdr_pitch': 1,  'comp': False, 'endian': 1}  # L8 / A8
    if bit_count == 16 and a_mask == 0xFF00 and (r_mask == 0xFF or g_mask == 0xFF):
        return {'idx': 10, 'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}  # A8L8
    if bit_count == 16 and r_mask == 0xFFFF and a_mask == 0 and g_mask == 0 and b_mask == 0:
        return {'idx': 24, 'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}  # L16
    # HDR / float formats (identified by FourCC in calling code, but cover RGBA16 here)
    if bit_count == 64:
        return {'idx': 26, 'pitch': 8, 'hdr_pitch': 8,  'comp': False, 'endian': 2}  # RGBA16
    if bit_count == 32 and r_mask == 0xFFFFFFFF and g_mask == 0 and b_mask == 0 and a_mask == 0:
        return {'idx': 36, 'pitch': 4, 'hdr_pitch': 4,  'comp': False, 'endian': 2}  # R32F / R32
    # Fallback: generic RGBA8
    if bpp == 4:
        return {'idx': 6,  'pitch': 4, 'hdr_pitch': 4,  'comp': False, 'endian': 2}
    if bpp == 2:
        return {'idx': 10, 'pitch': 2, 'hdr_pitch': 2,  'comp': False, 'endian': 1}
    if bpp == 1:
        return {'idx': 2,  'pitch': 1, 'hdr_pitch': 1,  'comp': False, 'endian': 1}
    return None

# Float/HDR FourCC codes (legacy D3D9 format codes stored as FourCC)
FLOAT_FOURCC_MAP = {
    b'\x6F\x00\x00\x00': {'idx': 30, 'pitch': 2,  'hdr_pitch': 2,  'comp': False, 'endian': 1},  # D3DFMT_R16F (111)
    b'\x70\x00\x00\x00': {'idx': 31, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # D3DFMT_G16R16F (112)
    b'\x71\x00\x00\x00': {'idx': 32, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # D3DFMT_A16B16G16R16F (113)
    b'\x72\x00\x00\x00': {'idx': 36, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # D3DFMT_R32F (114)
    b'\x73\x00\x00\x00': {'idx': 37, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # D3DFMT_G32R32F (115)
    b'\x74\x00\x00\x00': {'idx': 38, 'pitch': 16, 'hdr_pitch': 16, 'comp': False, 'endian': 2},  # D3DFMT_A32B32G32R32F (116)
    b'\x24\x00\x00\x00': {'idx': 26, 'pitch': 8,  'hdr_pitch': 8,  'comp': False, 'endian': 2},  # D3DFMT_A16B16G16R16 (36)
    b'\x3D\x00\x00\x00': {'idx': 25, 'pitch': 4,  'hdr_pitch': 4,  'comp': False, 'endian': 2},  # D3DFMT_G16R16 (61)
}

def compute_packed_mip_offsets(mip_infos, is_wider, block_size):
    """
    Compute (sx_offset, sy_offset) in block units for each packed mip.

    Square / taller (height >= width):
      sx = first_bw >> pl            counts down: 4,2,1,0,0,...
      sy = num_packed - pl           counts down when sx hits 0

    Wider (width > height):
      sy = first_bh >> pl            counts down: 4,2,1,0,0,...
      sx = first_bw >> (pl-log2_bh)  counts down when sy hits 0
    """
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
        # Work in pixel space for a consistent threshold across compressed (block_size=4)
        # and uncompressed (block_size=1) formats.
        #
        # sx branch: while (first_bw * block_size) >> pl >= 4 pixels
        #   sx = ((first_bw_px >> pl) // block_size) in block units
        # tail branch (sy): sy = first_bh >> (tail_pl + 1) in block units
        #   where tail_pl = pl - pl_tail
        #   This uses first_bh so taller textures (large first_bh) get larger sy steps.
        import math
        first_bw_px = first_bw * block_size
        pl_tail     = (int(math.log2(first_bw_px // 4)) + 1) if first_bw_px >= 4 else 0

        for pl, m in enumerate(packed):
            if pl < pl_tail:
                sx = (first_bw_px >> pl) // block_size
                sy = 0
            else:
                tail_pl = pl - pl_tail
                sy = first_bh >> (tail_pl + 1)
                sx = 0
            offsets[m['level']] = (sx, sy)

    return offsets

# --- MAIN CONVERTER ---

def resolve_format(fourcc, dds_header, extra_data):
    """
    Resolve DDS pixel format to a Xenos format info dict.
    Priority: FourCC compressed -> float FourCC -> DX10 extended -> uncompressed bitmask.
    """
    DDPF_FOURCC = 0x04
    pf_flags = struct.unpack("<I", dds_header[76:80])[0]

    # 1. FourCC-identified compressed formats
    if fourcc in FOURCC_FORMAT_MAP:
        return FOURCC_FORMAT_MAP[fourcc]

    # 2. Legacy D3D float/HDR formats (stored as integer FourCC)
    if fourcc in FLOAT_FOURCC_MAP:
        return FLOAT_FOURCC_MAP[fourcc]

    # 3. DX10 extended header
    if fourcc == b'DX10' and len(extra_data) >= 20:
        dxgi_fmt = struct.unpack("<I", extra_data[0:4])[0]
        if dxgi_fmt in DXGI_FORMAT_MAP:
            return DXGI_FORMAT_MAP[dxgi_fmt]
        raise Exception(f"Unsupported DXGI format {dxgi_fmt}")

    # 4. Uncompressed bitmask identification
    if not (pf_flags & DDPF_FOURCC):
        info = identify_uncompressed_format(dds_header)
        if info:
            return info

    raise Exception(f"Unsupported DDS format (FourCC={fourcc!r})")


def tile_dds_for_xbox(input_path):
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'DDS ':
            raise Exception("Not a valid DDS file")

        header = f.read(124)
        height  = struct.unpack("<I", header[8:12])[0]
        width   = struct.unpack("<I", header[12:16])[0]
        mips    = struct.unpack("<I", header[24:28])[0]
        fourcc  = header[80:84]

        # DX10 extended header lives right after the main 128-byte header
        extra_data = f.read(20) if fourcc == b'DX10' else b''
        raw_data   = f.read()

    if mips == 0:
        mips = 1

    f_info     = resolve_format(fourcc, header, extra_data)
    pitch      = f_info['pitch']
    hdr_pitch  = f_info['hdr_pitch']
    is_comp    = f_info['comp']
    endian     = f_info['endian']
    block_size = 4 if is_comp else 1
    is_wider   = width > height

    # chunksize: smallest aligned tiled mip size
    chunk_size = 4096 * pitch // block_size

    # MipAddress = mip0 tiled size / 4096 (GPU address units)
    if is_comp:
        mip0_tw    = align(width, 128)
        mip0_th    = align(height, 128)
        mip0_tbw   = mip0_tw // block_size
        mip0_tiled = mip0_tbw * (mip0_th // block_size) * pitch
        # hdr_pitch for compressed = mip0 tiled block-row width / 8
        hdr_pitch  = mip0_tbw // 8
    else:
        mip0_tw    = align(width, 32)
        mip0_th    = align(height, 32)
        mip0_tiled = mip0_tw * mip0_th * pitch
        # hdr_pitch for uncompressed = bytes per pixel (from format table)
        hdr_pitch  = f_info['hdr_pitch']
    mip_address = mip0_tiled // 4096

    fetch_constant = generate_header(width, height, mips, f_info['idx'],
                                 hdr_pitch, mip_address, endian,
                                 SwizzleY=2 if is_comp else 0)

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

    first_packed   = next((m for m in mip_infos if m['is_packed']), None)
    packed_offsets = compute_packed_mip_offsets(mip_infos, is_wider, block_size)

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
            if len(tiled) >= chunk_size and len(tiled) % chunk_size != 0:
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

    fmt_name = f"idx={f_info['idx']} {'comp' if is_comp else 'linear'} {pitch}bpb"
    print(f"  {width}x{height}, {mips} mip(s), fourcc={fourcc!r}, {fmt_name}")

    return fetch_constant, output_chunks

def convert_dds_to_x360(input_path, output_path):
    fetch_constant, output_chunks = tile_dds_for_xbox(input_path)
    with open(output_path, 'wb') as out:
        out.write(fetch_constant)
        for chunk in output_chunks:
            out.write(chunk)

    total = sum(len(c) for c in output_chunks)
    print(f"Converted {input_path} -> {output_path}")
    print(f"  Fetch Constant: {len(fetch_constant)}B | Data: {hex(total)} | Total: {hex(len(fetch_constant)+total)}")


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
