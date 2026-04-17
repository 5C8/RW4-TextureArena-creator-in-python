"""
RW4 Texture Arena Writer
Supports PS3 (.rps3) and XBOX 360 (.rx2) multi-texture arena files.
Made to complement the reader by tuukkas (2023-24).

Usage
-----
    from rw4_writer import RW4Writer, TextureEntry, PS3Format

    textures = [
        TextureEntry(
            name="team_textures\\\\1.Texture",
            data=dxt5_bytes,      # raw compressed texture data
            width=256, height=128,
            fmt=PS3Format.DXT5,
            mipmap=1,
        ),
        ...
    ]
    writer = RW4Writer(platform="PS3")
    data = writer.build(textures)
    with open("output.rps3", "wb") as f:
        f.write(data)

For XBOX 360 supply the 24-byte GPUTEXTURE_FETCH_CONSTANT blob per texture:

    TextureEntry(
        name="...", data=xenos_bytes, width=256, height=256, fmt=0,
        fetch_constant=bytes.fromhex("020000020000005400..."),
    )

Round-trip test (CLI)
---------------------
    python rw4_writer.py esrb.rps3
    python rw4_writer.py coachfrank.rx2
"""

import struct, io
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RWHASH64_INIT = 0xCBF29CE484222325
RW_64_PRIME   = 0x00000100000001B3

MAGIC_PREFIX  = bytes([0x89, 0x52, 0x57, 0x34])   # .RW4
MAGIC_SUFFIX  = bytes([0x0D, 0x0A, 0x1A, 0x0A])

PLATFORM_MAGIC = {
    "PS3":  bytes([0x70, 0x73, 0x33, 0x00]),
    "XBOX": bytes([0x78, 0x62, 0x32, 0x00]),
    "WII":  bytes([0x72, 0x65, 0x76, 0x00]),
}

RWOBJECTTYPE_BASERESOURCE_PS3  = 0x00010034
RWOBJECTTYPE_BASERESOURCE_XBOX = 0x00010031
RWGOBJECTTYPE_TEXTURE          = 0x000200E8
RWOBJECTTYPE_TABLEOFCONTENTS   = 0x00EB000B
RWOBJECTTYPE_VERSIONDATA       = 0x00EB0008

SECTION_MANIFEST       = 0x00010004
SECTION_TYPES          = 0x00010005
SECTION_EXTERNALARENAS = 0x00010006
SECTION_SUBREFERENCES  = 0x00010007
SECTION_ATOMS          = 0x00010008

TOC_UNKNOWN  = 0x9B0F1678
TOC_TYPE_TEX = 0xAC462E4A

VERSION_DATA = struct.pack(">II", 0x00000019, 0x00000002)

# ---------------------------------------------------------------------------
# PS3 texture format constants
# ---------------------------------------------------------------------------

class PS3Format:
    DXT1     = 0x86
    DXT3     = 0x87
    DXT5     = 0x88
    ARGB8888 = 0xA5

_BLOCK_SIZE = {PS3Format.DXT1: 8, PS3Format.DXT3: 16, PS3Format.DXT5: 16}

def ps3_pitch(width: int, fmt: int) -> int:
    return ((width + 3) // 4) * _BLOCK_SIZE.get(fmt, 16)

def dxt_data_size(width: int, height: int, fmt: int, mipmap: int = 1) -> int:
    bs = _BLOCK_SIZE.get(fmt, 16)
    total, w, h = 0, width, height
    for _ in range(mipmap):
        total += max(1, (w+3)//4) * max(1, (h+3)//4) * bs
        w, h = max(1, w//2), max(1, h//2)
    return total

# ---------------------------------------------------------------------------
# ID / GUID helpers
# ---------------------------------------------------------------------------

def rw_hash64(s: str, seed: int = RWHASH64_INIT) -> int:
    """FNV-1 64-bit hash used by the engine asset manager."""
    h = seed
    for b in s.encode("utf-8"):
        h = (h * RW_64_PRIME) & 0xFFFFFFFFFFFFFFFF
        h ^= b
    return h

def filename_to_asset_id(filename: str) -> int:
    """
    Convert a texture name like 'my_textures\\\\1.Texture' to its 64-bit GUID.
    Matches the engine's FilenameToAssetID logic exactly.
    """
    last = max(filename.rfind("/"), filename.rfind("\\"))
    base = filename[last+1:] if last != -1 else filename
    if len(base) >= 2 and base[0] == "0" and base[1] in "xX":
        hex_id = base.split('.')[0]
        return int(hex_id, 16) & 0xFFFFFFFFFFFFFFFF
    return rw_hash64(filename)

def _align(v: int, a: int) -> int:
    return v if a == 0 or v % a == 0 else v + a - v % a

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TextureEntry:
    """One texture to be packed into the arena."""
    name:   str    # e.g. "my_textures\\\\1.Texture"
    data:   bytes  # Raw texture data
    width:  int
    height: int
    fmt:    int    # PS3Format constant (ignored for XBOX if fetch_constant supplied)
    mipmap: int = 1

    # PS3 specific (sane defaults matching all sample files)
    remap:       int = 0x0000AAE4
    depth:       int = 1
    location:    int = 0
    store_type:  int = 0x00000002   # TYPE_2D
    store_flags: int = 0x00000000
    tex_unknown: int = 0x5572       # constant in all PS3 samples
    tex_offset:  int = 0            # pixel data offset within base resource
                                    # (0 for tex[0], 2 for tex[1], etc. in multi-tex)

    # XBOX specific: 24-byte GPUTEXTURE_FETCH_CONSTANT blob
    fetch_constant: Optional[bytes] = None

    # Internal: preserved resources_used[0].size from original file (runtime value)
    _resources_used_0: int = 0

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class RW4Writer:
    """
    Builds a binary RW4 multi-texture arena file (PS3 or XBOX 360).

    File layout
    -----------
    [Magic 12 B]
    [Arena_Header 20 B]
    [Arena struct 36 B]
    [Resource descriptors  rd_count×8 B]   rd=6 PS3 / 5 XBOX
    [Resources used        rd_count×8 B]
    [Target resources      tr_count×4 B]   tr=7 PS3 / 6 XBOX
    ── fixed_header: 0xC0 (PS3) / 0xAC (XBOX) ──
    [Manifest 28 B]
    [Types (12 + n×4) B]
    [External Arenas 36 B]
    [Subreferences 28 B]
    [Atoms 12 B]
    ── sections_end ──
    [Texture info structs  tex_count × 0x28 (PS3) / 0x34 (XBOX)]
    [Pad to 0x10]
    [TOC section]
    [Pad to 0x10]
    [Version data 8 B]
    [Dictionary  num_dict × 24 B]
    ── header_size (= dict_end, no extra padding) ──
    [Texture data blocks, each aligned to base_align in virtual address space]
    """

    def __init__(self, platform: str, arena_id: Optional[int] = None,
                 resources_used_0: int = 0):
        """
        platform          : "PS3" or "XBOX"
        arena_id          : 32-bit arena ID. If None, derived from texture names.
        resources_used_0  : Override for resources_used[0].size (runtime value).
                            Pass the value from the original file for exact recreation.
        """
        if platform not in ("PS3", "XBOX"):
            raise ValueError(f"Unsupported platform '{platform}'. Use 'PS3' or 'XBOX'.")
        self.platform         = platform
        self.arena_id         = arena_id
        self.resources_used_0 = resources_used_0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, textures: List[TextureEntry]) -> bytes:
        """Return the complete arena file as bytes."""
        if not textures:
            raise ValueError("Need at least one texture.")
        n   = len(textures)
        BE  = ">"

        # ---- arena_id ----
        arena_id = self.arena_id
        if arena_id is None:
            seed = RWHASH64_INIT
            for t in textures:
                seed = rw_hash64(t.name, seed)
            arena_id = seed & 0xFFFFFFFF

        # ---- platform geometry ----
        rd_count   = self._rd_count()
        tr_count   = self._tr_count()
        base_align = self._base_align()
        ti_size    = self._ti_size()
        rd_tex_idx = 5 if self.platform == "PS3" else 2   # which rd slot holds tex data

        # ---- fixed header size ----
        fixed_hdr = 12 + 20 + 36 + rd_count * 8 * 2 + tr_count * 4

        # ---- section sizes (constant per platform) ----
        types_list   = self._types_list()
        MANIFEST_SZ  = 12 + 4 * 4
        TYPES_SZ     = 12 + len(types_list) * 4
        EXT_ARENAS_SZ= 12 + 12 + 3 * 4
        SUBREFS_SZ   = 7 * 4
        ATOMS_SZ     = 3 * 4
        sections_sz  = MANIFEST_SZ + TYPES_SZ + EXT_ARENAS_SZ + SUBREFS_SZ + ATOMS_SZ

        off_types   = MANIFEST_SZ
        off_ext     = off_types  + TYPES_SZ
        off_subrefs = off_ext    + EXT_ARENAS_SZ
        off_atoms   = off_subrefs + SUBREFS_SZ

        # ---- positions of metadata objects in header ----
        ti_start  = fixed_hdr + sections_sz
        ti_end    = ti_start  + n * ti_size
        toc_start = _align(ti_end, 0x10)
        toc_sz    = self._toc_byte_size(textures)
        toc_end   = toc_start + toc_sz
        ver_start = _align(toc_end, 0x10)
        ver_end   = ver_start + 8
        dict_start= ver_end
        num_dict  = n * 2 + 2
        dict_end  = dict_start + num_dict * 24
        header_size = dict_end          # no extra alignment

        subrefs_ptr = header_size       # points past the header into texture data space

        # ---- virtual texture layout (NO alignment after the last block) ----
        tex_ptrs   = []
        cur_vptr   = 0
        for i, t in enumerate(textures):
            tex_ptrs.append(cur_vptr)
            # Add raw length
            cur_vptr += len(t.data)
            
            # Only align if there is another texture following this one
            if i < n - 1:
                cur_vptr = _align(cur_vptr, base_align)
                
        total_tex_virt = cur_vptr  # This will now be 0x61088 (396864)

        # ---- resources_used[0] ----
        ru0_size = self.resources_used_0
        if ru0_size == 0:
            ru0_size = header_size      # reasonable default

        # ---- assemble ----
        buf = io.BytesIO()

        # Magic
        buf.write(MAGIC_PREFIX)
        buf.write(PLATFORM_MAGIC[self.platform])
        buf.write(MAGIC_SUFFIX)

        # Arena_Header
        buf.write(bytes([0x01, 0x20, 0x04, 0x00]))
        buf.write(b"454\x00")
        buf.write(b"000\x00")
        buf.write(struct.pack(BE + "II", 0, arena_id))

        # Arena struct
        buf.write(struct.pack(BE + "IIIIIIIII",
            num_dict,           # numEntries
            num_dict,           # numUsed
            0x00000010,         # alignment (constant in all samples)
            0x00000000,         # virt
            dict_start,         # DictionaryStart (absolute)
            fixed_hdr,          # Arena_Section_Manifest (absolute)
            0x00000000,         # base
            0x00000000,         # m_unfixContext
            0x00000000,         # m_fixContext
        ))

        assert buf.tell() == fixed_hdr - rd_count * 8 * 2 - tr_count * 4

        # Resource descriptors
        # [0]: header_size / 0x10
        # [rd_tex_idx]: total virtual texture data size / base_align
        # rest: 0 / 1
        for i in range(rd_count):
            if i == 0:
                buf.write(struct.pack(BE + "II", header_size, 0x00000010))
            elif i == rd_tex_idx:
                buf.write(struct.pack(BE + "II", total_tex_virt, base_align))
            else:
                buf.write(struct.pack(BE + "II", 0, 1))

        # Resources used
        for i in range(rd_count):
            if i == 0:
                buf.write(struct.pack(BE + "II", ru0_size, 0x00000004))
            else:
                buf.write(struct.pack(BE + "II", 0, 1))

        # Target resources (all zero)
        for _ in range(tr_count):
            buf.write(struct.pack(BE + "I", 0))

        assert buf.tell() == fixed_hdr

        # ---- Sections ----

        # Manifest
        buf.write(struct.pack(BE + "III", SECTION_MANIFEST, 4, 0x0C))
        buf.write(struct.pack(BE + "IIII", off_types, off_ext, off_subrefs, off_atoms))

        # Types
        buf.write(struct.pack(BE + "III", SECTION_TYPES, len(types_list), 0x0C))
        for t in types_list:
            buf.write(struct.pack(BE + "I", t))

        # External Arenas
        buf.write(struct.pack(BE + "III", SECTION_EXTERNALARENAS, 3, 0x18))
        buf.write(struct.pack(BE + "III", arena_id, 0xFFB00000, arena_id))
        buf.write(struct.pack(BE + "III", 0, 0, 0))

        # Subreferences
        buf.write(struct.pack(BE + "IIIIIII",
            SECTION_SUBREFERENCES, 0, 0, 0,
            subrefs_ptr, subrefs_ptr, 0))

        # Atoms
        buf.write(struct.pack(BE + "III", SECTION_ATOMS, 0, 0))

        assert buf.tell() == ti_start

        # ---- Texture info structs ----
        for i, t in enumerate(textures):
            buf.write(self._build_ti(t, i))

        assert buf.tell() == ti_end

        # Pad to TOC
        buf.write(b"\x00" * (toc_start - ti_end))

        # ---- TOC ----
        buf.write(self._build_toc(textures, toc_start))
        assert buf.tell() == toc_end

        # Pad to version
        buf.write(b"\x00" * (ver_start - toc_end))

        # ---- Version data ----
        buf.write(VERSION_DATA)
        assert buf.tell() == ver_end

        # ---- Dictionary ----
        base_type_id  = RWOBJECTTYPE_BASERESOURCE_PS3 if self.platform == "PS3" else RWOBJECTTYPE_BASERESOURCE_XBOX
        base_type_idx = self._type_idx(base_type_id)
        ti_type_idx   = self._type_idx(RWGOBJECTTYPE_TEXTURE)
        toc_type_idx  = self._type_idx(RWOBJECTTYPE_TABLEOFCONTENTS)
        ver_type_idx  = self._type_idx(RWOBJECTTYPE_VERSIONDATA)

        for i, t in enumerate(textures):
            # Base resource entry — ptr is virtual (relative to start of data space)
            buf.write(struct.pack(BE + "IIIIII",
                tex_ptrs[i], 0, len(t.data), base_align,
                base_type_idx, base_type_id))
            # Texture info entry — ptr is absolute in-file offset into header
            buf.write(struct.pack(BE + "IIIIII",
                ti_start + i * ti_size, 0, ti_size, 0x00000004,
                ti_type_idx, RWGOBJECTTYPE_TEXTURE))

        # TOC
        buf.write(struct.pack(BE + "IIIIII",
            toc_start, 0, toc_sz, 0x00000010,
            toc_type_idx, RWOBJECTTYPE_TABLEOFCONTENTS))

        # Version
        buf.write(struct.pack(BE + "IIIIII",
            ver_start, 0, 0x00000008, 0x00000010,
            ver_type_idx, RWOBJECTTYPE_VERSIONDATA))

        assert buf.tell() == header_size

        # ---- Texture data (with alignment padding between blocks) ----
        for i, t in enumerate(textures):
            buf.write(t.data)
            # Pad to next aligned boundary (except after last texture)
            if i < n - 1:
                padded_end = _align(tex_ptrs[i] + len(t.data), base_align)
                padding    = padded_end - (tex_ptrs[i] + len(t.data))
                buf.write(b"\x00" * padding)

        result = buf.getvalue()
        expected = header_size + total_tex_virt
        assert len(result) == expected, \
            f"header_size: {header_size} total_tex_virt: {total_tex_virt} Size mismatch: got {len(result):#x}, expected {expected:#x}"
        return result

    # ------------------------------------------------------------------
    # Sub-builders
    # ------------------------------------------------------------------

    def _build_ti(self, t: TextureEntry, index: int) -> bytes:
        """Build a TextureInformationPS3 or TextureInformationX360 struct."""
        BE = ">"
        if self.platform == "PS3":
            pitch = ps3_pitch(t.width, t.fmt)
            # tex_offset field: index * 2 in multi-texture files (observed pattern)
            tex_off = t.tex_offset if t.tex_offset != 0 else index * 2
            return struct.pack(BE + "BBBB I HHH BB IIIII HBB",
                t.fmt, t.mipmap, 0x02, 0x00,
                t.remap,
                t.width, t.height, t.depth,
                t.location, 0x00,
                pitch,
                0x00000000,     # buffer (runtime)
                tex_off,        # offset field = index * 2
                t.store_type, t.store_flags,
                t.tex_unknown, 0x00, t.fmt,
            )
        else:   # XBOX
            fc = t.fetch_constant
            if fc is None or len(fc) != 0x18:
                raise ValueError(f"'{t.name}': XBOX needs a 24-byte fetch_constant.")
            return struct.pack(BE + "IIIIIII",
                0x00000003, 0x00000001,
                0x00000000, 0x00000000, 0x00000000,
                0xFFFF0000, 0xFFFF0000,
            ) + fc

    def _build_toc(self, textures: List[TextureEntry], toc_abs: int) -> bytes:
        """
        Build the Table of Contents section.
        nameOff in each entry is relative to start of the TOC section.
        """
        BE = ">"
        n  = len(textures)
        p_array = 0x14   # 5 header uint32s = 20 bytes

        # Build names blob first to know offsets
        names_blob, name_offsets = b"", []
        for t in textures:
            name_offsets.append(p_array + n * 24 + len(names_blob))
            names_blob += t.name.encode("utf-8") + b"\x00"
        while len(names_blob) % 4:
            names_blob += b"\x00"

        p_names    = p_array + n * 24
        p_type_map = p_names + len(names_blob)

        buf = io.BytesIO()
        buf.write(struct.pack(BE + "IIIII", n, p_array, p_names, 0, p_type_map))

        for i, t in enumerate(textures):
            idx = i * 2 + 1   # 1-based dict index of base resource
            buf.write(struct.pack(BE + "II Q II",
                name_offsets[i], TOC_UNKNOWN,
                filename_to_asset_id(t.name),
                TOC_TYPE_TEX, idx,
            ))

        buf.write(names_blob)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Platform helpers
    # ------------------------------------------------------------------

    def _types_list(self) -> list:
        if self.platform == "PS3":
            return [0x00000000,
                    0x00010030, 0x00010031, 0x00010032, 0x00010033, 0x00010034,
                    0x00010010, 0x000200E8, 0x00EB0008, 0x00EB000B]
        else:
            return [0x00000000,
                    0x00010030, 0x00010031, 0x00010032, 0x00010033,
                    0x00010010, 0x000200E8, 0x00EB0008, 0x00EB000B]

    def _type_idx(self, tid: int) -> int:
        return self._types_list().index(tid)

    def _rd_count(self) -> int:
        return 6 if self.platform == "PS3" else 5

    def _tr_count(self) -> int:
        return 7 if self.platform == "PS3" else 6

    def _base_align(self) -> int:
        return 0x80 if self.platform == "PS3" else 0x1000

    def _ti_size(self) -> int:
        return 0x28 if self.platform == "PS3" else 0x34

    def _toc_byte_size(self, textures: List[TextureEntry]) -> int:
        names_blob = b""
        for t in textures:
            names_blob += t.name.encode("utf-8") + b"\x00"
        while len(names_blob) % 4:
            names_blob += b"\x00"
        return 5 * 4 + len(textures) * 24 + len(names_blob)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def write_rps3(textures: List[TextureEntry], arena_id: Optional[int] = None,
               resources_used_0: int = 0) -> bytes:
    """Build a PS3 RW4 arena. Returns raw bytes."""
    return RW4Writer("PS3", arena_id, resources_used_0).build(textures)

def write_rx2(textures: List[TextureEntry], arena_id: Optional[int] = None,
              resources_used_0: int = 0) -> bytes:
    """Build an XBOX 360 RW4 arena. Returns raw bytes."""
    return RW4Writer("XBOX", arena_id, resources_used_0).build(textures)


# ---------------------------------------------------------------------------
# Minimal reader (for round-trip testing / extraction)
# ---------------------------------------------------------------------------

def read_arena(path: str):
    """
    Read all textures from an existing PS3 or XBOX arena file.

    Returns
    -------
    (platform: str, arena_id: int, resources_used_0: int, textures: list[TextureEntry])

    Preserving arena_id and resources_used_0 lets you reproduce the original
    file byte-for-byte with RW4Writer(platform, arena_id, resources_used_0).build(textures).
    """
    with open(path, "rb") as f:
        raw = f.read()

    BE = ">"
    def r32(o): return struct.unpack_from(BE+"I", raw, o)[0]

    body = raw[4:8]
    if   body == PLATFORM_MAGIC["PS3"]:  platform = "PS3"
    elif body == PLATFORM_MAGIC["XBOX"]: platform = "XBOX"
    else: raise ValueError("Unknown platform magic")

    # Arena_Header.id at offset 12+4+4+4+4 = 28
    arena_id = r32(28)

    rd_count    = 6 if platform == "PS3" else 5
    ru0_off     = 12 + 20 + 36 + rd_count * 8     # start of resources_used array
    ru0_size    = r32(ru0_off)

    arena_off   = 32
    num_entries = r32(arena_off)
    dict_start  = r32(arena_off + 16)
    res_desc_sz = r32(arena_off + 36)              # = header_size

    dicts = []
    for i in range(num_entries):
        o = dict_start + i * 24
        ptr, reloc, size, align, tidx, tid = struct.unpack_from(BE+"IIIIII", raw, o)
        dicts.append((ptr, reloc, size, align, tidx, tid))

    # Find TOC
    toc_off = None
    for ptr, _, size, _, _, tid in dicts:
        if tid == RWOBJECTTYPE_TABLEOFCONTENTS:
            toc_off, toc_sz = ptr, size
            break
    if toc_off is None:
        raise ValueError("No TOC section found")

    ui_items, p_array, p_names, _, p_typemap = struct.unpack_from(BE+"IIIII", raw, toc_off)

    textures = []
    for i in range(ui_items):
        e         = toc_off + p_array + i * 24
        name_off  = r32(e)
        unk       = r32(e + 4)
        guid,     = struct.unpack_from(BE+"Q", raw, e + 8)
        typ       = r32(e + 16)
        idx       = r32(e + 20)

        name_abs = toc_off + name_off
        name_end = raw.index(b"\x00", name_abs)
        name     = raw[name_abs:name_end].decode("utf-8")

        di        = idx - 1
        base_ptr, _, base_sz, _, _, _ = dicts[di]
        data_off  = base_ptr + res_desc_sz
        data      = raw[data_off : data_off + base_sz]

        ti_ptr    = dicts[di + 1][0]
        if platform == "PS3":
            fmt, mip       = struct.unpack_from("BB", raw, ti_ptr)
            w, h           = struct.unpack_from(BE+"HH", raw, ti_ptr + 8)
            remap,         = struct.unpack_from(BE+"I",  raw, ti_ptr + 4)
            tex_off,       = struct.unpack_from(BE+"I",  raw, ti_ptr + 20)   # offset field
            store_type,    = struct.unpack_from(BE+"I",  raw, ti_ptr + 28)
            tex_unknown,   = struct.unpack_from(BE+"H",  raw, ti_ptr + 36)
            textures.append(TextureEntry(
                name=name, data=data, width=w, height=h,
                fmt=fmt, mipmap=mip, remap=remap,
                store_type=store_type, tex_unknown=tex_unknown,
                tex_offset=tex_off,
            ))
        else:   # XBOX
            fc = raw[ti_ptr + 28 : ti_ptr + 28 + 0x18]
            textures.append(TextureEntry(
                name=name, data=data, width=0, height=0,
                fmt=0, fetch_constant=fc,
            ))

    return platform, arena_id, ru0_size, textures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    in_path = sys.argv[1]
    print(f"Round-trip test: {in_path}")

    platform, orig_arena_id, ru0, textures = read_arena(in_path)
    print(f"  Platform : {platform}")
    print(f"  Arena ID : {orig_arena_id:#010x}")
    print(f"  ru0_size : {ru0:#010x}")
    print(f"  Textures : {len(textures)}")
    for i, t in enumerate(textures):
        print(f"    [{i:2d}] {t.name!r:52s} {t.width}x{t.height}  "
              f"fmt=0x{t.fmt:02x}  {len(t.data)} B")

    writer  = RW4Writer(platform, arena_id=orig_arena_id, resources_used_0=ru0)
    rebuilt = writer.build(textures)

    with open(in_path, "rb") as f:
        original = f.read()

    if rebuilt == original:
        print(f"\n  ✓ EXACT MATCH ({len(rebuilt)} bytes)")
    else:
        total_diff = sum(a != b for a, b in zip(original, rebuilt))
        print(f"\n  ✗ {total_diff} bytes differ "
              f"(orig={len(original)}, rebuilt={len(rebuilt)})")
        for i, (a, b) in enumerate(zip(original, rebuilt)):
            if a != b:
                lo, hi = max(0, i-8), min(len(original), i+48)
                print(f"  First diff at {i:#x}:")
                for j in range(lo, hi, 16):
                    oc = original[j:j+16]
                    rc = rebuilt[j:j+16] if j < len(rebuilt) else b""
                    ho = " ".join(f"{x:02X}" for x in oc)
                    hr = " ".join(f"{x:02X}" for x in rc)
                    print(f"  {j:#08x}  {ho:<48}  orig")
                    print(f"  {j:#08x}  {hr:<48}  rebl")
                break

    out = os.path.basename(in_path).replace(".", "_rebuilt.")
    with open(out, "wb") as f:
        f.write(rebuilt)
    print(f"  Written  : {out}")