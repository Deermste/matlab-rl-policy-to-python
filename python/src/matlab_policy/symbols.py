"""Find exported symbols in a compiled policy library.

MATLAB Coder generates C++ for RL policies and does not wrap the entry point in
``extern "C"``, so the exported names are mangled (MSVC on Windows:
``?evaluatePolicy@@YANQEBM@Z``, Itanium on Linux: ``_Z14evaluatePolicyPKf``).
ctypes needs the exact name, so the export table is read directly. The parsers
below use only ``struct``; no pefile or pyelftools is required.
"""

import struct


def enumerate_exports(path):
    """Return the exported symbol names of a PE (.dll) or ELF (.so) file.

    Returns None if the file cannot be read or is neither PE nor ELF.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    try:
        if data[:4] == b"\x7fELF":
            return _elf_exports(data)
        if data[:2] == b"MZ":
            return _pe_exports(data)
    except (struct.error, ValueError, IndexError):
        return None
    return None


def resolve_symbol(lib, base_name, exports):
    """Return the ctypes function object for ``base_name`` or None.

    Tries the plain name first (works if the library used extern "C"), then the
    MSVC and Itanium mangling prefixes against the export list.
    """
    try:
        return getattr(lib, base_name)
    except (AttributeError, OSError):
        pass
    for prefix in mangling_prefixes(base_name):
        for name in exports or ():
            if name.startswith(prefix):
                try:
                    return getattr(lib, name)
                except (AttributeError, OSError):
                    continue
    return None


def mangling_prefixes(base_name):
    return (f"?{base_name}@@",                      # MSVC
            f"_Z{len(base_name)}{base_name}")       # Itanium


def _pe_exports(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    n_sections, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    dir_off = opt + (96 if magic == 0x10B else 112)      # PE32 / PE32+
    export_rva, _export_size = struct.unpack_from("<II", data, dir_off)
    if export_rva == 0:
        return []

    sections = []
    for i in range(n_sections):
        s = opt + opt_size + 40 * i
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, s + 8)
        sections.append((vaddr, max(vsize, rawsize), rawptr))

    def rva_to_off(rva):
        for vaddr, size, rawptr in sections:
            if vaddr <= rva < vaddr + size:
                return rawptr + rva - vaddr
        raise ValueError("RVA outside all sections")

    exp = rva_to_off(export_rva)
    n_names, = struct.unpack_from("<I", data, exp + 24)
    names_rva, = struct.unpack_from("<I", data, exp + 32)
    names_off = rva_to_off(names_rva)
    out = []
    for i in range(n_names):
        name_rva, = struct.unpack_from("<I", data, names_off + 4 * i)
        start = rva_to_off(name_rva)
        out.append(data[start:data.index(b"\0", start)].decode("utf-8", "replace"))
    return out


def _elf_exports(data):
    is64 = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if is64:
        e_shoff, = struct.unpack_from(end + "Q", data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from(end + "HH", data, 0x3A)
        sh_fmt, sym_fmt, sym_size, shndx_i = end + "IIQQQQIIQQ", end + "IBBHQQ", 24, 3
    else:
        e_shoff, = struct.unpack_from(end + "I", data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from(end + "HH", data, 0x2E)
        sh_fmt, sym_fmt, sym_size, shndx_i = end + "IIIIIIIIII", end + "IIIBBH", 16, 5

    sections = [struct.unpack_from(sh_fmt, data, e_shoff + i * e_shentsize)
                for i in range(e_shnum)]
    names = []
    for sec in sections:
        if sec[1] != 11:                                  # SHT_DYNSYM
            continue
        sh_offset, sh_size, sh_link = sec[4], sec[5], sec[6]
        str_off, str_size = sections[sh_link][4], sections[sh_link][5]
        for k in range(sh_size // sym_size):
            ent = struct.unpack_from(sym_fmt, data, sh_offset + k * sym_size)
            st_name, st_shndx = ent[0], ent[shndx_i]
            if st_name == 0 or st_shndx == 0:             # unnamed / undefined
                continue
            s = str_off + st_name
            names.append(data[s:data.index(b"\0", s, str_off + str_size)]
                         .decode("utf-8", "replace"))
    return names
