import random
import array
import time
import json
import os
import random
import pefile
import tempfile
import subprocess
import math
import struct
import itertools
from pathlib import Path
from envs.utils.interface import fetch_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORT_API_PATH = Path(os.environ.get(
    "RLAEG_IMPORT_API_PATH",
    PROJECT_ROOT / "data" / "dll_imports.json",
))
SECTION_NAMES_PATH = Path(os.environ.get(
    "RLAEG_SECTION_NAMES_PATH",
    PROJECT_ROOT / "data" / "section_names.txt",
))

if IMPORT_API_PATH.exists():
    with open(IMPORT_API_PATH, 'r') as import_file:
        ImportAPI = json.load(import_file)
else:
    ImportAPI = {}

COMMON_SECTION_NAMES = SECTION_NAMES_PATH.read_text().rstrip().split('\n')


# document number
def doc_number(file_path):
    file_dict = {}
    num = 0
    for root, ds, fs in os.walk(file_path):
        for f in fs:
            file_name = f
            file_path = os.path.join(root, f)
            file_dict[num] = file_path
            num += 1
    return file_dict


pure_benign_path = os.environ.get(
    "RLAEG_PURE_BENIGN_DIR",
    str(PROJECT_ROOT / "data" / "Pure_Benign"),
)
file_dict = doc_number(pure_benign_path)
strings_dict = {0:'I love ToucanStrike <3', 1:'!This program cannot be run in DOS mode.', 2:'"" "" "" "" "', 3:"# ## ## ## ## #"}


# Benign content injection overlay
def overlay_append(bytez, index):

    # Injected by the proportion
    file_size = len(bytez)
    binary = pefile.PE(data=bytez)
    overlay = binary.get_overlay()
    file_alignment = binary.OPTIONAL_HEADER.FileAlignment
    append_length = (int(file_size * 1e-2 / file_alignment) + 1) * file_alignment

    # Get the benign content
    file_path = file_dict[index]
    benign_binary = pefile.PE(file_path)
    benign_overlay = benign_binary.get_overlay()
    if len(overlay) == 0:
        overlay = benign_overlay
        append_length = len(overlay)
        new_bytez = bytez + bytes(overlay)
    else:
        for sec in benign_binary.sections:
            if str(sec.Name.decode('UTF-8').strip(b'\x00'.decode())) == ".rdata":
                benign_bytez = sec.get_data()
                if len(benign_bytez) < append_length:
                    new_bytez = bytez + bytes(benign_bytez)
                    append_length = len(benign_bytez)
                else:
                    new_bytez = bytez + bytes(benign_bytez[:append_length])
                    append_length = append_length
                break

    return new_bytez, append_length


# section_injection
def section_add(bytez, index):
    binary = pefile.PE(data=bytez)

    last_section = binary.FILE_HEADER.NumberOfSections - 1
    file_alignment = binary.OPTIONAL_HEADER.FileAlignment
    section_alignment = binary.OPTIONAL_HEADER.SectionAlignment
    new_section_offset = (binary.sections[last_section].get_file_offset() + 40)

    # Injected by the proportion
    file_image = binary.OPTIONAL_HEADER.SizeOfImage
    add_size = (int(file_image * 1e-2 / file_alignment) + 1) * file_alignment

    # Get the benign content
    file_path = file_dict[index]
    benign_binary = pefile.PE(file_path)
    for sec in benign_binary.sections:
        if str(sec.Name.decode('UTF-8').strip(b'\x00'.decode())) == ".rdata":
            benign_bytez = sec.get_data()
            break

    name = "." + "".join(chr(random.randrange(ord('a'), ord('z'))) for _ in range(7))
    name = name.encode('utf-8')

    if len(benign_bytez) < add_size:
        data = benign_bytez
        raw_size = len(benign_bytez)
        virtual_size = len(benign_bytez)
    else:
        data = benign_bytez[:add_size]
        raw_size = add_size
        virtual_size = add_size

    if data and raw_size < len(data):
        raise Exception("Invalid raw_size.")
    if data and virtual_size < len(data):
        raise Exception("Invalid virtual_size.")

    def align(val_to_align, alignment):
        return int((val_to_align + alignment - 1) / alignment) * alignment

    # Look for valid values for the new section header
    raw_size = align(raw_size, file_alignment)
    virtual_size = align(virtual_size, section_alignment)
    raw_offset = align((binary.sections[last_section].PointerToRawData +
                           binary.sections[last_section].SizeOfRawData),
                          file_alignment)

    virtual_offset = align((binary.sections[last_section].VirtualAddress +
                               binary.sections[last_section].Misc_VirtualSize),
                              section_alignment)

    characteristics = 0xE0000020

    # Create the section
    # Set the name
    binary.set_bytes_at_offset(new_section_offset, name)
    # Set the virtual size
    binary.set_dword_at_offset(new_section_offset + 8, virtual_size)
    # Set the virtual offset
    binary.set_dword_at_offset(new_section_offset + 12, virtual_offset)
    # Set the raw size
    binary.set_dword_at_offset(new_section_offset + 16, raw_size)
    # Set the raw offset
    binary.set_dword_at_offset(new_section_offset + 20, raw_offset)
    # Set the following fields to zero
    binary.set_bytes_at_offset(new_section_offset + 24, (12 * b'\x00'))
    # Set the characteristics
    binary.set_dword_at_offset(new_section_offset + 36, characteristics)

    # Edit the value in the File and Optional headers
    binary.FILE_HEADER.NumberOfSections += 1
    binary.OPTIONAL_HEADER.SizeOfImage = virtual_size + virtual_offset

    # prepare section data
    if data:
        written_data = data
        if len(written_data) < raw_size:
            written_data += (raw_size - len(written_data)) * b'\x00'
    else:
        written_data = raw_size * b'\x00'

    # extend file
    if len(binary.__data__) < raw_offset:
        binary.__data__ += (raw_offset - len(binary.__data__)) * b'\x00'
    binary.__data__ = binary.__data__[:raw_offset] + written_data

    # reparse PE file
    new_bytez = binary.write()

    return new_bytez, raw_size


# Dos Header Content Replacement
def dos_change(bytez, index):
    binary = pefile.PE(data=bytez)
    sign_add = binary.DOS_HEADER.e_lfanew

    # Replacement with benign content
    file_path = file_dict[index]
    benign_bytez = fetch_file(os.path.basename(file_path), file_path)
    benign_binary = pefile.PE(file_path)

    bytez = bytearray(bytez)
    benign_bytez = bytearray(benign_bytez)

    # Replacement of useless part of DOS header
    bytez[2: 60] = benign_bytez[2: 60]
    # Replacement of DOS stub
    for sec in benign_binary.sections:
        if str(sec.Name.decode('UTF-8').strip(b'\x00'.decode())) == ".rdata":
            benign_bytez = bytearray(sec.get_data())[:512]
    if sign_add - 64 <= len(benign_bytez):
        bytez[64: sign_add] = benign_bytez[:sign_add - 64]
    else:
        bytez[64: 64 + len(benign_bytez)] = benign_bytez

    new_bytez = array.array('B', bytez).tobytes()

    return new_bytez, 0


# File Header and Optional Header Replacement
def head_disrupt(bytez):
    binary = pefile.PE(data=bytez)
    sign_add = binary.DOS_HEADER.e_lfanew

    bytez = bytearray(bytez)

    # File Header
    bytez[sign_add + 12: sign_add + 16] = bytes([random.randint(0, 255) for _ in range(4)])

    # Optional Header
    bytez[sign_add + 26: sign_add + 27] = bytes([random.randint(0, 255) for _ in range(1)])
    bytez[sign_add + 64: sign_add + 72] = bytes([random.randint(0, 255) for _ in range(8)])

    new_bytez = array.array('B', bytez).tobytes()

    return new_bytez, 0

# padding_bytez
def section_append(bytez, index):
    binary = pefile.PE(data=bytez)
    select_string = strings_dict[index]
    avail_len = 0
    for s in binary.sections:
        section_size = s.SizeOfRawData
        section_vsize = s.Misc_VirtualSize
        avail_len = section_size - section_vsize
        if avail_len > 0:
            x = list(bytearray(bytez))
            pattern = itertools.cycle(select_string)
            offset = s.PointerToRawData
            if avail_len > 4096:
                avail_len = 4096
            [x.insert(offset+section_vsize, ord(next(pattern))) for _ in range(avail_len)]
            s.Misc_VirtualSize += avail_len
            x = bytearray(bytez)
            break
    if avail_len > 0:
        bytez = array.array('B', x).tobytes()
    if avail_len < 0:
        avail_len = 0
    return bytez, avail_len

def section_rename(bytez, seed=1):
    # rename a random section
    random.seed(seed)
    binary = pefile.PE(data=bytez)
    targeted_section = random.choice(binary.sections)
    targeted_section.Name = random.choice(COMMON_SECTION_NAMES)[:7].encode('UTF-8')
    new_bytez = binary.write()

    return new_bytez, 0

def change_timestamp(bytez, seed=1):
    # change timestamp of the file
    random.seed(seed)
    binary = pefile.PE(data=bytez)
    binary.FILE_HEADER.TimeDateStamp = int(''.join(str(random.choice(range(1, 10))) for _ in range(9)))
    new_bytez = binary.write()

    return new_bytez, 0

def upx_pack(bytez, seed=1):
    # tested with UPX 3.91
    random.seed(seed)
    tmpfilename = os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

    # dump bytez to a temporary file
    with open(tmpfilename, 'wb') as outfile:
        outfile.write(bytez)

    options = ['--force', '--overlay=copy']
    compression_level = random.randint(1, 9)
    options += ['-{}'.format(compression_level)]
    # --exact
    # compression levels -1 to -9
    # --overlay=copy [default]

    # optional things:
    # --compress-exports=0/1
    # --compress-icons=0/1/2/3
    # --compress-resources=0/1
    # --strip-relocs=0/1
    options += ['--compress-exports={}'.format(random.randint(0, 1))]
    options += ['--compress-icons={}'.format(random.randint(0, 3))]
    options += ['--compress-resources={}'.format(random.randint(0, 1))]
    options += ['--strip-relocs={}'.format(random.randint(0, 1))]

    with open(os.devnull, 'w') as DEVNULL:
        retcode = subprocess.call(
            ['upx'] + options + [tmpfilename, '-o', tmpfilename + '_packed'], stdout=DEVNULL, stderr=DEVNULL)

    os.unlink(tmpfilename)

    if retcode == 0:  # successfully packed

        with open(tmpfilename + '_packed', 'rb') as infile:
            bytez = infile.read()

        os.unlink(tmpfilename + '_packed')
    return bytez, 0

def upx_unpack(bytez, seed=1):
    # dump bytez to a temporary file
    tmpfilename = os.path.join(
        tempfile._get_default_tempdir(), next(tempfile._get_candidate_names()))

    with open(tmpfilename, 'wb') as outfile:
        outfile.write(bytez)

    with open(os.devnull, 'w') as DEVNULL:
        retcode = subprocess.call(
            ['upx', tmpfilename, '-d', '-o', tmpfilename + '_unpacked'], stdout=DEVNULL, stderr=DEVNULL)

    os.unlink(tmpfilename)

    if retcode == 0:  # sucessfully unpacked
        with open(tmpfilename + '_unpacked', 'rb') as result:
            bytez = result.read()

        os.unlink(tmpfilename + '_unpacked')
    return bytez, 0

def remove_signature(bytez):
    binary = pefile.PE(data=bytez)
    Directory = binary.dump_dict()['Directories']
    has_signature = False
    for dir in Directory:
        if dir['Structure'] == 'IMAGE_DIRECTORY_ENTRY_SECURITY':
            has_signature = True
            cer_rva_offs = dir['VirtualAddress']['FileOffset']
            # cer_rva = dir['VirtualAddress']['Value']
            cer_size_offs = dir['Size']['FileOffset']
            # cer_size = dir['Size']['Value']

            bytez = bytearray(bytez)
            bytez[cer_rva_offs: cer_rva_offs + 4] = bytes([0, 0, 0, 0])
            bytez[cer_size_offs: cer_size_offs + 4] = bytes([0, 0, 0, 0])
            bytez = array.array('B', bytez).tobytes()

            break

    return bytez, 0

def remove_debug(bytez):
    binary = pefile.PE(data=bytez)
    Directory = binary.dump_dict()['Directories']
    has_debug = False
    for dir in Directory:
        if dir['Structure'] == 'IMAGE_DIRECTORY_ENTRY_SECURITY':
            has_debug = True
            deb_rva_offs = dir['VirtualAddress']['FileOffset']
            # deb_rva = dir['VirtualAddress']['Value']
            deb_size_offs = dir['Size']['FileOffset']
            # deb_size = dir['Size']['Value']

            bytez = bytearray(bytez)
            bytez[deb_rva_offs: deb_rva_offs + 4] = bytes([0, 0, 0, 0])
            bytez[deb_size_offs: deb_size_offs + 4] = bytes([0, 0, 0, 0])
            bytez = array.array('B', bytez).tobytes()

            break

    return bytez, 0

def break_optional_header_checksum(bytez):
    binary = pefile.PE(data=bytez)
    binary.OPTIONAL_HEADER.CheckSum = 0
    new_bytez = binary.write()
    return new_bytez, 0

def overlay_replace(bytez, index):
    # If overlay not exist, execute overlay_append
    file_size = len(bytez)
    binary = pefile.PE(data=bytez)
    overlay = binary.get_overlay()
    if len(overlay) == 0:
        new_bytez, lens = overlay_append(bytez, index)
        return new_bytez, lens
    else:
        L1 = len(overlay)

    # Get the benign content
    file_path = file_dict[index]
    benign_binary = pefile.PE(file_path)
    benign_overlay = benign_binary.get_overlay()
    for sec in benign_binary.sections:
        if str(sec.Name.decode('UTF-8').strip(b'\x00'.decode())) == ".rdata":
            benign_bytez = sec.get_data()
            break

    L3 = len(benign_overlay)
    L2 = len(benign_bytez)
    address = 0
    for i in binary.sections:
        address = i.PointerToRawData + i.SizeOfRawData
    B = bytearray(bytez)
    if L1 <= L2 + L3:
        if L1 <= L3:
            B[address:] = bytearray(benign_overlay[:L1])
        else:
            B[address:address + L3] = bytearray(benign_overlay[:L3])
            B[address + L3:] = bytearray(benign_bytez[:L1 - L3])
    else:
        B[address:address + L3] = bytearray(benign_overlay)
        B[address + L3:address + L2 + L3] = bytearray(benign_bytez)
    new_bytez = bytes(B)
    return new_bytez, 0

def shift_header(bytez, index):
    binary = pefile.PE(data=bytez)
    select_string = strings_dict[index]
    file_alignment = binary.OPTIONAL_HEADER.SectionAlignment
    file_image = binary.OPTIONAL_HEADER.SizeOfImage
    pe_position = binary.DOS_HEADER.e_lfanew
    extension_amount = int(
        math.ceil(file_image * 1e-2 / file_alignment) + 1) * file_alignment
    x = bytearray(bytez)
    x[0x3C:0x40] = struct.pack("<I", pe_position + extension_amount)

    x[pe_position + 60 + 20 + 4: pe_position + 60 + 20 + 4 + 4] = struct.pack("<I",
                                                                                     binary.OPTIONAL_HEADER.SizeOfHeaders + extension_amount)
    pattern = itertools.cycle(select_string)
    x = list(x)
    [x.insert(pe_position, ord(next(pattern))) for _ in range(extension_amount)]
    x = bytearray(x)
    bytez = bytes(x)
    for ix, _ in enumerate(binary.sections):
        pe_position = binary.DOS_HEADER.e_lfanew + extension_amount
        optional_header_size = binary.FILE_HEADER.SizeOfOptionalHeader
        coff_header_size = 24
        section_entry_length = 40
        size_of_raw_data_pointer = 20
        shift_position = (
                pe_position
                + coff_header_size
                + optional_header_size
                + (ix * section_entry_length)
                + size_of_raw_data_pointer
        )
        old_value = struct.unpack("<I", bytez[shift_position: shift_position + 4])[0]
        new_value = old_value + extension_amount
        new_value = struct.pack("<I", new_value)
        # print(new_value)
        x = bytearray(bytez)
        # print(x[shift_position: shift_position + 4])
        x[shift_position: shift_position + 4] = bytearray(new_value)
        # print(x[shift_position: shift_position + 4])
        bytez = bytes(x)
    new_bytez = bytez
    return new_bytez, 4096

def shift_content(bytez, index):
    binary = pefile.PE(data=bytez)
    select_string = strings_dict[index]
    time1 = time.perf_counter()
    file_alignment = binary.OPTIONAL_HEADER.SectionAlignment
    file_image = binary.OPTIONAL_HEADER.SizeOfImage
    if binary.sections[0].PointerToRawData == 0:
        for i, _ in enumerate(binary.sections):
            if binary.sections[i].PointerToRawData != 0:
                first_content_offset = binary.sections[i].PointerToRawData
                break
    else:
        first_content_offset = binary.sections[0].PointerToRawData
    extension_amount = int(
        math.ceil(file_image * 1e-2 / file_alignment) + 1) * file_alignment
    index_to_perturb = list(range(first_content_offset, first_content_offset + extension_amount))
    for i, _ in enumerate(binary.sections):
        pe_position = binary.DOS_HEADER.e_lfanew
        optional_header_size = binary.FILE_HEADER.SizeOfOptionalHeader
        coff_header_size = 24
        section_entry_length = 40
        size_of_raw_data_pointer = 20
        shift_position = (
                pe_position
                + coff_header_size
                + optional_header_size
                + (i * section_entry_length)
                + size_of_raw_data_pointer
        )
        old_value = struct.unpack("<I", bytez[shift_position: shift_position + 4])[0]
        new_value = old_value + extension_amount
        new_value = struct.pack("<I", new_value)
        x = bytearray(bytez)
        x[shift_position: shift_position + 4] = bytearray(new_value)
        bytez = bytes(x)
    x = bytearray(bytez)
    pattern = itertools.cycle(select_string)
    [x.insert(first_content_offset, ord(next(pattern))) for _ in range(extension_amount)]
    new_bytez = bytes(x)
    time2 = time.perf_counter()
    if time2 - time1 > 10:
        return bytez, 0
    return new_bytez, extension_amount


# action table
ACTION_TABLE = {
    1: 'dos_change',
    2: 'section_append',
    3: 'overlay_append',
    4: 'section_add',
    5: 'section_rename',
    6: 'change_time_stamp',
    7: 'remove_signature',
    8: 'remove_debug',
    9: 'upx_pack',
    10: 'upx_unpack',
    11: 'break_optional_header_checksum',
    12: 'overlay_replace',
    13: 'shift_header',
    14: 'header_disrupt',
    15: 'shift_content'
}
