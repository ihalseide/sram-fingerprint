# testing skip_space

import os
from serial_analysis import file_skip_space, file_read_next_hex4

if __name__ == '__main__':
    fname = "test.txt"

    if os.path.exists(fname):
        print(f"file {fname} exists")
        exit(1)
    with open(fname, "w") as f_out:
        print(f"  1111  2222", end='', file=f_out)

    try:

        with open(fname, "rb") as f_in:
            file_skip_space(f_in)
            assert f_in.tell() == 2
            assert 0x1111 == file_read_next_hex4(f_in)
            assert f_in.tell() == 6

        with open(fname, "rb") as f_in:
            file_read_next_hex4(f_in)
            file_read_next_hex4(f_in)
            file_read_next_hex4(f_in)

    finally:
        os.remove(fname)