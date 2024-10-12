
import sys, os
from serial_analysis import *


def convert(filename, filename_out, num_words):
    with open(filename, 'rb') as file_in:
        with open(filename_out, 'w') as file_out:
            while line := file_in.readline():
                line2 = None
                try:
                    line_str = line.decode('ascii')
                except UnicodeDecodeError as e:
                    print("Error at:")
                    print(f"file pos = {file_in.tell()}")
                    print(f"line = {line}")
                    print(f"previous line = {line_str}")
                    print(f"previous line2 = {line2}")
                    raise e
                if line_str.strip() == '[BINARY]':
                    line2 = file_in.readline().decode('ascii')
                    assert line2.strip() == '[begin memory dump]', "file has [BINARY] without beginnning data dump"
                    print(line2[:-1], end='', file=file_out)
                    
                    remaining_bytes_to_read = num_words * 2 # 1 word is 2 bytes
                    data = file_in.read(remaining_bytes_to_read)

                    if len(data) != remaining_bytes_to_read:
                        raise ValueError("not enough data")
                    for i in range(num_words):
                        w_hi = data[i * 2 + 0]
                        w_lo = data[i * 2 + 1]
                        # Print word as hi,low
                        print(f"{w_hi:02X}{w_lo:02X} ", end='', file=file_out)
                        # Newline after every 16 words
                        if (i + 1) % 16 == 0 and (i != num_words - 1):
                            print(file=file_out)
                else:
                    print(line_str[:-1], end='', file=file_out)


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        raise ValueError("need input file argument")
    
    if len(sys.argv) > 2:
        filename_out = sys.argv[2]
    else:
        raise ValueError("need output file argument")
    
    if os.path.exists(filename_out):
        raise ValueError(f"output file {filename_out} already exists")
    
    convert(filename, filename_out, NUM_WORDS)


if __name__ == '__main__':
    main()