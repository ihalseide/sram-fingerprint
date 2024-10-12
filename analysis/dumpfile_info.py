'''dump file statistics'''

import numpy as np
import sys
import serial_analysis
from serial_analysis import hex4_i, percent


def main():
    if len(sys.argv) > 1:
        dump_filename = sys.argv[1]
    else:
        dump_filename = input("enter file name: ")

    if len(sys.argv) > 2:
        binary_format = (sys.argv[2].lower() == "true")
    else:
        binary_format = (input("Enter 'true' if dump uses binary format: ").lower() == "true")

    with open(dump_filename, 'rb') as file_in:
        num_words = 2**18
        serial_analysis.file_seek_next_data_dump(file_in, binary_format)
        print(f"* data word count = {num_words}")

        hw = 0
        for _ in range(num_words):
            #print(file_in.tell())
            hw += serial_analysis.hamming_weight(serial_analysis.file_read_next_hex4(file_in, binary_format))
        hw_percent = percent(hw, num_words*16)
        print(f"* Hamming weight = {hw} = {hw_percent}%")

        # Go to beginning of data dump again and find the min, max, and mode word value
        file_in.seek(0)
        serial_analysis.file_seek_next_data_dump(file_in, binary_format)
        all_data = serial_analysis.file_read_hex4_dump_as_words(file_in, num_words, binary_format)
        print(f"* First word = 0x{all_data[0]:04X}")
        print(f"* Last word = 0x{all_data[-1]:04X}")
        max_val = np.max(all_data)
        min_val = np.min(all_data)
        unique, counts = np.unique(all_data, return_counts=True)
        mode_index = np.argmax(counts)
        mode_val = unique[mode_index]
        mode_count = counts[mode_index]
        del all_data
        print(f"* max word value  = 0x{hex4_i(max_val)} = {max_val}")
        print(f"* min word value  = 0x{hex4_i(min_val)} = {min_val}")
        print(f"* mode word value = 0x{hex4_i(mode_val)} = {mode_val}")
        print(f"* mode word value occurences = {mode_count} = {percent(mode_count, num_words)}%")


if __name__ == '__main__':
    main()