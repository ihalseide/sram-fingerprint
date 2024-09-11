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

    with open(dump_filename, 'rb') as file_in:
        num_words = serial_analysis.file_seek_next_data_dump_and_count_it(file_in)
        print(f"* data word count = {num_words}")

        hw = 0
        for _ in range(num_words):
            hw += serial_analysis.hamming_weight(serial_analysis.file_read_next_hex4(file_in))
        hw_percent = percent(hw, num_words)
        print(f"* Hamming weight = {hw} = {hw_percent}%")

        # Go to beginning of data dump again and find the min, max, and mode word value
        serial_analysis.file_seek_next_data_dump_and_count_it(file_in)
        all_data = serial_analysis.file_read_hex4_dump_as_words(file_in, num_words)
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
        print(f"* mode word value occurences = {mode_count} = {percent(mode_count, num_words)}% of the time")


if __name__ == '__main__':
    main()