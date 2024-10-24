'''
Get the Hamming Distance between multiple files.
'''


import sys, os, math
import numpy as np
from serial_analysis import *


def ask_file_list() -> list[str]:
    result = []

    base_path = input("Enter file directory base path (and maybe the common prefix for the file names): ")
    print("Add files to compare, one by one (entering a blank line will stop adding file)...")

    while f := input("add file: "):
        # This is intentionally string concatenation, not path concatenation!
        path = base_path + f

        if os.path.isfile(path):
            result.append(base_path + f)
            print(f"Added '{path}'")
        else:
            print(f"Skipped adding '{path}' because it is NOT a valid file path")
    
    print("(end of list)")

    return result


def main():
    file_list = sys.argv[1:]

    if not file_list:
        file_list = ask_file_list()

    print("Input files:")
    for f in file_list:
        print(f"- '{f}'")

    n = len(file_list)
    num_combos = (n * (n - 1)) // 2
    print(f"Comparing Hamming Distance of each file against each other file ({num_combos} combinations) ...")

    file_list_diff_print(NUM_WORDS, file_list)

    print("Done with Hamming Distances")
    

def file_list_diff_print(num_words, file_list: list[str]) -> None:
    n = 0
    for i, path_a in enumerate(file_list):
        # Only compare this element to elements after this one, because order doesn't matter.
        # I.E. HD(a, b) == HD(b, a)
        for path_b in file_list[i + 1:]:
            print(f"{n + 1}: '{path_a}' (X) '{path_b}' = ", end='', flush=True)
            diff = bit_diff_files(path_a, path_b, num_words, do_print=False)
            p = percent(diff, num_words * BITS_PER_WORD)
            print(f"{p:.3f}%", flush=True)
            n += 1


if __name__ == '__main__':
    main()