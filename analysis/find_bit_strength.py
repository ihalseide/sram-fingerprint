'''
This script takes a file that contains multiple dumps (probably generated from command #17 in the Arduino code)
and creates (and saves) a final array that represents the number of votes for a bit being 1 for each bit in the dump file.

Next, you can use the 'plot_bit_strength.py' script to plot the resulting array.
'''


import sys, os
import numpy as np
from serial_analysis import *


def combine_captures_as_votes(num_captures: int, input_file_name: str, num_words: int = NUM_WORDS) -> np.ndarray:
    '''
    Use majority voting on all of the bits from a file with multiple memory dumps.
    The file at 'input_file_name' should have multiple memory dumps (at least 'num_captures' of them) inside.
    '''
    assert num_captures % 2 != 0, "num_captures must be odd so there can be no voting ties"

    # print(f"input file name: \"{input_file_name}\"")
    # print(f"output file name: \"{output_file_name}\"")

    # Array to map the data's bit index to number of votes that it should be set to a '1'
    bit_votes_for_one = np.zeros(num_words * BITS_PER_WORD, dtype='b')

    print(f"- loading file \"{input_file_name}\"")
    with open(input_file_name, "rb") as file_in:
        for c in range(1, num_captures + 1):
            print(f"Processing capture #{c} in the file")

            if file_seek_next_data_dump(file_in) is None:
                raise ValueError(f"not enough memory dumps in input file '{input_file_name}': expected {num_captures} but stopped at {c}")

            # Read each data word from the data file
            for word_i in range(num_words):
                try:
                    word = file_read_next_hex4(file_in)
                except ValueError as e:
                    print(f"ERROR at word index = {word_i}")
                    raise e
                # Log the "bit vote" for each bit of the word
                for word_bit_i in range(BITS_PER_WORD):
                    bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                    # Test bit number 'bit_i' and increment a vote if it is set
                    if word & (1 << word_bit_i) != 0:
                        bit_votes_for_one[bit_i] += 1

    return bit_votes_for_one


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("filename: ")

    if len(sys.argv) > 2:
        num_captures = int(sys.argv[2])
    else:
        num_captures = int(input("number of captures in the file: "))

    if len(sys.argv) > 3:
        num_words = int(sys.argv[3])
    else:
        num_words = int(input("number of words per dump: "))

    if len(sys.argv) > 4:
        filename_out = sys.argv[4]
    else:
        filename_out = input("output filename: ")

    if os.path.isfile(filename_out):
        msg = f"output file \"{filename_out}\" exists, OK to overwrite? (y/n): "
        if input(msg).lower()[0] != 'y':
            print("Canceled")
            return
    
    # Allow shortcut for 0 -> NUM_WORDS
    if num_words == 0:
        print(f"Defaulting number of words to {NUM_WORDS=}")
        num_words = NUM_WORDS

    combined = combine_captures_as_votes(num_captures, filename, num_words)

    print(f"Saving combined bit votes Numpy array to file \"{filename_out}\"...")
    np.save(filename_out, combined)

    print("done")


if __name__ == '__main__':
    main()