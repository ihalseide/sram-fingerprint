import numpy as np
from serial_analysis import *


def main():
    filename = input("input numpy array filename: ")
    num_trials = int(input("number of trials: "))

    bit1_counts = np.load(filename)
    num_words = bit1_counts.shape[0]
    print(f"File contains data for {num_words} words")

    for i, votes in enumerate(bit1_counts):
        if votes == 0:
            # No votes for bit being a 1, so it is a strong 0
            print(f"0x0x{i:04X} : strong 0")
        elif votes == num_trials:
            # All votes for bit being a 1, so it is a strong 1
            print(f"0x0x{i:04X} : strong 1")
        elif votes < 0 or votes > num_trials:
            print(f"ERROR: found that address 0x{i:04X} has a vote value of {votes}, which does not make sense for data from {num_trials} trials")
            break
        else:
            # Ignore "weak" bits
            pass


if __name__ == '__main__':
    main()