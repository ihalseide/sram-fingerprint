'''
This script plots the output from the 'find_bit_strength.py' script.
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
from serial_analysis import *


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("input file name for saved Numpy array: ")

    votes = np.load(filename)
    max_votes_num = np.max(votes)

    print("Max votes value = ", max_votes_num)

    ax = plt.subplot()
    ax.set_title("Histogram of power-up 1's")
    ax.set_xlabel("Bit=1 votes")
    ax.set_ylabel("Occurences")
    ax.hist(votes, max_votes_num + 1)
    
    plt.show()


if __name__ == '__main__':
    main()