import os, sys, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from serial_analysis import *
from find_bit_strength import combine_captures_as_votes
from convert_dump import convert as binary_dump_to_text


def show_binary_image_from_data(ax, ndarray):
    '''A "binary image" is a black-and-white only 2D image (no grayscale) and this function displays one'''
    ax.set_xticks([])
    ax.set_yticks([])
    # These 'if' tests are only needed because otherwise an all-white image shows up as all-black for some reason
    # (thanks, matplotlib)
    if len(unique_arr := np.unique(ndarray)) == 1:
        if unique_arr[0] == 1:
            ax.imshow(ndarray, matplotlib.colors.ListedColormap(['white']))
        else:
            ax.imshow(ndarray, matplotlib.colors.ListedColormap(['black']))
    else:
        # ideally, this is the only branch we need (but it is not, see the above comment)
        ax.imshow(ndarray, matplotlib.colors.ListedColormap(['black', 'white']))


def run(input_dir: str, output_dir: str) -> bool:
    if not os.path.isdir(input_dir):
        print(f"{sys.argv[0]}: error: \"{input_dir}\" is not a directory")
        return False
    
    if os.path.isdir(output_dir) or os.path.isfile(output_dir):
        print(f"{sys.argv[0]}: error: path \"{output_dir}\" already exists")
        return False
    
    # Create empty output directory
    os.mkdir(output_dir)

    num_captures = 11
    num_words = NUM_WORDS

    for i, in_fname in enumerate(os.listdir(input_dir)):
        # Log
        print(f"({i + 1}): Processing \"{in_fname}\"")

        # Construct the string for the full input file path
        in_path = os.path.join(input_dir, in_fname)

        # Create a new folder to hold all of the results for processing this file
        out_path = os.path.join(output_dir, in_fname)
        print(f"Will save output in directory \"{out_path}\"")
        os.mkdir(out_path)

        # Create plain text dump file
        print("Converting binary dump to plaintext hex dump")
        text_dump_path = os.path.join(out_path, "Hex-dumps.txt")
        binary_dump_to_text(in_path, text_dump_path, num_words)

        # Create PUF file
        print("Creating gold PUF")
        gold_puf_fname = os.path.join(out_path, "Gold-PUF.txt")
        create_gold_puf_v2(num_captures, text_dump_path, gold_puf_fname, num_words)

        # Get Hamming Weight
        hweight = file_hamming_weight(gold_puf_fname, num_words)
        hweight_p = percent(hweight, num_words * BITS_PER_WORD)

        # Create gold PUF salt-and-pepper sample image
        for size in (64, 128, 256, 512):
            print(f"Creating {size}x{size} salt-and-pepper image")
            img_data = file_read_image(gold_puf_fname, size, size)
            f = plt.figure()
            ax = f.gca()
            ax.set_title(f"Gold PUF {size}x{size}")
            ax.set_xlabel(f"HW = {hweight_p:.3f}%")
            show_binary_image_from_data(ax, img_data)
            f.savefig(os.path.join(out_path, f"Salt-and-pepper-{size}x{size}.png"))

        # Create bit stability/strengh vote figures
        for num_votes in (11, 25, 49):
            if num_votes > num_captures:
                # Try different amounts of max votes, but only as many as are in the actual data
                break

            print("Creating bit stability/strengh vote histogram figures")
            binary_votes = combine_captures_as_votes(num_captures, text_dump_path, num_words)
            
            max_votes_num = np.max(binary_votes)

            f = plt.figure()
            ax = f.gca()
            ax.set_title("Histogram of power-up 1's")
            ax.set_xlabel("Times appeared as a 1")
            ax.set_ylabel("Bits")
            ax.hist(binary_votes, max_votes_num + 1)
            f.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}.png"))

    return True


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input-directory output-directory")
        exit(1)

    if not run(sys.argv[1], sys.argv[2]):
        exit(1)


if __name__ == '__main__':
    main()