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
        ax.imshow(ndarray, matplotlib.colors.ListedColormap(['black', 'white']), interpolation="nearest")


def run1(in_path: str, out_path: str, num_captures: int, num_words: int):
    '''Process one multi-dump input file and generate the relevant report files into the out_path directory'''

    assert(os.path.isfile(in_path))
    assert(os.path.isdir(out_path))

    # Create plain text dump file
    # print("Converting binary dump to plaintext hex dump")
    # text_dump_path = os.path.join(out_path, "Hex-dumps.txt")
    # binary_dump_to_text(in_path, text_dump_path, num_words)

    # Create file with list of Hamming Weights for each dump...
    # Get each dump's Hamming weight.
    print("Reading Hamming weights for each dump")
    hweights = []
    with open(in_path, "r") as hex_dump_in:
        for _ in range(num_captures):
            file_seek_next_data_dump(hex_dump_in)
            hw = file_hamming_weight(hex_dump_in, num_words)
            hweights.append(hw)
    # Save Hamming weights to a new file
    print("Saving Hamming weights")
    hweights_file = os.path.join(out_path, "Hamming-weights.txt")
    with open(hweights_file, "w") as hweights_file_out:
        for i, hw in enumerate(hweights):
            print(i, hw, percent(hw, num_words * BITS_PER_WORD), file=hweights_file_out)

    # Create PUF file
    print("Creating gold PUF")
    gold_puf_fname = os.path.join(out_path, "Gold-PUF.txt")
    gold_puf_num_captures = num_captures
    if gold_puf_num_captures % 2 == 0:
        # Force the number of captures for the gold PUF to be odd (by excluding the last one)
        gold_puf_num_captures -= 1
    create_gold_puf_v2(gold_puf_num_captures, in_path, gold_puf_fname, num_words)

    # Get Hamming Distance between PUF and the dumps
    print("Calculating Hamming distances for each dump and the gold PUF")
    hdistances = []
    gold_puf_data = file_read_hex4_dump_as_words(gold_puf_fname, num_words)
    # create numpy vectorized function for later
    hamming_weight_vec = np.vectorize(hamming_weight)
    with open(in_path, "rb") as hex_dump_in:
        for _ in range(num_captures):
            file_seek_next_data_dump(hex_dump_in)
            dump_data = file_read_hex4_dump_as_words(hex_dump_in, num_words)
            hd = np.sum(hamming_weight_vec(np.bitwise_xor(gold_puf_data, dump_data)))
            hdistances.append(hd)
    # Save Hamming weights to a new file
    print("Saving Hamming distances")
    hdistances_file = os.path.join(out_path, "Hamming-distances.txt")
    with open(hdistances_file, "w") as hdistances_file_out:
        for i, d in enumerate(hdistances):
            print(i, d, percent(d, num_words * BITS_PER_WORD), file=hdistances_file_out)

    # Get gold PUF Hamming Weight (to add this info to the images below)
    puf_hweight = file_hamming_weight(gold_puf_fname, num_words)
    puf_hweight_p = percent(puf_hweight, num_words * BITS_PER_WORD)

    # Create salt-and-pepper sample image from first few bits of the gold PUF 
    for size in (64, 128, 256, 512):
        print(f"Creating {size}x{size} salt-and-pepper image")
        img_data = file_read_image(gold_puf_fname, size, size)
        f = plt.figure()
        ax = f.gca()
        ax.set_title(f"Gold PUF {size}x{size}")
        ax.set_xlabel(f"HW = {puf_hweight_p:.3f}%")
        show_binary_image_from_data(ax, img_data)
        f.savefig(os.path.join(out_path, f"Salt-and-pepper-{size}x{size}.png"))

    # Create bit stability/strengh vote figures
    # for num_votes in (11, 25, 49):
    for num_votes in (49,):
        # Try different amounts of max votes, but only as many as are in the actual data
        if num_votes > num_captures:
            # There aren't enough captures to have this many votes
            break

        print(f"Creating bit stability/strengh vote histogram figures ({num_votes} votes)")
        binary_votes = combine_captures_as_votes(num_votes, in_path, num_words)
        
        # max_votes_num = np.max(binary_votes)
        max_votes_num = num_votes

        f = plt.figure()
        ax = f.gca()
        ax.set_title("Histogram of power-up 1's")
        ax.set_xlabel("Times appeared as a 1")
        ax.set_ylabel("Bits")
        ax.hist(binary_votes, max_votes_num + 1, align='mid')
        f.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}.png"))


def run(input_dir: str, output_dir: str) -> bool:
    if not os.path.isdir(input_dir):
        print(f"{sys.argv[0]}: error: \"{input_dir}\" is not a directory")
        return False
    
    if not os.path.isdir(output_dir):
        print(f"{sys.argv[0]}: error: path \"{output_dir}\" is not a directory")
        return False

    # num_captures = 11
    num_captures = 49
    num_words = NUM_WORDS

    no_failures = True

    listing = list(os.listdir(input_dir))
    i = 0
    for in_fname in listing:
        # Log
        print(f"({i + 1}): Processing \"{in_fname}\"")

        # Get input filename before the extension to use to automatically create the output folder name
        in_fname_no_ext, _ = in_fname.rsplit('.', maxsplit=1)

        # Construct the string for the full input file path
        in_path = os.path.join(input_dir, in_fname)

        if not os.path.isfile(in_path):
            continue

        # Create a new folder to hold all of the results for processing this file
        out_path = os.path.join(output_dir, in_fname_no_ext)
        print(f"Will save output in directory \"{out_path}\"")
        os.mkdir(out_path)

        try:
            run1(in_path, out_path, num_words, num_captures)
            i += 1
        except (ValueError, UnicodeError) as e:
            print(f"({i + 1}): Encountered error when processing \"{in_fname}\"...")
            print(e)
            no_failures = False
            i += 1
            continue

    return no_failures


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input-directory output-directory")
        exit(1)

    if not run(sys.argv[1], sys.argv[2]):
        exit(1)


if __name__ == '__main__':
    main()