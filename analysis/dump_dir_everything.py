import os, sys, matplotlib
import numpy as np
import matplotlib.pyplot as plt

from serial_analysis import *
from find_bit_strength import combine_captures_as_votes


def file_load_captures(file_in: TextIO, num_captures: int, num_words: int) -> np.ndarray:
    result = np.empty((num_captures, num_words), dtype="uint16")

    for i in range(num_captures):
        #print(f"Reading capture {i+1}/{num_captures}")
        file_seek_next_data_dump(file_in)
        for j in range(num_words):
            try:
                result[i, j] = file_read_next_hex4(file_in)
            except ValueError as e:
                print(f"(error in capture #{i}, word #{j}) at file position #{file_in.tell()}")
                raise e

    return result


def create_votes_np(captures: np.ndarray) -> np.ndarray:
    num_captures = captures.shape[0]
    num_words = captures.shape[1]
    num_bits = num_words * BITS_PER_WORD

    capture_votes = np.zeros((num_captures, num_bits), dtype="uint8")

    shifts = np.tile(np.arange(BITS_PER_WORD)[::-1], reps=num_words)

    for c in range(num_captures):
        #print(f"Combing capture {c + 1}/{num_captures}")
        cap = captures[c].repeat(BITS_PER_WORD)
        capture_votes[c] = (cap >> shifts) & 1

    return np.sum(capture_votes, axis=0)


def create_puf_np(bit_votes_for_1: np.ndarray, threshold: int) -> np.ndarray:
    "Take an array of each bit's number of votes for powering-up to a value of 1, and convert it to an array of (multi-bit) words"
    num_bits = bit_votes_for_1.shape[0]
    num_words = num_bits // BITS_PER_WORD

    bits = (bit_votes_for_1 > threshold).reshape((num_words, BITS_PER_WORD))

    # Reference: https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
    #return bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))

    return np.packbits(bits, bitorder="big").view(np.uint16).byteswap(inplace=True)


def show_binary_image_from_data(ax, ndarray):
    '''A "binary image" is a black-and-white only 2D image (no grayscale) and this function displays one'''
    ax.set_xticks([])
    ax.set_yticks([])
    # These 'if' tests below are only needed because otherwise an all-white image shows up as all-black for some reason
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

    num_bits = num_words * BITS_PER_WORD

    # create numpy vectorized function for later
    hamming_weight_vec = np.vectorize(hamming_weight)

    print("Loading data")
    with open(in_path, "r") as hex_dump_in:
        try:
            captures_data = file_load_captures(hex_dump_in, num_captures=num_captures, num_words=num_words)
        except ValueError:
            print("File has invalid data")
            return
    
    print("Combining captures")
    captures_bit_votes = create_votes_np(captures_data)

    captures_data_file_name = os.path.join(out_path, f"Captures-{num_captures}.npy")
    np.save(captures_data_file_name, captures_data, allow_pickle=False)

    # Create file with list of Hamming Weights for each dump...
    # Get each dump's Hamming weight.
    print(f"Reading Hamming weights for each dump ({num_captures} of them)")
    hweights = np.empty(num_captures)
    for c in range(num_captures):
        hweights[c] = np.sum(hamming_weight_vec(captures_data[c]))
    hweight_avg = float(np.average(hweights))
    hweight_avg_p = percent(hweight_avg, num_bits)
        
    # Save Hamming weights to a new file
    print("Saving Hamming weights")
    hweights_file = os.path.join(out_path, "Hamming-weights.txt")
    with open(hweights_file, "w") as hweights_file_out:
        for i, hw in enumerate(hweights):
            print(i, int(hw), percent(hw, num_bits), file=hweights_file_out)
        print(f"Average Hamming weight = {hweight_avg} = {hweight_avg_p:.3f}%", file=hweights_file_out)

    # Create bit stability/strengh vote figures
    # for num_votes in (11, 25, 49):
    for num_votes in (50,):
        # Try different amounts of max votes, but only as many as are in the actual data
        if num_votes > num_captures:
            # There aren't enough captures to have this many votes
            print(f"Not doing the {num_votes}-capture bit stability")
            break

        print(f"Creating bit stability/strengh vote histogram figures ({num_votes} votes)")

        votes_data_path = os.path.join(out_path, "Votes.npy")
        np.save(votes_data_path, captures_bit_votes, allow_pickle=False)
        
        # max_votes_num = np.max(binary_votes)
        max_votes_num = num_votes

        f = plt.figure()
        ax = f.gca()
        ax.set_title("Histogram of power-up 1's")
        ax.set_xlabel("Times appeared as a 1")
        ax.set_ylabel("Bits")
        # ax.hist(binary_votes, max_votes_num + 1, align='mid')
        ax.hist(captures_bit_votes, max_votes_num)
        f.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}.png"))

        vote_occurrences_path = os.path.join(out_path, f"Bit-Stability-{num_votes}-Bins.txt")
        with open(vote_occurrences_path, "w") as f_out:
            print("Bit stability vote occurrences:", file=f_out)
            occ = np.bincount(captures_bit_votes)
            for i, n in enumerate(occ):
                print(f"{i}: {n} bits = {percent(n, num_bits):3f}%", file=f_out)

            print(file=f_out)

            n_stable = occ[0] + occ[-1]
            n_unstable = sum(occ[1:-1])
            assert n_stable + n_unstable == num_bits, "should add up"
            print(f"Number of stable bits = {n_stable} = {percent(n_stable, num_bits):.3f}%", file=f_out)
            print(f"Number of unstable bits = {n_unstable} = {percent(n_unstable, num_bits):.3f}%", file=f_out)

    # Create PUF file
    print("Creating gold PUF")
    gold_puf_fname1 = os.path.join(out_path, "Gold-PUF.txt")
    gold_puf_fname2 = os.path.join(out_path, "Gold-PUF.npy")
    gold_puf_num_captures = num_captures
    # Force the number of captures for the gold PUF to be odd (by excluding the last one)
    if gold_puf_num_captures % 2 == 0:
        gold_puf_num_captures -= 1
    gold_puf_data = create_puf_np(captures_bit_votes, (gold_puf_num_captures + 1) // 2)
    # Save Hex dump file
    with open(gold_puf_fname1, "w") as f_out:
        file_write_words(f_out, gold_puf_data)
    # Save numpy data file
    np.save(gold_puf_fname2, gold_puf_data)

    # Get gold PUF Hamming Weight (to add this info to the images below)
    puf_hweight = np.sum(hamming_weight_vec(gold_puf_data))
    puf_hweight_p = percent(puf_hweight, num_bits)

    # Get Hamming Distance between PUF and the dumps
    print("Calculating Hamming distances for each dump and the gold PUF")
    hdistances = np.empty(num_captures)
    for i in range(num_captures):
        hd = np.sum(hamming_weight_vec(np.bitwise_xor(captures_data[i], gold_puf_data)))
        hdistances[i] = hd
    hdistances_avg = float(np.average(np.array(hdistances)))
    hdistances_avg_p = percent(hdistances_avg, num_bits)
    # Save Hamming weights to a new file
    print("Saving Hamming distances")
    hdistances_file = os.path.join(out_path, "Hamming-distances.txt")
    with open(hdistances_file, "w") as hdistances_file_out:
        for i, d in enumerate(hdistances):
            print(f"{i}: {d} bits = {percent(d, num_bits):.3f}%", file=hdistances_file_out)
        print(f"Average Hamming distance = {hdistances_avg} = {hdistances_avg_p:.3f}%", file=hdistances_file_out)

    # Create salt-and-pepper sample image from first few bits of the gold PUF 
    for size in (64, 128, 256, 512):
        print(f"Creating {size}x{size} salt-and-pepper image")
        # Use the file instead of the loaded data because its easy to re-use the file_read_image function
        with open(gold_puf_fname1, "r") as f_in:
            img_data = file_read_image(f_in, size, size)
        f = plt.figure()
        ax = f.gca()
        ax.set_title(f"Gold PUF {size}x{size}")
        ax.set_xlabel(f"HW = {puf_hweight_p:.3f}%")
        show_binary_image_from_data(ax, img_data)
        f.savefig(os.path.join(out_path, f"Salt-and-pepper-{size}x{size}.png"))


def run(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f"{sys.argv[0]}: error: \"{input_dir}\" is not a directory")
        return
    
    if not os.path.isdir(output_dir):
        print(f"{sys.argv[0]}: error: path \"{output_dir}\" is not a directory")
        return

    # num_captures = 3
    # num_captures = 11
    num_captures = 50
    num_words = NUM_WORDS

    listing = list(os.listdir(input_dir))
    i = 0
    for in_fname in listing:
        # Log
        print(f"({i + 1}): Processing \"{in_fname}\"")

        # Construct the string for the full input file path
        in_path = os.path.join(input_dir, in_fname)

        if not os.path.isfile(in_path):
            print(f"Skipping non-file {in_fname}")
            continue
        
        # Get input filename before the extension to use to automatically create the output folder name
        in_fname_no_ext, _ = in_fname.rsplit('.', maxsplit=1)

        if not os.path.isfile(in_path):
            continue

        if "binary" in in_fname:
            print("Skipping 'binary' file")
            continue

        # Create a new folder to hold all of the results for processing this file
        out_path = os.path.join(output_dir, in_fname_no_ext)
        print(f"Will save output in directory \"{out_path}\"")
        os.mkdir(out_path)

        run1(in_path, out_path, num_captures, num_words)
        i += 1


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input-directory output-directory")
        exit(1)

    run(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()