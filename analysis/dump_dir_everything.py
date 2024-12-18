from typing import Any
import os, sys, random
from timeit import default_timer as timer
import numpy as np
from scipy.stats import stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from serial_analysis import *


# Convert Hamming Weight function to numpy vectorized function for later
hw_vec_fn = np.vectorize(hamming_weight)


def show_binary_image_from_data(ax, ndarray, color0="black", color1="white", interpolation="nearest"):
    '''A "binary image" is a black-and-white only 2D image (no grayscale) and this function displays one'''
    ax.set_xticks([])
    ax.set_yticks([])
    # These 'if' tests below are only needed because otherwise an all-white image shows up as all-black for some reason
    # (thanks, matplotlib)
    if len(unique_arr := np.unique(ndarray)) == 1:
        if unique_arr[0] == 1:
            return ax.imshow(ndarray, ListedColormap([color1]))
        else:
            return ax.imshow(ndarray, ListedColormap([color0]))
    else:
        # ideally, this is the only branch we need (but it is not, see the above comment)
        return ax.imshow(ndarray, ListedColormap([color0, color1]), interpolation=interpolation)


def create_salt_and_pepper_fig(file_path: str, puf_data_words: np.ndarray, width: int, height: int, title: str, start_word_address: int = 0):
    print(f"Creating {width}x{height} salt & pepper image \"{title}\"")
    a0 = start_word_address * BITS_PER_WORD
    bit_count = width * height
    puf_data_bits = words_to_bits_np(puf_data_words)
    img_data = puf_data_bits[a0 : a0 + bit_count].reshape((width, height))
    f, ax = plt.subplots()
    # hw = np.sum(hw_vec_fn(img_data))
    # ax.set_xlabel(f"HW = {percent(hw, bit_count):.3f}%")
    big = width > 512 or height > 512
    if big:
        plt.gcf().set_size_inches(11,8.5)
    show_binary_image_from_data(ax, img_data, interpolation="bilinear" if big else "none")
    ax.set_title(title)
    ax.set_xticks(np.arange(0, width+1, width//8))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticks(np.arange(0, height+1, height//8))
    plt.savefig(file_path)
    plt.close()


def create_heatmap_fig(file_path: str, bit_vote_counts: np.ndarray, width: int, height: int, title: str, start_word_address: int = 0):
    print(f"Creating {width}x{height} bit bias heatmap image \"{title}\"")
    a0 = start_word_address * BITS_PER_WORD
    bit_count = width * height
    img = np.reshape(bit_vote_counts[a0 : a0 + bit_count], (width, height))
    big = width > 512 or height > 512
    plt.imshow(img, "Blues", interpolation="bilinear" if big else "none")
    if big:
        plt.gcf().set_size_inches(11,8.5)
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xticks(np.arange(0, width+1, width//8))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticks(np.arange(0, height+1, height//8))
    plt.colorbar(label='Bit vote count (normalized)') 
    plt.savefig(file_path)
    plt.close()


def create_heatmap_fig_2(file_path: str, bit_vote_counts: np.ndarray, width: int, height: int, title: str, start_word_address: int = 0, cmap="Blues"):
    print(f"Creating {width}x{height} bit bias heatmap image \"{title}\"")
    a0 = start_word_address * BITS_PER_WORD
    bit_count = width * height
    modified_arr = 2 * np.abs(bit_vote_counts - 0.5)
    img = np.reshape(modified_arr[a0 : a0 + bit_count], (width, height))
    big = width > 512 or height > 512
    if big:
        plt.imshow(img, cmap, interpolation="bilinear")
        plt.gcf().set_size_inches(11,8.5)
    else:
        plt.imshow(img, cmap, interpolation="none")
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xticks(np.arange(0, width, width//8))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticks(np.arange(0, height, height//8))
    plt.colorbar(label='Bit stability (normalized)') 
    plt.savefig(file_path)
    plt.close()


def run1(in_path: str, out_path: str, num_captures: int, num_words: int):
    '''Process one multi-dump input file and generate the relevant report files into the out_path directory'''

    if not os.path.isfile(in_path):
        print(f"ERROR: path \"{in_path}\" is not a valid path for a file")
        return
    if not os.path.isdir(out_path):
        print(f"ERROR: path \"{out_path}\" is not a valid path for a directory")
        return

    num_bits = num_words * BITS_PER_WORD

    # Extract chip name from file path
    name_match = re.search(r"(\w+)-([0-9a-z]+)nm-([0-9a-z]+)", in_path, re.IGNORECASE)
    if name_match is None:
        raise ValueError("Could not find the chip name in file path")
    name_whole = name_match.group(0)
    name_man = name_match.group(1)
    name_nm = name_match.group(2)
    name_num = name_match.group(3)

    # Check if pre-loaded captures numpy file already exists
    captures_npy_file_exists = False
    for reduced_captures_num in range(num_captures, 5, -1):
        captures_data_file_name = os.path.join(out_path, f"Captures-{reduced_captures_num}.npy")
        captures_npy_file_exists = os.path.isfile(captures_data_file_name)
        if captures_npy_file_exists:
            break

    print("Loading data")
    if captures_npy_file_exists:
        print("Found numpy data file to load instead")
        captures_data = np.load(captures_data_file_name)
    else:
        with open(in_path, "r") as hex_dump_in:
            try:
                captures_data = file_load_captures_fallback(hex_dump_in, num_captures=num_captures, num_words=num_words)
            except ValueError:
                print("File has invalid data")
                return
        
    loaded_num_captures = captures_data.shape[0]
    if loaded_num_captures != num_captures:
        print(f"NOTE: only able to load {loaded_num_captures}/{num_captures} captures!")
        print(f"Continuing with {loaded_num_captures} captures...")
        num_captures = loaded_num_captures
    
    print("Combining captures")
    captures_bit_votes = create_votes_np(captures_data)

    # Save captures file
    if not captures_npy_file_exists:
        np.save(captures_data_file_name, captures_data, allow_pickle=False)

    # Create file with list of Hamming Weights for each dump...
    # Get each dump's Hamming weight.
    print(f"Reading Hamming weights for each dump ({num_captures} of them)")
    hweights = np.empty(num_captures)
    for c in range(num_captures):
        hweights[c] = np.sum(hw_vec_fn(captures_data[c]))
    hweight_avg = float(np.average(hweights))
    hweight_avg_p = percent(hweight_avg, num_bits)
        
    # Save Hamming weights to a new file
    print("Saving Hamming weights")
    hweights_file = os.path.join(out_path, "Hamming-weights.txt")
    with open(hweights_file, "w") as hweights_file_out:
        for i, hw in enumerate(hweights):
            print(i, int(hw), percent(hw, num_bits), file=hweights_file_out)
        print(f"Average Hamming weight = {hweight_avg} = {hweight_avg_p:.3f}%", file=hweights_file_out)

    # Create bit stability/strength vote figures
    for num_votes in np.unique([40, num_captures]):
        # Try different amounts of max votes, but only as many as are in the actual data
        if num_votes > num_captures:
            # There aren't enough captures to have this many votes
            print(f"Not doing the {num_votes}-capture bit stability")
            continue

        print(f"Creating bit stability/strength vote histogram figures ({num_votes} votes)")

        votes_data_path = os.path.join(out_path, f"Votes-{num_votes}.npy")
        np.save(votes_data_path, captures_bit_votes, allow_pickle=False)
        
        # max_votes_num = np.max(binary_votes)
        max_votes_num = num_votes

        # Make linear plot
        f = plt.figure()
        ax = f.gca()
        ax.set_title(f"{name_whole} Histogram of power-up 1's")
        ax.set_xlabel("Times appeared as a 1")
        ax.set_ylabel("Bits")
        # ax.hist(binary_votes, max_votes_num + 1, align='mid')
        ax.hist(captures_bit_votes, max_votes_num)
        f.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}.png"))
        plt.close()

        # Make log plot
        f = plt.figure()
        ax = f.gca()
        ax.set_title("Histogram of power-up 1's")
        ax.set_xlabel("Times appeared as a 1")
        ax.set_ylabel("Bits")
        ax.set_yscale('log')
        # ax.hist(binary_votes, max_votes_num + 1, align='mid')
        ax.hist(captures_bit_votes, max_votes_num)
        f.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}-log.png"))
        plt.close()

        # Make dual axis log plot
        middle_index = num_votes // 2
        # print(f"middle index is {middle_index}")
        # captures_bit_votes_hist, caps_edges = np.histogram(captures_bit_votes, bins=num_votes)
        captures_bit_votes_0 = captures_bit_votes[np.where(captures_bit_votes < middle_index)]
        captures_bit_votes_1 = captures_bit_votes[np.where(captures_bit_votes >= middle_index)]
        ax1 = plt.subplot(211)
        ax1.set_xlim(0, middle_index)
        ax1.invert_xaxis()    
        ax1.set_title("Histogram of power-up 1's")
        ax1.set_ylabel("0 Bits (log)")
        ax1.set_yscale('log')
        ax1.hist(captures_bit_votes_0, num_votes//2, color="r")
        ax2 = plt.subplot(212, sharey=ax1)
        ax2.set_xlim(middle_index, num_votes)
        ax2.set_ylabel("1 Bits (log)")
        ax2.set_yscale('log')
        ax2.hist(captures_bit_votes_1, num_votes//2, color="b")
        # ax2.invert_yaxis()
        plt.savefig(os.path.join(out_path, f"Binary-votes-out-of-{num_votes}-dual.png"))
        plt.close()

        # Save actual numerical values to a text file
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
    gold_puf_fname1 = os.path.join(out_path, "Gold-PUF.txt")
    gold_puf_npy_fname = os.path.join(out_path, "Gold-PUF.npy")
    gold_puf_num_captures = num_captures
    if os.path.isfile(gold_puf_fname1):
        gold_puf_data = np.load(gold_puf_npy_fname)
    else:
        print("Creating gold PUF")
        # Force the number of captures for the gold PUF to be odd (by excluding the last one)
        if gold_puf_num_captures % 2 == 0:
            gold_puf_num_captures -= 1
        gold_puf_data = create_puf_np(captures_bit_votes, (gold_puf_num_captures + 1) // 2)
        # Save Hex dump file
        with open(gold_puf_fname1, "w") as f_out:
            file_write_words(f_out, gold_puf_data)
        # Save numpy data file
        np.save(gold_puf_npy_fname, gold_puf_data)

    # Get gold PUF Hamming Weight (to add this info to the images below)
    puf_hweight = np.sum(hw_vec_fn(gold_puf_data))
    puf_hweight_p = percent(puf_hweight, num_bits)

    # Get Hamming Distance between PUF and the dumps
    print("Calculating Hamming distances for each dump and the gold PUF")
    hdistances = np.empty(num_captures)
    for i in range(num_captures):
        hd = np.sum(hw_vec_fn(np.bitwise_xor(captures_data[i], gold_puf_data)))
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

    # Create different sample images from the gold PUF
    sizes = (64, 128, 256, 512, 1024)
    offset = 0
    offset_str = f"0x{offset:X}"
    for size in sizes:
        # "Salt and pepper" image
        s_and_p_path = os.path.join(out_path, f"Salt-and-pepper-{size}x{size}-at-{offset_str}.png")
        create_salt_and_pepper_fig(s_and_p_path, gold_puf_data, size, size, f"{name_whole} Gold PUF {size}x{size} at {offset_str}", offset)
        # Heatmap image
        heatmap_path = os.path.join(out_path, f"Heatmap-{size}x{size}-at-{offset_str}.png")
        create_heatmap_fig(heatmap_path, captures_bit_votes/num_captures, size, size, f"{name_whole} Bit State Heatmap at {offset_str}", offset)

    # Full-chip images...
    full_size = int(NUM_BITS**0.5)

    # Full-chip salt-and-pepper image
    s_and_p_path = os.path.join(out_path, f"Salt-and-pepper-full.png")
    create_salt_and_pepper_fig(s_and_p_path, gold_puf_data, full_size, full_size, f"{name_whole} Gold PUF")

    # Full-chip Heatmap image
    heatmap_path = os.path.join(out_path, f"Heatmap-full.png")
    create_heatmap_fig(heatmap_path, captures_bit_votes/num_captures, full_size, full_size, f"{name_whole} Bit State Heatmap")

    # Modified heatmap image(s)
    highlight_color = np.array([0.05, 0.05, 0.95])
    colors_last = np.ones((10, 3))
    colors_butlast = np.ones((10, 3))
    colors_half = np.ones((10, 3))
    colors_last[-1] = highlight_color
    colors_butlast[:-1] = highlight_color
    colors_half[5:] = highlight_color
    views: dict[str, Any] = {
        "Heatmap-rescale-continuous.png": "Blues",
        "Heatmap-rescale-stable.png": ListedColormap(colors_last),
        "Heatmap-rescale-half-stable.png": ListedColormap(colors_half),
        "Heatmap-rescale-unstable.png": ListedColormap(colors_butlast),
    }
    for file_name, cmap in views.items():
        heatmap_path_2 = os.path.join(out_path, file_name)
        create_heatmap_fig_2(heatmap_path_2, captures_bit_votes/num_captures, full_size, full_size, f"{name_whole} Stable Bit Heatmap", cmap=cmap)


def main_dumpfile():
    if len(sys.argv) > 1:
        dump_filename = sys.argv[1]
    else:
        dump_filename = input("enter file name: ")

    with open(dump_filename, 'rb') as file_in:
        num_words = file_seek_next_data_dump_and_count_it(file_in)
        print(f"* data word count = {num_words}")

        hw = 0
        for _ in range(num_words):
            hw += hamming_weight(file_read_next_hex4(file_in))
        hw_percent = percent(hw, num_words)
        print(f"* Hamming weight = {hw} = {hw_percent}%")

        # Go to beginning of data dump again and find the min, max, and mode word value
        file_seek_next_data_dump_and_count_it(file_in)
        all_data = file_read_hex4_dump_as_words(file_in, num_words)
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


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} input-dump-file output-directory [num-captures]")
        exit(1)

    num_captures = 50
    if len(sys.argv) > 3:
        num_captures = int(sys.argv[3])

    os.mkdir(sys.argv[2])
    run1(sys.argv[1], sys.argv[2], num_captures=num_captures, num_words=NUM_WORDS)


hw_vec = np.vectorize(hamming_weight)


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
    

def file_list_diff_print(num_words, file_list: list[str]) -> None:
    n = 0
    for i, path_a in enumerate(file_list):
        # Only compare this element to elements after this one, because order doesn't matter.
        # I.E. HD(a, b) == HD(b, a)
        with open(path_a, "r") as file_a:
                data_a = file_load_capture(file_a, num_words)

        for path_b in file_list[i + 1:]:
            path_a_short = shorten(path_a)
            path_b_short = shorten(path_b)
            print(f"{n + 1}: '{path_a_short}' (X) '{path_b_short}'...", flush=True)

            with open(path_b, "r") as file_b:
                data_b = file_load_capture(file_b, num_words)

            diff = np.sum(hw_vec(np.bitwise_xor(data_a, data_b)))
            p = percent(diff, num_words * BITS_PER_WORD)
            print(f"* HD = {p:.3f}%")

            c = stats.pearsonr(data_a, data_b).correlation
            print(f"* Pearson correlation = {c:.4f}")
            n += 1


def file_list_diff_save(num_words, file_list: list[str], cmat_fname: str, hdmat_fname: str) -> None:    
    """
    `cmat_fname`: correlation matrix filename
    `hdmat_fname`: Hamming distance matrix filename
    """
    num_bits = num_words * BITS_PER_WORD

    n = len(file_list)
    mat_hd = np.zeros((n, n))
    mat_c = np.zeros((n, n))

    # Rows [0...n-1]
    # Cols [1...n]

    for i in range(n):
        path_a = file_list[i]
        data_a = np.load(path_a)
        path_a_short = shorten(path_a)
        
        for j in range(n):
            if (j-1 >= i): continue

            path_b = file_list[j]
            data_b = np.load(path_b)
            path_b_short = shorten(path_b)

            print(f"Comparing {path_a_short} with {path_b_short}")

            mat_hd[i, j] = np.sum(hw_vec(np.bitwise_xor(data_a, data_b))) / num_bits
            mat_c[i, j] = stats.pearsonr(data_a, data_b).correlation

    np.save(hdmat_fname, mat_hd)
    np.save(cmat_fname, mat_c)


def plot_correlation_matrix(file_list: list[str], file_path: str):
    mat_c = np.load(file_path)
    labels = list(map(shorten, file_list))
    ai = plt.matshow(mat_c)
    ax = plt.gca()
    plt.colorbar(ai)
    ax.xaxis.tick_bottom()
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45)
    # ax.set_xticklabels()
    ax.axis("image")
    ax.set_title("Gold PUF Pearson Correlation Matrix")
    plt.show()


def shorten(path: str) -> str:
    m = re.search(r"\\([^\\-]+-[^\\-]*nm-[^\\-]*)", path)
    if not m:
        raise ValueError()
    return m.group(1)


def main_gold_puf_grid():
    file_list = sys.argv[1:]

    file_list = [
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\RT_maybe-30s-50dumps-2024.10.22.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\RT-30s-50dumps-2024.10.21.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-3\RT-30s-50dumps-2022.10.22.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-Z\RT-30s-50dumps.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-1\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-2\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-2\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-3\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-4\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-5\RT-30s-50dumps.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-1-rad\RT-15s-20dumps.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\IDP-130nm-1\50_captures_15_second_delay.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\RT-30s-50dumps-2024.10.22.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\RT-30s-50dumps-2024.10.22.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\2024.10.22-normal-30s-50dumps.txt-results\Gold-PUF.npy",
        r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-rad\RT-30s-20dumps.txt-results\Gold-PUF.npy",
    ]

    if not file_list:
        file_list = ask_file_list()

    print("Input files:")
    for f in file_list:
        print(f"- '{f}'")

    n = len(file_list)
    num_combos = (n * (n - 1)) // 2
    print(f"Comparing Hamming Distance of each file against each other file ({num_combos} combinations) ...")

    # file_list_diff_print(NUM_WORDS, file_list)
    fname1 = "Correlation-matrix.npy"
    fname2 = "HD-matrix.npy"
    # file_list_diff_save(NUM_WORDS, file_list, fname1, fname2)
    plot_correlation_matrix(file_list, fname1)

    print("Done with Hamming Distances")


if __name__ == '__main__':
    if len(sys.argv) != 1:
        main()
    else:
        paths = [
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\RT_maybe-30s-50dumps-2024.10.22.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\0C-30s-50dumps-2024.10.23.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\0C-240s-50dumps-2024.10.25.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\30C-30s-50dumps-2024.11.05.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-1\50C-30s-50dumps-2024.10.22.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\50C-30s-50dumps-2024.10.21.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\RT-30s-50dumps-2024.10.21.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\0C-30s-50dumps-2024.10.23.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\0C-240s-50dumps-2024.10.25.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-2\30C-30s-50dumps-2024.11.05.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-3\0C-30s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-3\30C-60s-50dumps-2024.11.05.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-3\50C-30s-50dumps-2024.10.24.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-65nm-3\RT-30s-50dumps-2022.10.22.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-1\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-1-rad\RT-15s-20dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-2\50_captures_15_second_delay_cap.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-2\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-LP-1\50C-60s-50dumps-2024.11.08.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-LP-1\0C-240s-50dumps-2024.11.01.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-90nm-Z\RT-30s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-2\50_captures_15_second_delay_cap.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-2\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-3\50_captures_15_second_delay_cap.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-3\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-4\50_captures_15_second_delay_cap.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-4\35_captures_15_second_delay_cap.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-4\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-150nm-5\RT-30s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\RT-30s-50dumps-2024.10.22.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\0C-30s-50dumps-2024.10.23.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\0C-240s-50dumps-2024.10.25.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\30C-30s-24dumps-2024.10.31.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\30C-60s-50dumps-2024.11.01.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\50C-30s-50dumps-2024.10.22-incomplete.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-1\50C-60s-50dumps-2024.11.08.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\RT-30s-50dumps-2024.10.22.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\0C-30s-50dumps-2024.10.23.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\0C-240s-50dumps-2024.10.25.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\30C-60s-25dumps-2024.10.31.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\30C-60s-50dumps-2024.11.01.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-2\50C-30s-50dumps-2024.10.21.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\2024.10.22-normal-30s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\0C-30s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\0C-240s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\30C-60s-50dumps-2024.11.05.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-3\50C-30s-50dumps-2024.10.24.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-4\RT-60s-50dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\CY-250nm-rad\RT-30s-20dumps.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\IDP-130nm-1\50_captures_15_second_delay_inductor.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\IDP-130nm-1\50_captures_15_second_delay.txt",
            r"C:\Users\ihals\OneDrive - Colostate\RAM_Lab\Senior_Design\Data\IDP-130nm-1\50_captures_15_second_delay_cap.txt",
        ]

        # max_img_bit_size = 1024
        # random_offset = random.randint(0, NUM_WORDS - (max_img_bit_size**2)//BITS_PER_WORD)
        # print(f"Random offset = 0x{random_offset:X}")

        start1 = timer()
        for i, p in enumerate(paths):
            print(f"[{i+1}/{len(paths)}]: {p}")
            p2 = p + "-results"
            if not os.path.isdir(p2):
                os.mkdir(p2)
            start2 = timer()
            run1(in_path=p, out_path=p2, num_captures=50, num_words=NUM_WORDS)
            end2 = timer()
            duration2 = end2 - start2
            print(f"Elapsed time: {duration2:2.02f}")
            print()
        end1 = timer()
        duration1 = end1 - start1
        print(f"Total elapsed time: {duration1:2.02f}")