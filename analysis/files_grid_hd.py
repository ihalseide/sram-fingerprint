'''
Get the Hamming Distance between multiple files.
'''


import sys, os, math, re
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from serial_analysis import *


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
    main_gold_puf_grid()