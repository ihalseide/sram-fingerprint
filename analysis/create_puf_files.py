'''
Create a average PUF memory data file from a given file that contains multiple memory data dumps
'''


import sys, os
from serial_analysis import *
from files_grid_hd import ask_file_list


def main():
    print("This will take a list of input files and create the voted 'Gold PUF' file for each one. The new output files will be a prefix + the same name as the input file")

    num_dumps = int(input("Enter the number of data dumps to combine use from each file: "))

    suffix = input("Enter the prefix string to add to the input file name to create the output file name: ")

    if sys.argv[1:]:
        print("Using files from command line")
        input_files = sys.argv[1:]
    else:
        input_files = ask_file_list()

    num_files = len(input_files)

    print(f"Will process {num_files} files...")

    for i, filename_in in enumerate(input_files):
        filename_out = filename_in + suffix

        print(f"({i + 1}/{num_files}) '{filename_in}' => '{filename_out}'")

        # Don't overwrite files
        if os.path.exists(filename_out):
            print(f"Error: a file named '{filename_out}' already exists")
            exit(1)

        try:
            create_gold_puf_v2(num_dumps, filename_in, filename_out, NUM_WORDS, binary_dump_format=True)
        except ValueError as e:
            print(f"Encountered an error while reading '{filename_in}'...")
            print(type(e), e)
            print("\nContinuing with next file...")

    print("Done creating PUF files")


if __name__ == '__main__':
    main()