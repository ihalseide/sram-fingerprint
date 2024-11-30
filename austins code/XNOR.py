import os
import re
import numpy as np

# Directory containing the files to compare
directory_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps"

BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

def read_hex_file_to_binary_array(file_path, num_words, bits_per_word):
    '''
    Reads a file containing hex words and converts it into a binary array.
    Each hex word is expanded to a 16-bit binary representation.
    '''
    binary_array = []
    try:
        with open(file_path, 'r') as file_in:
            for line in file_in:
                hex_words = HEX4_WORD_PATTERN.findall(line)
                for hex_word in hex_words:
                    # Convert hex to binary (padded to 16 bits)
                    word_binary = f'{int(hex_word, 16):016b}'
                    # Append the bits as integers (reversed to get correct bit order)
                    binary_array.extend([int(bit) for bit in reversed(word_binary)])
                    
                    # Stop when we have processed enough words
                    if len(binary_array) >= num_words * bits_per_word:
                        return np.array(binary_array[:num_words * bits_per_word])
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
'''
def write_xnor_output_to_file(output_file_path, binary_array, num_words, bits_per_word):

    Writes the XNOR result array to a file, formatting it as hex words.

    try:
        with open(output_file_path, 'w') as file_out:
            word_index = 0
            while word_index < num_words:
                # Create a 16-bit binary word from the binary array
                word_bits = binary_array[word_index * bits_per_word: (word_index + 1) * bits_per_word]
                word_str = ''.join(str(bit) for bit in reversed(word_bits))  # Reversing for correct bit order
                
                # Convert the 16-bit binary string to a hexadecimal word
                hex_word = f'{int(word_str, 2):04x}'
                
                # Write the hex word to the file, adding a space between words
                file_out.write(f"{hex_word} ")
                
                word_index += 1
                
                # Add a new line after every 8 words to make the output easier to read
                if word_index % 8 == 0:
                    file_out.write("\n")
                    
    except Exception as e:
        print(f"Error writing to file: {e}")
'''

def compare_files_in_directory(directory, num_words, bits_per_word):
    '''
    Compare each file in the directory against every other file and write the XNOR results.
    '''
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for i in range(len(file_paths)):
        file1 = file_paths[i]
        binary_array1 = read_hex_file_to_binary_array(file1, num_words, bits_per_word)

        if binary_array1 is None:
            print(f"Skipping {file1} due to error reading file.")
            continue

        for j in range(i + 1, len(file_paths)):
            file2 = file_paths[j]
            binary_array2 = read_hex_file_to_binary_array(file2, num_words, bits_per_word)

            if binary_array2 is None:
                print(f"Skipping {file2} due to error reading file.")
                continue

            # Perform XNOR operation between the two binary arrays
            xnor_result_array = np.logical_not(np.logical_xor(binary_array1, binary_array2)).astype(int)

            # Generate the output file path for the XNOR result
            #output_file_name = f"XNOR_{os.path.basename(file1)}_vs_{os.path.basename(file2)}.txt"
            #output_file_path = os.path.join(directory, output_file_name)

            # Write the XNOR result to a file
            #write_xnor_output_to_file(output_file_path, xnor_result_array, num_words, bits_per_word)

            #print(f"XNOR result saved to {output_file_path}")

            # Calculate the percentage of correct values (i.e., where XNOR resulted in 1)
            num_correct_bits = np.sum(xnor_result_array == 1)
            total_bits = len(xnor_result_array)
            percentage_correct = (num_correct_bits / total_bits) * 100

            print(f"Comparison between {os.path.basename(file1)} and {os.path.basename(file2)}: {percentage_correct:.2f}% correct")

# Run the comparison for all files in the directory
compare_files_in_directory(directory_path, NUM_WORDS, BITS_PER_WORD)
