import os
import re
import numpy as np
import csv
 
 # User-defined paths
template_dir = r"C:\Users\afeig\Documents\school\Fall_2024\e-days-templates"
 
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)
 
 
import os
import numpy as np
import re

# Assume HEX4_WORD_PATTERN is defined as:
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

def read_hex_file_to_binary_array_with_skip(file_path):
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
                if len(binary_array) >= NUM_WORDS * BITS_PER_WORD:
                    return np.array(binary_array[:NUM_WORDS * BITS_PER_WORD])

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def compare_arrays(array1, array2):
    '''
    Compares two binary arrays and returns the percentage of correct bits.
    '''
    # Truncate arrays to the same size if needed
    min_length = min(len(array1), len(array2))
    array1 = array1[:min_length]
    array2 = array2[:min_length]

    # XNOR comparison (bitwise)
    xnor_result_array = np.logical_not(np.logical_xor(array1, array2)).astype(int)

    # Calculate the number of correct bits and total bits
    num_correct_bits = np.sum(xnor_result_array == 1)
    total_bits = np.sum(xnor_result_array != -1)  # Exclude "don't care" bits (if any)
    return (num_correct_bits / total_bits) * 100 if total_bits > 0 else 0

def main(input_array):
    # User-defined path to templates
    template_names = []
    percentages = []

    # Loop through all templates in the template_dir
    for template_file in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_file)
        
        # Read template binary array
        template_array = read_hex_file_to_binary_array_with_skip(template_path)
        
        if template_array is None:
            continue  # Skip this template if it couldn't be read

        # Compare input array to the template array
        percentage_correct = compare_arrays(input_array, template_array)

        # Store results
        template_names.append(template_file)
        percentages.append(percentage_correct)

    return template_names, percentages
