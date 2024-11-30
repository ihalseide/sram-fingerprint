import os
import re
import numpy as np

# Adjust the path to your environment
heat_map_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_130nm-1_diff_manu_inductor.txt"
data_dump_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\IDP-130nm-1\50_captures_15_second_delay.txt"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

def read_hex_file_to_binary_array(file_path, num_words, bits_per_word):
    '''Reads a file containing hex words and converts it into a binary array.'''
    binary_array = []
    try:
        with open(file_path, 'r') as file_in:
            for line in file_in:
                hex_words = HEX4_WORD_PATTERN.findall(line)
                for hex_word in hex_words:
                    word_binary = f'{int(hex_word, 16):016b}'
                    binary_array.extend([int(bit) for bit in reversed(word_binary)])
                    if len(binary_array) >= num_words * bits_per_word:
                        return np.array(binary_array[:num_words * bits_per_word])
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def file_skip_until_memory_dump(file_in):
    '''Skips the file content until we reach the "[begin memory dump]" line.'''
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            return True
    return False

def file_read_next_hex4(file_in):
    '''Reads the next 4-digit hex number until reaching "[end memory dump]".'''
    for line in file_in:
        if line.strip() == "[end memory dump]":
            break
        hex_words = HEX4_WORD_PATTERN.findall(line)
        if hex_words:
            for word_hex in hex_words:
                yield int(word_hex, 16)

def process_hex_file_with_multiple_dumps(file_path, heat_map_binary_array):
    '''Processes hex words across multiple memory dumps, counting '1's for each bit.'''
    percentages = []
    
    try:
        with open(file_path, 'r') as file_in:
            while True:
                # Skip to the next memory dump
                if not file_skip_until_memory_dump(file_in):
                    break

                # Reinitialize the binary array for the current memory dump
                binary_array2 = []
                word_index = 0

                # Read words until the end of the current memory dump
                for word in file_read_next_hex4(file_in):
                    if word_index >= NUM_WORDS:
                        raise IndexError(f"Word index {word_index} exceeds expected NUM_WORDS")
                    word_binary = f'{word:016b}'  # Get 16-bit binary
                    binary_array2.extend([int(bit) for bit in reversed(word_binary)])
                    word_index += 1

                # Reset to the beginning of the heat map binary array for comparison
                xnor_result_array = np.logical_not(np.logical_xor(heat_map_binary_array, binary_array2)).astype(int)

                # Calculate percentage of correct bits
                num_correct_bits = np.sum(xnor_result_array == 1)
                total_bits = len(xnor_result_array)
                percentage_correct = (num_correct_bits / total_bits) * 100
                percentages.append(percentage_correct)

    except FileNotFoundError:
        print("File not found. Check the file path.")
    
    return percentages

# Read the heat map into a binary array
heat_map_binary_array = read_hex_file_to_binary_array(heat_map_file_path, NUM_WORDS, BITS_PER_WORD)

if heat_map_binary_array is not None:
    percentages = process_hex_file_with_multiple_dumps(data_dump_file_path, heat_map_binary_array)

    # Only display the highest and lowest percentages
    if percentages:
        highest_percentage = max(percentages)
        lowest_percentage = min(percentages)
        print(f"Highest percentage of correct bits: {highest_percentage:.2f}%")
        print(f"Lowest percentage of correct bits: {lowest_percentage:.2f}%")
