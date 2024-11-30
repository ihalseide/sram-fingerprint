import os
import re
import numpy as np

BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

def read_hex_file_to_binary_array_with_skip(file_path, num_words, bits_per_word):
    '''
    Reads a file containing hex words, processes any skip bits if present,
    and converts it into a binary array. Each hex word is expanded to a 16-bit binary representation.
    '''
    binary_array = []
    skip_bits = set()
    
    try:
        with open(file_path, 'r') as file_in:
            first_line = file_in.readline().strip()

            # Try to parse skip bits from the first line
            try:
                skip_bits = set(map(int, first_line.split(',')))
            except ValueError:
                # If parsing fails, assume the first line is not skip bits
                print(f"Invalid skip bits line in {file_path}, skipping to hex values.")
                file_in.seek(0)  # Reset to the beginning of the file
            
            # Now process the file as normal, looking for hex values
            for line in file_in:
                hex_words = HEX4_WORD_PATTERN.findall(line)
                for hex_word in hex_words:
                    # Convert hex to binary (padded to 16 bits)
                    word_binary = f'{int(hex_word, 16):016b}'
                    # Append the bits as integers (reversed to get correct bit order)
                    binary_array.extend([int(bit) for bit in reversed(word_binary)])
                    
                    # Stop when we have processed enough words
                    if len(binary_array) >= num_words * bits_per_word:
                        return np.array(binary_array[:num_words * bits_per_word]), skip_bits
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None, None

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

def compare_heatmap_with_memory_dump(heatmap_file, memory_dump_file, num_words, bits_per_word):
    '''
    Compare the heatmap file against the memory dump file.
    '''
    heatmap_array, skip_bits = read_hex_file_to_binary_array_with_skip(heatmap_file, num_words, bits_per_word)

    if heatmap_array is None:
        print(f"Skipping {heatmap_file} due to error reading file.")
        return

    highest_percentage = float('-inf')  # Initialize to negative infinity
    lowest_percentage = float('inf')     # Initialize to positive infinity

    print(f"Processing memory dump: {memory_dump_file}")

    try:
        with open(memory_dump_file, 'r') as file_in:
            while True:
                # Skip to the next memory dump
                if not file_skip_until_memory_dump(file_in):
                    break

                # Initialize the binary array for the current memory dump
                binary_array2 = []
                word_index = 0

                # Read words until the end of the current memory dump
                for word in file_read_next_hex4(file_in):
                    if word_index >= NUM_WORDS:
                        raise IndexError(f"Word index {word_index} exceeds expected NUM_WORDS")
                    word_binary = f'{word:016b}'  # Get 16-bit binary
                    binary_array2.extend([int(bit) for bit in reversed(word_binary)])
                    word_index += 1

                # Ensure both arrays are the same size
                if len(binary_array2) != len(heatmap_array):
                    min_length = min(len(heatmap_array), len(binary_array2))
                    print(f"Warning: Size mismatch, truncating arrays to {min_length} bits.")
                    heatmap_array = heatmap_array[:min_length]
                    binary_array2 = binary_array2[:min_length]

                # Perform XNOR operation between the heatmap and the current memory dump
                xnor_result_array = np.logical_not(np.logical_xor(heatmap_array, binary_array2)).astype(int)

                # Set the skip bits in the XNOR result to -1 (or any value indicating 'don't care')
                if skip_bits:
                    for bit in skip_bits:
                        if bit < len(xnor_result_array):
                            xnor_result_array[bit] = -1  # -1 for 'don't care'

                # Calculate the percentage of correct values (i.e., where XNOR resulted in 1)
                num_correct_bits = np.sum(xnor_result_array == 1)
                total_bits = np.sum(xnor_result_array != -1)  # Only count bits that aren't 'don't care'
                percentage_correct = (num_correct_bits / total_bits) * 100 if total_bits > 0 else 0

                # Update highest and lowest percentages
                highest_percentage = max(highest_percentage, percentage_correct)
                lowest_percentage = min(lowest_percentage, percentage_correct)

    except FileNotFoundError:
        print(f"Error reading {memory_dump_file}. Skipping.")
        return

    # Print the results
    print(f"Highest percentage correct: {highest_percentage:.2f}%")
    print(f"Lowest percentage correct: {lowest_percentage:.2f}%")

# Call the comparison function


def main():
	Heat_map_file_array = [r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_90nm.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_90nm-2.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_90nm-3.txt"
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_150nm-1.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_150nm-2.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_150nm-3.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_130nm-1_diff_manu.txt",
                       r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\BitHeatMap_130nm-2_diff_manu.txt"]
	memory_dump_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\Garbage_Values_90nm-2\50_captures_15_second_delay.txt" 
	for i in range(len(Heat_map_file_array)):
		read_hex_file_to_binary_array_with_skip(Heat_map_file_array[i], NUM_WORDS, BITS_PER_WORD)
		compare_heatmap_with_memory_dump(Heat_map_file_array[i], memory_dump_file_path, NUM_WORDS, BITS_PER_WORD)
