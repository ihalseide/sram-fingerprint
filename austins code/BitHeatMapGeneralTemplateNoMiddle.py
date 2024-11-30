import re
import numpy as np
import os

# Directory containing the files
directory_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\General_BitHeatMaps_NoMiddle\Memory_dumps\CY_150nm"
output_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\General_BitHeatMaps_NoMiddle\Memory_dumps\150_templates\CY-150nm(10-90).txt"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

# Array to track the number of times each bit is '1' across all reads in the directory
bit_votes_for_one = [0] * (NUM_WORDS * BITS_PER_WORD)

# Function to skip until memory dump begins
def file_skip_until_memory_dump(file_in):
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            return True
    return False

# Function to read hex4 words until memory dump ends
def file_read_next_hex4(file_in):
    for line in file_in:
        if line.strip() == "[end memory dump]":
            break
        hex_words = HEX4_WORD_PATTERN.findall(line)
        if hex_words:
            for word_hex in hex_words:
                yield int(word_hex, 16)

# Process each file in the directory
def process_directory_with_multiple_files(directory_path):
    file_count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r') as file_in:
                    file_count += 1
                    memory_dump_count = 0
                    while memory_dump_count <= 50:
                        # Skip until memory dump
                        if not file_skip_until_memory_dump(file_in):
                            break

                        word_index = 0
                        for word in file_read_next_hex4(file_in):
                            if word_index >= NUM_WORDS:
                                raise IndexError(f"Word index {word_index} exceeds expected NUM_WORDS")
                            for word_bit_i in range(BITS_PER_WORD):
                                bit_i = (word_index * BITS_PER_WORD) + word_bit_i
                                if word & (1 << word_bit_i) != 0:
                                    bit_votes_for_one[bit_i] += 1
                            word_index += 1
                        memory_dump_count += 1

            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping.")

    return file_count

# Process the directory and accumulate the bit votes
num_files_processed = process_directory_with_multiple_files(directory_path)

# Convert vote counts to a numpy array
bit_votes_array = np.array(bit_votes_for_one)

# Adjusted thresholds for don't care range
lower_threshold = 0.01 * num_files_processed * 50  # percentage number to exclude
upper_threshold = 0.99 * num_files_processed * 50  # percentage number to exclude

# Create a binary array based on the thresholds and collect "don't care" positions
binary_bit_array = []
skip_bit_locations = []

for i, vote_count in enumerate(bit_votes_array):
    if vote_count <= lower_threshold:
        binary_bit_array.append(0)  # Mark as '0' if below 20% threshold
    elif vote_count >= upper_threshold:
        binary_bit_array.append(1)  # Mark as '1' if above 80% threshold
    else:
        binary_bit_array.append(0)  # Mark as '0' in final output
        skip_bit_locations.append(i)  # Track as "don't care"

# Write the binary array and skip bit locations to the output file
def write_bits_to_file(output_file_path, binary_array, num_words, bits_per_word, skip_bits):
    '''
    Writes the binary array to a file, formatting it as hex words, and writes
    the bit locations to skip in the first line.
    '''
    try:
        with open(output_file_path, 'w') as file_out:
            # Write the skipped bit locations as the first line (comma-separated)
            file_out.write(','.join(map(str, skip_bits)) + '\n')
            
            word_index = 0
            while word_index < num_words:
                # Create a 16-bit binary word from the binary array
                word_bits = binary_array[word_index * bits_per_word: (word_index + 1) * bits_per_word]
                word_str = ''.join(str(bit) for bit in reversed(word_bits))
                
                # Convert the 16-bit binary string to a hexadecimal word
                hex_word = f'{int(word_str, 2):04x}'
                
                # Write the hex word to the file, adding a space between words
                file_out.write(f"{hex_word} ")
                
                word_index += 1
                
                # Add a new line after every 16 words for readability
                if word_index % 16 == 0:
                    file_out.write("\n")
                    
    except Exception as e:
        print(f"Error writing to file: {e}")

# Write to the file, including the bit locations to skip
write_bits_to_file(output_file_path, binary_bit_array, NUM_WORDS, BITS_PER_WORD, skip_bit_locations)

print(f"Processed {num_files_processed} files.")
print(f"Binary bit data and skip locations saved to {output_file_path}")
