import re
import numpy as np

# Adjust the path to your environment
file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\250nm-30s-30C-Hex-dumps-x50.txt"
output_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\BitHeatMap_250nm.txt"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

# Array to track the number of times each bit is '1' across all reads in the file
bit_votes_for_one = [0] * (NUM_WORDS * BITS_PER_WORD)

def file_skip_until_memory_dump(file_in):
    '''
    Skips the file content until we reach the "[begin memory dump]" line.
    '''
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            break
    return True

def file_read_next_hex4(file_in):
    '''
    Reads the next 4-digit hex number from the file until it reaches the "[end memory dump]" line.
    '''
    for line in file_in:
        if line.strip() == "[end memory dump]":
            break
        hex_words = HEX4_WORD_PATTERN.findall(line)
        if hex_words:
            for word_hex in hex_words:
                yield int(word_hex, 16)

def process_hex_file_with_multiple_dumps(file_path):
    '''
    Processes hex words across multiple memory dumps within a single file,
    counting '1's for each bit. Stops after processing 78 memory dumps.
    '''
    try:
        with open(file_path, 'r') as file_in:
            memory_dump_count = 0
            while memory_dump_count <= 50:
                # Skip until the next memory dump or break if we reach the end of the file
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
        print("File not found. Check the file path.")

# Process the single file with multiple memory dumps
process_hex_file_with_multiple_dumps(file_path)

# Convert the vote counts to a numpy array
bit_votes_array = np.array(bit_votes_for_one)

# Get the total number of memory dumps (to determine percentage)
num_dumps = 50  # Assuming we process 78 dumps

# Threshold calculation: if more than 50% of dumps have the bit set to '1', keep it as '1'
threshold = 0.5 * num_dumps

# Create a binary array: 1 if bit_votes > threshold, otherwise 0
binary_bit_array = np.where(bit_votes_array > threshold, 1, 0)

# Now, we need to write this binary array to a file in the correct format
def write_bits_to_file(output_file_path, binary_array, num_words, bits_per_word):
    '''
    Writes the binary array to a file, formatting it as hex words.
    Each hex word represents a 16-bit binary word.
    '''
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
                if word_index % 16 == 0:
                    file_out.write("\n")
                    
    except Exception as e:
        print(f"Error writing to file: {e}")

# Write the binary bit array to the output file in the correct format
write_bits_to_file(output_file_path, binary_bit_array, NUM_WORDS, BITS_PER_WORD)

print(f"Binary bit data saved to {output_file_path}")
