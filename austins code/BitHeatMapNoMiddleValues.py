import re
import numpy as np
import os

directory_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\Memory_dumps_capacitor"
output_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\Capacitor_templates\excluded_20_80"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

# Array to track the number of times each bit is '1' across all reads in the file

# Function to skip until memory dump begins
def file_skip_until_memory_dump(file_in):
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            break
    return True

# Function to read hex4 words until memory dump ends
def file_read_next_hex4(file_in):
    for line in file_in:
        if line.strip() == "[end memory dump]":
            break
        hex_words = HEX4_WORD_PATTERN.findall(line)
        if hex_words:
            for word_hex in hex_words:
                yield int(word_hex, 16)

# Processing the hex file
def process_hex_file_with_multiple_dumps(file_path, bit_votes_for_one):
    try:
        with open(file_path, 'r') as file_in:
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
        print("File not found. Check the file path.")



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


def main():
	for filename in os.listdir(directory_path):
		if filename.endswith(".txt"):
			bit_votes_for_one = [0] * (NUM_WORDS * BITS_PER_WORD)
			file_path = os.path.join(directory_path, filename)

			# Extract the base name of the file up to the `_` character
			base_name = filename.split('_')[0]

            # Construct the output file path
			output_path = os.path.join(
                os.path.dirname(output_file_path),  # Use the same directory as the original output_file_path
                f"{base_name}.txt"
            )

           # Process the file
			process_hex_file_with_multiple_dumps(file_path, bit_votes_for_one)

		# Convert vote counts to a numpy array
			bit_votes_array = np.array(bit_votes_for_one)

		# Threshold calculation
			num_dumps = 50  # Assuming 50 dumps
			threshold = 0.5 * num_dumps

		# Create a binary array based on the threshold
			binary_bit_array = np.where(bit_votes_array > threshold, 1, 0)

		# Adjusted thresholds for don't care range
			lower_threshold = 0.05 * 50  # percentage number to exclude
			upper_threshold = 0.95 * 50  # percentage number to exclude

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

            # Write the results to the dynamically named file
			write_bits_to_file(output_path, binary_bit_array, NUM_WORDS, BITS_PER_WORD, skip_bit_locations)

			print(f"Binary bit data and skip locations saved to {output_path}")

            
            
if __name__ == "__main__":
    main()