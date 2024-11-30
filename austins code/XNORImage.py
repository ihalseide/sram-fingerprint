import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Adjust the path to your environment
file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\xnor_output.txt"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

# Array to track the XNOR output bits
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

def process_hex_file_with_single_dump(file_path):
    '''
    Processes hex words within a single memory dump, counting '1's for each bit from XNOR outputs.
    '''
    try:
        with open(file_path, 'r') as file_in:
            # Skip until the next memory dump or break if we reach the end of the file
            if not file_skip_until_memory_dump(file_in):
                return

            previous_word = None
            word_index = 0
            for word in file_read_next_hex4(file_in):
                if previous_word is not None:
                    # Compute XNOR
                    xnor_result = ~(word ^ previous_word) & 0xFFFF  # 16 bits
                    for word_bit_i in range(BITS_PER_WORD):
                        # Increment count based on the XNOR result
                        if xnor_result & (1 << word_bit_i):
                            bit_votes_for_one[(word_index * BITS_PER_WORD) + word_bit_i] = 1  # Correct bit
                        else:
                            bit_votes_for_one[(word_index * BITS_PER_WORD) + word_bit_i] = 0  # Incorrect bit
                previous_word = word
                word_index += 1
                if word_index >= NUM_WORDS:
                    raise IndexError(f"Word index {word_index} exceeds expected NUM_WORDS")

    except FileNotFoundError:
        print("File not found. Check the file path.")

# Process the single file with the XNOR output
process_hex_file_with_single_dump(file_path)

# Convert the vote counts to a numpy array
bit_votes_array = np.array(bit_votes_for_one)

# Set up the image dimensions
img_width = 2048  # Adjust the width as needed
img_height = 2048

# Ensure the reshaped array fits perfectly; truncate excess bits if necessary
bit_votes_array = bit_votes_array[:img_height * img_width]

# Reshape the array to fit the image dimensions
bit_votes_image = bit_votes_array.reshape((img_height, img_width))

# Now, upscale the image so each bit is represented by a 4x4 block
scaling_factor = 4  # Each bit will be a 4x4 block (i.e., 16 pixels)
bit_votes_image_upscaled = np.kron(bit_votes_image, np.ones((scaling_factor, scaling_factor)))

# Define custom colors: white for 0 (incorrect), dark blue for 1 (correct)
colors = ['white', 'red']
cmap = ListedColormap(colors)

# Define boundaries: 0 (white) and 1 (dark blue)
bounds = [0, 1]
norm = BoundaryNorm(bounds, cmap.N)

# Apply the custom colormap and boundaries
plt.imshow(bit_votes_image_upscaled, cmap=cmap, norm=norm)

# Add a color bar to indicate the bit correctness
plt.colorbar(label='Bit correctness', ticks=[0, 1], format='%d')

# Save and show the image
plt.savefig("xnor_bit_votes_visualization.png", dpi=20)
plt.show()
