import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Adjust the path to your environment
file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\CY-250nm-1\2024.10.22-normal-30s-50dumps.txt"
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

# Normalize bit votes for display (0-1 range)
max_votes = max(bit_votes_for_one)
if max_votes == 0:
    max_votes = 1  # Avoid division by zero
bit_votes_array = bit_votes_array / max_votes  # Normalize to range 0-1

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

# Define custom colors for each range
colors = ['white', 'lightblue', 'blue', 'darkblue']
cmap = ListedColormap(colors)

# Define boundaries: 0-25%, 26-50%, 51-75%, 76-100%
bounds = [0, 0.25, 0.50, 0.75, 1]
norm = BoundaryNorm(bounds, cmap.N)

# Apply the custom colormap and boundaries
plt.imshow(bit_votes_image_upscaled, cmap=cmap, norm=norm)

# Add a color bar to indicate the intensity of the votes
plt.colorbar(label='Bit vote count (normalized)', ticks=[0, 0.25, 0.50, 0.75, 1])

# Save and show the image
plt.title('Sectioned Heatmap CY 250nm')
plt.savefig("bit_votes_colormap_visualization_custom_upscaled.png", dpi=20)
plt.show()
