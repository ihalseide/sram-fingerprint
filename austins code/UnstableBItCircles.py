import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle

# Adjust the path to your environment
file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\CY-250nm-1-rad\RT-30s-20dumps.txt"
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
            while memory_dump_count <= 20:
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

# Normalize bit votes (in terms of instability) for display
max_votes = max(bit_votes_for_one)
if max_votes == 0:
    max_votes = 1  # Avoid division by zero
bit_votes_array = bit_votes_array / max_votes  # Normalize to range 0-1

# Calculate instability (deviation from 50%)
instability_array = np.abs(bit_votes_array - 0.5) * 2  # 0 to 1, with 1 being most unstable

# Set up the image dimensions
img_width = 2048  # Adjust the width as needed
img_height = 2048

# Ensure the reshaped array fits perfectly; truncate excess bits if necessary
instability_array = instability_array[:img_height * img_width]

# Reshape the array to fit the image dimensions
instability_image = instability_array.reshape((img_height, img_width))

# Now, upscale the image so each bit is represented by a 4x4 block
scaling_factor = 4  # Each bit will be a 4x4 block (i.e., 16 pixels)
instability_image_upscaled = np.kron(instability_image, np.ones((scaling_factor, scaling_factor)))

# Create a custom colormap (White -> Red -> White)
colors = ["blue", "red", "yellow"]  # Red for 40-60%, white for others
cmap = ListedColormap(colors)

# Define color boundaries for the instability (0-40% white, 40-60% red, 60-100% white)
bounds = [0, 0.05, 0.95, 1]
norm = plt.Normalize(vmin=0, vmax=1)

# Apply the custom colormap and boundaries
fig, ax = plt.subplots()
im = ax.imshow(instability_image_upscaled, cmap=cmap, norm=norm)

# Add a color bar to indicate the intensity of the instability
plt.colorbar(im, label='Bit instability (normalized)', ticks=[0, 0.4, 0.6, 1])

# Use a Gaussian filter to smooth the image and find unstable regions
instability_density = gaussian_filter(instability_image, sigma=5)

# Plot circles around the most unstable regions
threshold = 0.4  # Set a threshold to identify unstable areas (40-60%)
for i in range(0, img_height, 50):
    for j in range(0, img_width, 50):
        if 0.4 <= instability_image[i, j] <= 0.6:  # Only draw circles in the 40-60% range
            instability_level = instability_image[i, j]
            circle_radius = 5 + (instability_level * 60)  # Circle size based on instability
            # Use red for the circle color as it's within the 40-60% range
            circle = Circle((j * scaling_factor, i * scaling_factor), radius=circle_radius, color='red', fill=False, lw=2)
            ax.add_patch(circle)

# Save and show the image with circles around unstable regions
plt.title('Unstable bits first 130nm Inductor')
plt.savefig("bit_instability_with_circles_40_to_60_percent.png", dpi=20)
plt.show()
