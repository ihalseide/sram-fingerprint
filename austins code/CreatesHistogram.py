from cmath import log
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust the path to your environment
path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\CY-250nm-1-rad\RT-30s-20dumps.txt"
BITS_PER_WORD = 16
NUM_WORDS = 262144  # Total number of words for a 4-megabit chip
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)

# Initialize num_files globally to keep track of how many memory dumps were processed
num_files = 0

# Array to track the number of times each bit is '1' across all memory dumps
bit_votes_for_one = [0] * (NUM_WORDS * BITS_PER_WORD)

def file_read_next_hex4(file_in):
    '''
    Reads the next 4-digit hex number from the file until it reaches the "[end memory dump]" line.
    '''
    global num_files
    in_dump = False
    word_index = 0
    
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            in_dump = True
            word_index = 0  # Reset word index for each dump
            continue
        elif line.strip() == "[end memory dump]":
            in_dump = False
            num_files += 1  # Increment after each memory dump
            continue
        
        if in_dump:
            hex_words = HEX4_WORD_PATTERN.findall(line)
            if hex_words:
                for word_hex in hex_words:
                    if word_index >= NUM_WORDS:
                        raise IndexError(f"Word index {word_index} exceeds expected NUM_WORDS")
                    word = int(word_hex, 16)
                    for word_bit_i in range(BITS_PER_WORD):
                        bit_i = (word_index * BITS_PER_WORD) + word_bit_i
                        if word & (1 << word_bit_i) != 0:
                            bit_votes_for_one[bit_i] += 1
                    word_index += 1

def process_hex_file(file_path):
    '''
    Processes the hex file, counting '1's for each bit across all memory dumps.
    '''
    with open(file_path, 'r') as file_in:
        file_read_next_hex4(file_in)

# Process the single file (containing multiple dumps)
file_path = os.path.join(path, "80_captures_100ms_delay.txt")  # Replace with your actual file name
print(f"Processing file: {file_path}")
process_hex_file(file_path)

# Convert the vote counts to a numpy array
bit_votes_array = np.array(bit_votes_for_one)

# Create a histogram where each bin counts how many bits fell into the same "number of ones" category
hist, bin_edges = np.histogram(bit_votes_array, bins=np.arange(num_files + 2) - 0.5)

# Normalize the histogram by dividing by the total sum of the histogram counts
hist_normalized = hist / hist.sum()

# Plot the normalized histogram as a bar chart
plt.bar(bin_edges[:-1], hist_normalized, color='blue', edgecolor='black', width=0.7)

# Show only even numbers on the x-axis
even_ticks = np.arange(0, num_files + 1, 2)
plt.xticks(even_ticks)  # Set ticks to show only even numbers

plt.title('Normalized Histogram of Powerup 1\'s')
plt.xlabel('Number of Ones')
plt.ylabel('Normalized Occurrences')
plt.grid(True)

# Display the graph
plt.show()
