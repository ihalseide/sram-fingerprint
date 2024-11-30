import os
import re
import numpy as np
import csv

# User-defined paths
template_dir = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\Capacitor_templates\excluded_5_95"
memory_dump_dir = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\Memory_dumps_normal"
csv_file_path = r"C:\Users\afeig\Documents\school\Current_Semester\ECE-401\5_15_second_shutoff\BitHeatMaps_NoMiddleValues\Capacitor_templates"

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
    for line in file_in:
        if line.strip() == "[begin memory dump]":
            return True
    return False


def file_read_next_hex4(file_in):
    for line in file_in:
        if line.strip() == "[end memory dump]":
            break
        hex_words = HEX4_WORD_PATTERN.findall(line)
        if hex_words:
            for word_hex in hex_words:
                yield int(word_hex, 16)


def compare_files(template_file, memory_dump_file, num_words, bits_per_word):
    template_array, skip_bits = read_hex_file_to_binary_array_with_skip(template_file, num_words, bits_per_word)
    if template_array is None:
        return None

    percentages = []
    try:
        with open(memory_dump_file, 'r') as file_in:
            while True:
                if not file_skip_until_memory_dump(file_in):
                    break

                binary_array2 = []
                for word in file_read_next_hex4(file_in):
                    word_binary = f'{word:016b}'
                    binary_array2.extend([int(bit) for bit in reversed(word_binary)])
                    if len(binary_array2) >= len(template_array):
                        break

                # Truncate arrays to the same size if needed
                min_length = min(len(template_array), len(binary_array2))
                if len(template_array) != len(binary_array2):
                    print(f"Warning: Size mismatch, truncating arrays to {min_length} bits.")
                template_array = template_array[:min_length]
                binary_array2 = binary_array2[:min_length]

                xnor_result_array = np.logical_not(np.logical_xor(template_array, binary_array2)).astype(int)

                if skip_bits:
                    for bit in skip_bits:
                        if bit < len(xnor_result_array):
                            xnor_result_array[bit] = -1

                num_correct_bits = np.sum(xnor_result_array == 1)
                total_bits = np.sum(xnor_result_array != -1)
                percentage_correct = (num_correct_bits / total_bits) * 100 if total_bits > 0 else 0
                percentages.append(percentage_correct)
    except FileNotFoundError:
        print(f"Error reading {memory_dump_file}.")
        return None

    return np.mean(percentages) if percentages else None


def main():
    template_files = [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]
    memory_dump_files = [f for f in os.listdir(memory_dump_dir) if os.path.isfile(os.path.join(memory_dump_dir, f))]

    memory_dump_headers = [f.split('_')[0] for f in memory_dump_files]
    output_csv_path = os.path.join(csv_file_path, "comparison_results(cap_template(5-95)_normal_dumps).csv")

    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header
        csv_writer.writerow(['Template File'] + memory_dump_headers)

        for template_file in template_files:
            row = [template_file]
            template_path = os.path.join(template_dir, template_file)
            print(template_path)
            
            for memory_dump_file in memory_dump_files:
                memory_dump_path = os.path.join(memory_dump_dir, memory_dump_file)
                avg_percentage = compare_files(template_path, memory_dump_path, NUM_WORDS, BITS_PER_WORD)
                row.append(f"{avg_percentage:.2f}" if avg_percentage is not None else "N/A")
            
            csv_writer.writerow(row)

    print(f"Comparison results saved to {output_csv_path}")


# Execute the main function
if __name__ == "__main__":
    main()
