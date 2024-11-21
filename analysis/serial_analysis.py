'''
Code to analyze the words and bits of one or many SRAM memory capture files from the Arduino DUE.
Author: INH
'''


import re, math
import numpy as np
from operator import xor
from typing import BinaryIO, Iterable, TextIO
from collections import defaultdict


BITS_PER_WORD = 16
NUM_WORDS = 2**18
NUM_BITS = BITS_PER_WORD * NUM_WORDS
HEX4_WORD_PATTERN = re.compile(r"[a-f0-9]{4}", re.IGNORECASE)


def main_create_gold_puf():
    num_captures = 11
    assert(num_captures % 2 != 0) # must be odd so there are no voting ties
    input_file_names = [ f"chip inh2 gold PUF/capture {i}.txt" for i in range(1, 1 + num_captures) ]
    output_file_name = "chip inh2 gold PUF/gold PUF.txt"
    
    print(f"output file name: \"{output_file_name}\"")

    # Array to map the data's bit index to number of votes that it should be set to a '1'
    bit_votes_for_one = [ 0 for _ in range(NUM_BITS) ]

    for file_name in input_file_names:
        print(f"- loading file \"{file_name}\"")
        with open(file_name, "rb") as file_in:
            # Read each data word from the data file
            for word_i in range(NUM_WORDS):
                word = file_read_next_hex4(file_in)
                # Log the "bit vote" for each bit of the word
                for word_bit_i in range(BITS_PER_WORD):
                    bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                    # Test bit number 'bit_i' and increment a vote if it is set
                    if word & (1 << word_bit_i):
                        bit_votes_for_one[bit_i] += 1

    # Iterate bit votes and print out best-voted word values
    print(f"saving result to output file")
    with open(output_file_name, "w") as file_out:
        for word_i in range(NUM_WORDS):
            word_value = 0
            for word_bit_i in range(BITS_PER_WORD):
                bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                # Resolve the bit votes to see if '1' or '0' wins the majority
                if bit_votes_for_one[bit_i] > (num_captures // 2):
                    word_value |= 1 << word_bit_i
            print(f"{word_value:04X}", file=file_out)


def combine_captures(capture_file_names: list[str], output_file):
    # Allow 'output_file' to be a file or file path
    if isinstance(output_file, str):
        print(f"combining memory dumps to file '{output_file}'...")
        with open(output_file, 'w') as output_file2:
            return combine_captures(capture_file_names, output_file2)
        
    for file_name in capture_file_names:
        print(f"adding data from '{file_name}'...")
        with open(file_name, 'rb') as file_in:
            print("[begin memory dump]", file=output_file)

            while (word := file_read_next_hex4_no_error(file_in)) is not None:
                print(hex4(word), file=output_file)

            print("[end memory dump]", file=output_file)


def create_gold_puf_v2(num_captures: int, input_file_name: str, output_file_name: str, num_words: int = NUM_WORDS):
    '''
    Use majority voting to create a PUF file that represents the desired start up memory state, given multiple memory dumps.
    The 'input_file_name' should have multiple memory dumps (at least 'num_captures' of them) inside.
    '''
    assert num_captures % 2 != 0, "num_captures must be odd so there can be no voting ties"

    # print(f"input file name: \"{input_file_name}\"")
    # print(f"output file name: \"{output_file_name}\"")

    # Array to map the data's bit index to number of votes that it should be set to a '1'
    bit_votes_for_one = [ 0 for _ in range(num_words * BITS_PER_WORD) ]

    print(f"- loading file \"{input_file_name}\"")
    with open(input_file_name, "r") as file_in:
        for c in range(1, num_captures + 1):
            print(f"Processing capture #{c} in the file")

            if file_seek_next_data_dump(file_in) is None:
                raise ValueError(f"not enough memory dumps in input file '{input_file_name}': expected {num_captures} but stopped at {c}")

            # Read each data word from the data file
            for word_i in range(num_words):
                word = file_read_next_hex4(file_in)
                # Log the "bit vote" for each bit of the word
                for word_bit_i in range(BITS_PER_WORD):
                    bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                    # Test bit number 'bit_i' and increment a vote if it is set
                    if word & (1 << word_bit_i) != 0:
                        bit_votes_for_one[bit_i] += 1

    # Iterate bit votes and write out best-voted word values to the output file
    print(f"saving result to output file...")
    with open(output_file_name, "w") as file_out:
        for word_i in range(num_words):
            word_value = 0
            for word_bit_i in range(BITS_PER_WORD):
                bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                # Resolve the bit votes to see if '1' or '0' wins the majority
                if bit_votes_for_one[bit_i] > (num_captures // 2):
                    word_value |= (1 << word_bit_i)
            # Save hex representation of the majority word to the output file
            ending = '' if word_i == (num_words - 1) else ' ' # no newline for the last line
            print(f"{word_value:04X}", file=file_out, end=ending)
            if (word_i + 1) % 16 == 0:
                print(file=file_out)

    print("done")


def create_gold_puf_classification(num_captures: int, input_file_name: str, output_file_name: str, mode: str):
    '''
    Use voting to create a PUF hex memory dump file where the only bits set to '1' are those bits that match the
    criteria represented by the 'mode' parameter.
    - NOTE: in this function, a "natural" 0 or 1 really means a STRONG 0 or 1
    - NOTE: The 'input_file_name' should have multiple memory dumps (at least 'num_captures' of them) inside.
    '''

    assert num_captures % 2 != 0, "num_captures must be odd so there can be no voting ties"
    allowed_modes = ('natural 0s', 'natural 1s', 'wavering')
    if mode not in allowed_modes:
        raise ValueError(f"'mode' must be one of {allowed_modes}")

    print(f"input file name: \"{input_file_name}\"")
    print(f"output file name: \"{output_file_name}\"")

    # Array to map the data's bit index to number of votes that it should be set to a '1'
    bit_votes_for_one = [ 0 for _ in range(NUM_BITS) ]

    print(f"- loading file \"{input_file_name}\"")
    with open(input_file_name, "r") as file_in:
        for c in range(1, num_captures + 1):
            print(f"Processing capture #{c} in the file")

            if file_seek_next_data_dump(file_in) is None:
                raise ValueError(f"not enough memory dumps in input file '{input_file_name}': expected {num_captures} but stopped at {c}")

            # Read each data word from the data file
            for word_i in range(NUM_WORDS):
                word = file_read_next_hex4(file_in)
                # Log the "bit vote" for each bit of the word
                for word_bit_i in range(BITS_PER_WORD):
                    bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                    # Test bit number 'bit_i' and increment a vote if it is set
                    if word & (1 << word_bit_i) != 0:
                        bit_votes_for_one[bit_i] += 1

    # Iterate bit votes and set the output bits to '1' that match the criteria for the given 'mode'
    print(f"saving result to output file...")
    with open(output_file_name, "w") as file_out:
        for word_i in range(NUM_WORDS):
            word_value = 0
            for word_bit_i in range(BITS_PER_WORD):
                bit_i = (word_i * BITS_PER_WORD) + word_bit_i
                # Resolve the bit votes to see if '1' or '0' wins the majority
                if mode == 'natural 0s':
                    # A 'natural 0' is where a bit is always zero
                    if bit_votes_for_one[bit_i] == 0:
                        word_value |= (1 << word_bit_i)
                elif mode == 'natural 1s':
                    # A 'natural 1' is where a bit is always one
                    if bit_votes_for_one[bit_i] == num_captures:
                        word_value |= (1 << word_bit_i)
                elif mode == 'wavering':
                    # A 'wavering' is where a bit is not always one or always zero
                    if bit_votes_for_one[bit_i] not in (0, num_captures):
                        word_value |= (1 << word_bit_i)
                else:
                    assert False, 'unreachable'
            # Save hex representation of the majority word to the output file
            ending = '' if word_i == (NUM_WORDS - 1) else '\n' # no newline for the last line
            print(f"{word_value:04X}", file=file_out, end=ending)

    print("done")


def file_read_hex4_dump_as_words(file_in: TextIO, num_words: int) -> np.ndarray:
    '''Load a hex4 memory dump file as a 1D array of 16-bit integer words.'''
    result = np.empty(num_words, dtype='uint16')
    for i in range(num_words):
        result[i] = file_read_next_hex4(file_in)
    return result


def main_stats():
    fileName = r"chip inh2 gold PUF/gold PUF.txt"
    expected_num_words = 2**18

    hex_re = HEX4_WORD_PATTERN
    word_sep = '\n'
    ham_weight = 0

    occurrences = defaultdict(lambda: 0)

    with open(fileName, "r") as file:
        print(f"Reading from file \"{fileName}\"")
        index = -1
        while (word := file.read(4)):
            index += 1
            if not hex_re.match(word):
                raise ValueError(f"bad hex word at address {index}: '{word}'")
            int_val = int(word, 16)
            occurrences[int_val] += 1
            ham_weight += bit_weight(int_val)
            if not (sep := file.read(1)):
                break
            assert(sep == word_sep)

    if index != expected_num_words - 1:
        print(f"[NOTE] got {index} words, but not get the expected number of words ({expected_num_words})")
    else:
        print("Got the expected amount of data words")

    print(f"Hamming weight: {ham_weight:,}")

    num_bits = expected_num_words * 16
    percent_ones = percent(ham_weight, num_bits)
    print(f"Percentage of 1's in the data: {percent_ones:.4f}%")
    print(f"Percentage of 0's in the data: {100-percent_ones:.4f}%")

    max_entry = max(occurrences, key=occurrences.get)
    max_entry_num = occurrences[max_entry]
    max_percent = percent(max_entry_num, expected_num_words)
    print(f"Most common value: '{hex(max_entry)}' = '{bin(max_entry)}', which occurs {max_entry_num:,} times, which is {max_percent:.3f}% of entries")


def diff_puf_and_multi_capture(puf_file_name: str, trials_dump_file_name: str, num_words=NUM_WORDS):
    with open(puf_file_name, "rb") as puf_file:
        with open(trials_dump_file_name, "rb") as trials_file:
            val_ms = 0.0
            percent_diff = 0.0
            while line := trials_file.readline().decode('ascii'):
                line = line.strip()
                if line == "[begin memory dump]":
                    # get memory dump bits and diff it with the PUF file
                    puf_file.seek(0) # go to beginning of PUF file
                    a_weight, b_weight, diff = bit_diff_within_files(puf_file, trials_file, num_words, puf_file_name, trials_dump_file_name)
                    #report_file_bit_diff(puf_file_name, trials_dump_file_name, a_weight, b_weight, diff)
                    percent_diff = percent(diff, num_words * BITS_PER_WORD)
                    print(f"Percent difference: {percent_diff:.3f}")  
                else:
                    print(line)


def diff_puf_and_trials_dump(puf_file_name: str, trials_dump_file_name: str, num_words=NUM_WORDS) -> tuple[tuple[float, float], ...]:
    '''Run a diff against the PUF file and multiple memory dumps that come from the Arduino'''
    data_result: list[tuple[float, float]] = list()
    print(f"Diffing file \"{puf_file_name}\" against trials file \"{trials_dump_file_name}\"")
    with open(puf_file_name, "rb") as puf_file:
        with open(trials_dump_file_name, "rb") as trials_file:
            val_ms = 0.0
            percent_diff = 0.0
            while (line := trials_file.readline().decode()):
                line = line.strip()
                if line.startswith("Beginning trial"): # once we find a start of trial, record the delay value (in milliseconds)
                    # extract delay value from the trial line
                    match = re.search(r"([0-9]+\.[0-9]+)ms", line) # Look for decimal number "<n>.<n>ms"
                    assert(match is not None)
                    val_ms = float(match.group(1))
                elif line == "[begin memory dump]": # once we find the start of the memory dump inside a trial, read files & get the bit difference
                    print(f"{len(data_result) + 1}", end=' ')
                    # get memory dump bits and diff it with the PUF file
                    puf_file.seek(0) # go to beginning of PUF file
                    a_weight, b_weight, diff = bit_diff_within_files(puf_file, trials_file, num_words, puf_file_name, trials_dump_file_name)
                    #report_file_bit_diff(puf_file_name, trials_dump_file_name, a_weight, b_weight, diff)
                    percent_diff = percent(diff, num_words * BITS_PER_WORD)
                    data_result.append( (val_ms, percent_diff,) )    
    print()
    return tuple(data_result)      


def remanence_experiment_get_images(dump_file, image_width: int, image_height: int) -> dict[float, np.ndarray]:
    # Use recursive case to accept a 'str' or a binary file for the first function argument
    if isinstance(dump_file, str):
        with open(dump_file, 'rb') as dump_file:
            assert (not isinstance(dump_file, str))
            return remanence_experiment_get_images(dump_file, image_width, image_height)
        
    delay_image_pairs: dict[float, np.ndarray] = dict()
    while (line := dump_file.readline().decode('ascii')):
        delay_str = file_seek_next_delay_line(dump_file)
        if delay_str is None:
            # Reached the end of the dump_file's memory dumps
            break
        current_delay = float(delay_str)
        file_seek_next_data_dump(dump_file)
        delay_image_pairs[current_delay] = file_read_image(dump_file, image_width, image_height)
    return delay_image_pairs


def file_load_delays(dump_file: TextIO, num_captures: int) -> np.ndarray:
    '''Collect `num_captures` delay values from a data dump file (this skips over any data dumps in-between).'''
    results = np.empty(num_captures)
    for i in range(num_captures):
        line = file_seek_next_delay_line(dump_file)
        results[i] = float(line)
    return results


def file_seek_next_delay_line(file_in: TextIO) -> str | None:
    pattern_comp = re.compile(r"delay of (\d+\.\d+)ms", re.IGNORECASE)
    while line := file_in.readline():
        if match := re.search(pattern_comp, line):
            return match.group(1)
    return None


def file_seek_next_data_dump(file_in) -> str | None:
    while line := file_in.readline():
        if line.strip() == "[begin memory dump]":
            return line
    return None


def file_seek_next_data_dump_and_count_it(file_in: BinaryIO) -> int:
    '''Find the size of the next data dump, and keep the read offset at the beginning of that data dump'''
    if not file_seek_next_data_dump(file_in):
        raise ValueError('did not find the start of a data dump')
    file_offset = file_in.tell() # Save offset so we can go back to it later
    count = file_count_data_dump(file_in)
    file_in.seek(file_offset) # Reset to the beginning of the original dump
    return count


def file_count_data_dump(file_in: BinaryIO) -> int:
    '''Count the number of lines from the current position in a file before encountering the "end of memory dump" marker/line.'''
    count = 0
    while line := file_in.readline().decode('ascii'):
        if line.strip() == "[end memory dump]":
            break
        count += 1
    return count


def bit_diff_within_files(file_a, file_b, num_words_to_read: int = NUM_WORDS, file_a_name="file_a", file_b_name="file_b") -> tuple[int, int, int]:
    a_weight = 0
    b_weight = 0
    different = 0
    for i in range(num_words_to_read):
        try:
            word_a = file_read_next_hex4(file_a)
        except ValueError as e:
            print(f"[ERROR] exception when reading word #{i} from \"{file_a_name}\"")
            raise e
        
        try:
            word_b = file_read_next_hex4(file_b)
        except ValueError as e:
            print(f"[ERROR] exception when reading word #{i} from \"{file_b_name}\"")
            raise e

        #print(f"#{i}: A {word_a} ; B {word_b}")
        
        a_b_diff = bit_difference(word_a, word_b)
        assert(a_b_diff >= 0)
        assert(a_b_diff <= 16)
        different += a_b_diff
        a_weight += bit_weight(word_a)
        b_weight += bit_weight(word_b)

    return a_weight, b_weight, different


def find_bit_flip(file_in, bit_index: int, bit_is_initially_one: bool) -> float | None:
    '''Return the experiment delay value (in milliseconds) at which a given bit in a memory dump flips (will be SLOW if iterating every bit and calling this function)'''
    # Allow the first argument to be a string or already-opened file
    if isinstance(file_in, str):
        with open(file_in, "rb") as arg:
            return find_bit_flip(arg, bit_index, bit_is_initially_one)

    word_index = bit_index // BITS_PER_WORD
    sub_index = bit_index % BITS_PER_WORD

    min_t: float | None = None
        
    while (delay_str := file_seek_next_delay_line(file_in)) is not None:
        t = float(delay_str)
        if (min_t is not None) and (t > min_t):
            # Skip searching for a flip in a later time than already found
            continue
        file_seek_next_data_dump(file_in)
        # Skip the first words of the memory dump before the bit index (if any)
        file_skip_hex4(file_in, word_index)
        word = file_read_next_hex4(file_in)
        bit = (word >> sub_index) & 1
        if (bit == 1) != bit_is_initially_one:
            if (min_t is None) or (t < min_t):
                # First, or Earlier bit flip detected here
                min_t = t

    # If this point is reached and min_t is still NONE, the bit never flipped in the given memory dumps
    return min_t


def file_find_bit_flips(file_in, expected_word_count: int | None = None, verbose=0, word_flip=True) -> tuple[tuple[int | None, ...], tuple[float, ...]]:
    '''Get a list for all of the bits in a memory dump(s) file and the delay times where they flip.'''

    # Allow the first argument to be a string or already-opened file
    if isinstance(file_in, str):
        with open(file_in, "rb") as arg:
            return file_find_bit_flips(arg, expected_word_count, verbose, word_flip)

    if verbose > 0:
        print("find_flip_bits()...")

    word_count = file_seek_next_data_dump_and_count_it(file_in) # file offset will be right before the first dump's data after this call

    if verbose > 0:
        print(f"For the first data dump in this file, {word_count=}")

    if (expected_word_count is not None) and (word_count != expected_word_count):
        print(f"expected {expected_word_count} words but found {word_count} instead")
        if expected_word_count > word_count:
            raise ValueError(f"actual {word_count=} < {expected_word_count=}")
        else:
            word_count = expected_word_count

    bit_count = word_count * BITS_PER_WORD

    # Read the initial data dump and use it as the starting values
    original_words: list[int] = [ file_read_next_hex4(file_in) for _ in range(word_count) ]

    # Array indeices are 0..(number of bits), that indicate at what time (if any) a bit was first seen to flip
    bit_flip_times: list[int | None] = [ None for _ in range(bit_count) ]

    # List of dump delays (maps a 'dump_index' in the 'bit_flip_times' to a delay float value)
    delays: list[float] = []

    # Iterate over all data dumps from the file...
    dump_index = 0
    while (delay_str := file_seek_next_delay_line(file_in)) is not None:
        # Save the delay for this dump index
        delays.append(float(delay_str))

        # check each new dump's word's bits
        file_seek_next_data_dump(file_in)
        for word_index in range(word_count):
            word = file_read_next_hex4(file_in)
            for bit_i in range(BITS_PER_WORD):
                # Index into the bit flip array
                bit_flip_index = (word_index * BITS_PER_WORD) + bit_i
                # Check each bit that has not yet flipped
                if bit_flip_times[bit_flip_index] is None:
                    original_bit = (original_words[word_index] >> bit_i) & 1
                    if word_flip:
                        this_bit = (word >> (15 - bit_i)) & 1
                    else:
                        this_bit = (word >> bit_i) & 1
                    bit_flipped = (original_bit != this_bit)
                    if bit_flipped:
                        bit_flip_times[bit_flip_index] = dump_index
        
        dump_index += 1

    if verbose > 0:
        print(f"Read {dump_index} dumps")

    return tuple(bit_flip_times), tuple(delays)



def file_find_bit_flips_v2(original_file, file_in, num_words: int, word_flip=True) -> np.ndarray:
    '''
    Get a list for all of the bits in a memory dump(s) file and the delay times where they first flip.
    This is an improvement over the original 'file_find_bit_flips' function.
    '''

    # Allow the first 2 arguments to be a string or already-opened file
    if isinstance(original_file, str):
        with open(original_file, 'rb') as arg:
            return file_find_bit_flips_v2(arg, file_in, num_words, word_flip)
    if isinstance(file_in, str):
        with open(file_in, "rb") as arg:
            return file_find_bit_flips_v2(original_file, arg, num_words, word_flip)

    num_bits = num_words * BITS_PER_WORD
    original_words = file_read_hex4_dump_as_words(original_file, num_words)

    # Array of bit flip time
    time_not_set = -1
    bit_flip_times = np.full(shape=num_bits, fill_value=time_not_set)

    # Iterate over all data dumps from the file...
    while (delay_str := file_seek_next_delay_line(file_in)) is not None:
        # Save the delay for this dump index
        t = float(delay_str)
        assert(t >= 0.0)
        # check each new dump's word's bits
        file_seek_next_data_dump(file_in)
        for word_index in range(num_words):
            try:
                word = file_read_next_hex4(file_in)
            except ValueError as e:
                print(f"NOTE: error at t = {t}, word_index = {word_index}")
                raise e
            for bit_i in range(BITS_PER_WORD):
                # Index into the bit flip array
                bit_flip_index = bit_i + (word_index * BITS_PER_WORD)
                # Check each bit that has not yet flipped
                recorded_time = bit_flip_times[bit_flip_index]
                if (recorded_time == time_not_set) or (t < recorded_time):
                    original_bit = (original_words[word_index] >> bit_i) & 1
                    if word_flip:
                        this_bit = (word >> (15 - bit_i)) & 1
                    else:
                        this_bit = (word >> bit_i) & 1
                    bit_flipped = (original_bit != this_bit)
                    if bit_flipped:
                        bit_flip_times[bit_flip_index] = t

    return bit_flip_times


def file_find_bit_flips_count(original_file: TextIO, file_in: TextIO, num_words: int) -> dict:
    '''
    Find how many bits flip from the original data for each dump taken at remanence time delay.
    '''
    # DO NOT allow the first 2 arguments to be a string or already-opened file
    if isinstance(original_file, str):
        raise TypeError("`original_file` given as a string instead of an open file")
    if isinstance(file_in, str):
        raise TypeError("`file_in` given as a string instead of an open file")

    original_words = file_read_hex4_dump_as_words(original_file, num_words)

    # Array of words to record if the first time a bit flipped has been recorded
    # (a 0 bit means not seen to flip yet, and a 1 bit means that it has previously flipped)
    bit_flips = np.zeros(shape=num_words, dtype="uint16")

    # Result to map a time to the number of new bit flips at that time
    flip_counts: dict[float, int] = dict()

    # Iterate over all data dumps from the file...
    while (delay_str := file_seek_next_delay_line(file_in)) is not None:
        # Save the delay for this dump index
        t = float(delay_str)
        flip_counts[t] = 0
        assert(t >= 0.0)
        # check each new dump's word's bits
        file_seek_next_data_dump(file_in)
        for word_index in range(num_words):
            try:
                word = file_read_next_hex4(file_in)
            except ValueError as e:
                print(f"NOTE: error at t = {t}, word_index = {word_index}")
                break#raise e
            original_word = original_words[word_index]
            new_mask = ~(bit_flips[word_index])
            diff = word ^ original_word
            # Get 1s only at bits that are newly flipped for this 't' time
            new_flips = diff & new_mask
            # Record these bits as flipped for next time
            bit_flips[word_index] |= new_flips
            # Add this word's flip count to the total flip count for this 't' time
            flip_counts[t] += hamming_weight(new_flips)

    return flip_counts


def file_find_bit_flips_count_with_natural_bits(original_file: TextIO, natural_file: TextIO, file_in: TextIO, num_words: int) -> dict:
    '''
    Find how many bits flip from the original data for each dump taken at remanence time delay.
    '''

    original_words = file_read_hex4_dump_as_words(original_file, num_words)
    natural_mask = file_read_hex4_dump_as_words(natural_file, num_words)

    # Array of words to record if the first time a bit flipped has been recorded
    # (a 0 bit means not seen to flip yet, and a 1 bit means that it has previously flipped)
    bit_flips = np.zeros(shape=num_words, dtype="uint16")

    # Result to map a time to the number of new bit flips at that time
    flip_counts: dict[float, int] = dict()

    # Iterate over all data dumps from the file...
    while (delay_str := file_seek_next_delay_line(file_in)) is not None:
        # Save the delay for this dump index
        t = float(delay_str)
        assert t >= 0
        flip_counts[t] = 0
        
        file_seek_next_data_dump(file_in)
        data_words = file_read_hex4_dump_as_words(file_in, num_words)

        # check each new dump's word's bits
        for i, word in enumerate(data_words):
            # Set 1s at bit flips for natural bits that are newly flipped for this 't' time
            flip_word = (word ^ original_words[i]) & (~(bit_flips[i])) & (~natural_mask[i])

            # Record these bits as flipped for next time
            bit_flips[i] |= flip_word

            # Add this word's flip count to the total flip count for this 't' time
            flip_counts[t] += hamming_weight(flip_word)

    return flip_counts


def file_skip_hex4(file_in, count: int):
    '''Skip 'count' amount of 4-digit hex words in a file'''
    for i in range(count):
        file_read_next_hex4(file_in)


# for example only
def __bit_difference(arrayA, arrayB, num_words) -> int:
    difference = 0
    for i in range(num_words):
        difference += hamming_weight(xor([i], arrayB[i]))
    return difference


def file_hamming_weight(file_in: TextIO, num_words: int) -> int:        
    return sum(map(hamming_weight, file_read_hex4_dump_as_words(file_in, num_words)))


def bit_diff_files(file_name_a: str, file_name_b: str, num_words: int=NUM_WORDS, do_print=True) -> int:
    if do_print:
        print(f"Comparing bits from data files '{file_name_a}' and '{file_name_b}'...")

    with open(file_name_a, "rb") as file_a:
        with open(file_name_b, "rb") as file_b:
            a_hw, b_hw, diff = bit_diff_within_files(file_a, file_b, num_words)

    if do_print:
        report_file_bit_diff(file_name_a, file_name_b, a_hw, b_hw, diff, num_words*BITS_PER_WORD)

    return diff


def bit_diff_files_full_ratio(file_a: str, file_b: str, num_words: int=NUM_WORDS) -> tuple[float, float, float]:
    '''With already open memory dump files, get the Hamming weights and the bits different between them (as ratios of the whole file).'''
    a_hw, b_hw, diff = bit_diff_within_files(file_a, file_b, num_words)
    num_bits = num_words * BITS_PER_WORD
    a, b, d = a_hw/num_bits, b_hw/num_bits, diff/num_bits
    assert(0.0 <= a <= 1.0)
    assert(0.0 <= b <= 1.0)
    assert(0.0 <= d <= 1.0)
    return a, b, d


def report_file_bit_diff(file_name_a: str, file_name_b: str, a_hw: int, b_hw: int, diff: int, num_bits=NUM_BITS) -> None:
    print(f"* Hamming weight for \"{file_name_a}\": {a_hw:,} ({percent(a_hw, NUM_BITS):.3f}%)")
    print(f"* Hamming weight for \"{file_name_b}\": {b_hw:,} ({percent(b_hw, NUM_BITS):.3f}%)")
    print(f"* Number of bits different between the two files: {diff:,} which is {percent(diff, num_bits):.3f}% of the given total size")


def file_skip_space(file_in):
    while file_in.peek(1)[:1].decode('ascii').isspace():
        file_in.read(1)


def file_read_next_hex4(file_in: TextIO) -> int:
    '''
    Within an already-open file for reading, skip whitespace and then read a 4-digit hex number.
    Raises a 'ValueError' if it cannot read that from the file.
    '''
    word = ""
    while True:
        c = file_in.read(1)
        if not c:
            raise ValueError(f"unexpected end of file at position #{file_in.tell()}")
        if c.isspace():
            continue
        word += c
        if len(word) == 4:
            return int(word, 16)


def file_read_next_hex4_no_error(file_in) -> int | None:
    '''Within an already-open file for reading, skip whitespace and then read a 4-digit hex number, or return 'None' upon failure'''
    try:
        return file_read_next_hex4(file_in)
    except ValueError:
        return None
    

def bit_differences(seq_a: Iterable[int], seq_b: Iterable[int]) -> int:
    return sum( bit_difference(a, b) for (a, b) in zip(seq_a, seq_b) )


def bit_difference(word_a: int, word_b: int) -> int:
    '''Find the number of bits different between two integers (in the binary representation of the numbers)'''
    return hamming_weight(xor(word_a, word_b))


def hamming_weight(x: int) -> int:
    '''Get the Hamming weight (the number of bits set to 1) of an integer. Same as the bit_weight() function.'''
    # result = 0
    # while x:
    #     result += 1
    #     x &= x - 1
    # return result
    return int(x).bit_count() # Use Python's built-in method


def bit_weight(x: int) -> int:
    '''Get the Hamming weight (the number of bits set to 1) of an integer. Same as the hamming_weight() function.'''
    return hamming_weight(x)


def check_file_increasing(file_in, num_words: int = NUM_WORDS) -> bool:
    '''check that each word value in a memory dump file is increasing'''
    print("check_file_increasing...")
    result = True
    for i in range(num_words):
        x = file_read_next_hex4(file_in)
        y = i % (16**4)
        if x != y:
            print(f"word #{i} is {x} instead of {y}")
            result = False
    print("ok")
    return result


def test():
    test_bit_difference()
    test_hex4()
    test_data_loss_percent()


def test_bit_difference():
    # print("begin testing")
    assert(bit_difference(0, 0) == 0)
    assert(bit_difference(0, 1) == 1)
    assert(bit_difference(1, 0) == 1)
    assert(bit_difference(1, 1) == 0)
    assert(bit_difference(3, 1) == 1)
    assert(bit_difference(7, 0) == 3)
    assert(bit_difference(0xf, 0) == 4)
    assert(bit_difference(0xff, 0) == 8)
    assert(bit_difference(0xffff, 0) == 16)
    assert(bit_difference(0b1010101010101010, 0b0101010101010101) == 16)
    assert(bit_difference(0xf00f, 0) == 8)
    assert(bit_difference(0xf00f, 0xffff) == 8)
    # print("end testing")


def test_hex4():
    assert(hex4(0) == '0000')
    assert(hex4(1) == '0001')
    assert(hex4(0xff) == '00FF')


def percent(numerator: float, denominator: float) -> float:
    return 100.0 * float(numerator) / float(denominator)


def hex4(x: int) -> str:
    '''
    Turn an int into a 4-digit hex string, with no prefix.
    (Requires that the int 'x' is within the range 0000 to FFFF too)
    '''
    if not isinstance(x, int): 
        raise TypeError(f"'x' should be an int, not {type(x)}")
    if x < 0x0000: 
        raise ValueError("'x' should not be negative")
    if x > 0xffff: 
        raise ValueError(f"'x' should not be bigger than 0xFFFF ({0xFFFF} in decimal)")
    return f"{x:04X}"


def hex4_i(x) -> str:
    '''Convert x to be a 4-digit hex value. Try to force x to be an int.'''
    return hex4(int(x))


def bin16(x: int) -> str:
    if not isinstance(x, int): 
        raise TypeError("'x' should be an int")
    if x < 0x0000: 
        raise ValueError("'x' should not be negative")
    if x > 0xffff: 
        raise ValueError(f"'x' should not be bigger than 0xFFFF ({0xFFFF} in decimal)")
    return f"{x:016b}"


def convert_hex_file(file_name_in: str, file_name_out: str):
    '''Convert a file that has one hex word like 0x0 per line to a denser format of 0-padded hex words'''
    print(f"Converting hex file \"{file_name_in}\" to \"{file_name_out}\"...")

    line_count = 0
    with open(file_name_in, "r") as file_in:
        with open(file_name_out, "w") as file_out:
            for line in file_in.readlines():
                if not line.strip():
                    break
                print(hex4(int(line[2:], 16)), file=file_out, end='')
                line_count += 1

    print(f"Data is {line_count} lines = {line_count*16:,} total bits")
    print("Done")


def file_read_image(file_in: TextIO, img_width: int, img_height: int) -> np.ndarray:
    '''Read 4-digit hex words from a dump file and convert it to a 2D numpy array, to act as a binary image'''        
    bits_2d = np.zeros([img_height, img_width], dtype='int16')

    num_img_bits = img_width * img_height
    num_img_words = round(num_img_bits / 16)
    
    for word_i in range(num_img_words):
        word = file_read_next_hex4(file_in)

        # break down a 16-bit word into bits
        for bit_i in range(16):
            i = (word_i * 16) + bit_i
            row = i // img_width
            col = i % img_width

            # set image bit to 1 if #'bit_i' is 1 (note that the 2d array is already initialized to 0)
            if (word & (1 << (15 - bit_i))):
                bits_2d[row, col] = 1

    return bits_2d


def convert_words_to_image(words: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    '''Convert a numpy array of words to a 2D numpy array of bits'''
    if (img_width * img_height / 16) > words.shape[0]:
        raise ValueError("the given words array is too short to contain the given image dimensions")
    
    # img_num_words = math.ceil(img_width * img_height / 16)
    bits_2d = np.zeros([img_height, img_width], dtype='bool')
    for word_i, word in enumerate(words):
        try:
            # break down a 16-bit word into bits
            for bit_i in range(16):
                i = (word_i * 16) + bit_i
                row = i // img_width
                col = i % img_width
                # set image bit to 1 if bit #'bit_i' is 1 (note that the 2d bits array is already initialized to 0)
                if (word & (1 << (15 - bit_i))):
                    bits_2d[row, col] = 1
        except IndexError:
            break
    return bits_2d


## This func. used to be used with matplotlib's ColorMap, but is now
## unused because I found a better way to visualize the bit remanence
def lerp(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    '''Linear interpolation of x from the range (in_min, in_max) to (out_min, out_max)'''
    return out_min + ((x - in_min)/(in_max - in_min)) * (out_max - out_min)


# def array_xor(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
#     if (shape1 := np.shape(arr1)) != (shape2 := np.shape(arr2)):
#         raise ValueError(f"numpy array shape mismatch: {shape1} vs {shape2}")
#     if len(shape1) != 1:
#         raise ValueError(f"numpy arrays must be 1D")
#     length = shape1[0]
#     result = np.zeros(length)
#     for i in range(length):
#         result[i] = arr1[i] ^ arr2[i]
#     return result


def data_loss_percent(pu_read: np.ndarray, data_original: np.ndarray, pu_ref: np.ndarray) -> float:
    '''
    Data Loss % = [# of set bits(Image_original XOR PU_read)]/[# of set bits(Image_original XOR PU_ref)] * 100%.
    '''
    set_bits_1 = bit_differences(data_original, pu_read)
    set_bits_2 = bit_differences(data_original, pu_ref)
    return percent(set_bits_1, set_bits_2)


def test_data_loss_percent():
    # incomplete tests (just silly)
    print('start testing data_loss_percent() ...')
    data1 = [1,2,3]
    data2 = [2,2,3]
    data3 = [3,3,3]
    assert(data_loss_percent(data1, data1, data3) == 0.0)
    assert(data_loss_percent(data2, data2, data3) == 0.0)
    assert(data_loss_percent(data1, data2, data3) == 100)
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    assert(data_loss_percent(data1, data1, data3) == 0.0)
    assert(data_loss_percent(data2, data2, data3) == 0.0)
    assert(data_loss_percent(data1, data2, data3) == 100)
    print('done testing data_loss_percent()')


def files_read_and_calc_data_loss(pu_read_file, data_original_file, pu_ref_file, num_words: int, num_word_skip: int = 0) -> float:
    '''
    Data Loss % = [# of set bits(Image_original XOR PU_read)] / [# of set bits(Image_original XOR PU_ref)] * 100%.
    Implemented for files.
    '''
    set_bits_top = 0
    set_bits_bot = 0

    ## Iterate the data words of each of the 3 files simultaneously
    for i in range(num_words):
        pu_read_word = file_read_next_hex4(pu_read_file)
        original_word = file_read_next_hex4(data_original_file)
        pu_ref_word = file_read_next_hex4(pu_ref_file)

        # Disregard the initial few words (if the argument 'num_word_skip' is more than 0)
        if i < num_word_skip:
            continue

        set_bits_top += bit_weight(xor(original_word, pu_read_word))
        set_bits_bot += bit_weight(xor(original_word, pu_ref_word))

    # assert(set_bits_top <= set_bits_bot)
    return percent(set_bits_top, set_bits_bot)


def files_data_loss_percent(data_original_file, pu_read_trials_file, pu_ref_file, num_words: int, num_word_skip: int = 0) -> list[tuple[float, float]]:
    '''Run a data loss calculation against the PUF file and multiple memory dumps that come from the Arduino'''

    ## Recursive calls to allow the first 3 arguments to be a file path string or an already open file
    if isinstance(data_original_file, str):
        # print(f"files_data_loss_percent: {data_original_file=}")
        with open(data_original_file, 'rb') as opened_file:
            return files_data_loss_percent(opened_file, pu_read_trials_file, pu_ref_file, num_words, num_word_skip)
    if isinstance(pu_read_trials_file, str):
        # print(f"files_data_loss_percent: {pu_read_trials_file=}")
        with open(pu_read_trials_file, 'rb') as opened_file:
            return files_data_loss_percent(data_original_file, opened_file, pu_ref_file, num_words, num_word_skip)
    if isinstance(pu_ref_file, str):
        # print(f"files_data_loss_percent: {pu_ref_file=}")
        with open(pu_ref_file, 'rb') as opened_file:
            return files_data_loss_percent(data_original_file, pu_read_trials_file, opened_file, num_words, num_word_skip)

    data_result: list[tuple[float, float]] = []
    val_ms = None
    while (line := pu_read_trials_file.readline().decode('ascii')):
        line = line.strip()
        if line.startswith("Beginning trial"): # once we find a start of trial, record the delay value (in milliseconds)
            assert val_ms is None
            # extract delay value from the trial line
            match = re.search(r"([0-9]+\.[0-9]+)ms", line) # Look for decimal number "<n>.<n>ms"
            assert match is not None
            val_ms = float(match.group(1))
        elif line == "[begin memory dump]": # once we find the start of the memory dump inside a trial, read files & get the bit difference
            assert val_ms is not None
            # get memory dump bits and diff it with the PUF file
            pu_ref_file.seek(0) # reset to beginning of PUF file so 'files_read_and_calc_data_loss' can read its data again
            data_original_file.seek(0) # reset to beginning of original data/image file so 'files_read_and_calc_data_loss' can read its data again
            trial_data_loss = files_read_and_calc_data_loss(pu_read_trials_file, data_original_file, pu_ref_file, num_words, num_word_skip)
            data_result.append((val_ms, trial_data_loss))
            val_ms = None

    return data_result


def files_data_loss_count(data_original_file, pu_read_trials_file, pu_ref_file, num_words: int) -> list[tuple[float, float]]:
    '''Run a data loss calculation against the PUF file and multiple memory dumps that come from the Arduino'''

    ## Recursive calls to allow the first 3 arguments to be a file path string or an already open file
    if isinstance(data_original_file, str):
        # print(f"files_data_loss_percent: {data_original_file=}")
        with open(data_original_file, 'rb') as opened_file:
            return files_data_loss_count(opened_file, pu_read_trials_file, pu_ref_file, num_words)
    if isinstance(pu_read_trials_file, str):
        # print(f"files_data_loss_count: {pu_read_trials_file=}")
        with open(pu_read_trials_file, 'rb') as opened_file:
            return files_data_loss_count(data_original_file, opened_file, pu_ref_file, num_words)
    if isinstance(pu_ref_file, str):
        # print(f"files_data_loss_percent: {pu_ref_file=}")
        with open(pu_ref_file, 'rb') as opened_file:
            return files_data_loss_count(data_original_file, pu_read_trials_file, opened_file, num_words)

    data_result: list[tuple[float, float]] = []
    val_ms = None
    while (line := pu_read_trials_file.readline().decode('ascii')):
        line = line.strip()
        if line.startswith("Beginning trial"): # once we find a start of trial, record the delay value (in milliseconds)
            assert val_ms is None
            # extract delay value from the trial line
            match = re.search(r"([0-9]+\.[0-9]+)ms", line) # Look for decimal number "<n>.<n>ms"
            assert match is not None
            val_ms = float(match.group(1))
        elif line == "[begin memory dump]": # once we find the start of the memory dump inside a trial, read files & get the bit difference
            assert val_ms is not None
            # get memory dump bits and diff it with the PUF file
            pu_ref_file.seek(0) # go to beginning of PUF file
            data_original_file.seek(0) # go to beginning of original data/image file

            # skip 1 word
            file_read_next_hex4(data_original_file)
            file_read_next_hex4(pu_read_trials_file)

            _, _, diff_from_image  = bit_diff_within_files(data_original_file, pu_read_trials_file, num_words - 1)
            # trial_data_loss = files_read_and_calc_data_loss(pu_read_trials_file, data_original_file, pu_ref_file, num_words)
            # print((val_ms, diff_from_image))
            data_result.append((val_ms, diff_from_image))
            val_ms = None

    return data_result


def file_xor(file_a, file_b, output_file, num_words: int):
    '''
    Read 'num_words' words from 2 files and write the XOR of the words to the output file.
    NOTE: adds more columns and additional data to the output file
    '''
    if isinstance(file_a, str):
        with open(file_a, 'rb') as opened_file:
            return file_xor(opened_file, file_b, output_file, num_words)
    if isinstance(file_b, str):
        with open(file_b, 'rb') as opened_file:
            return file_xor(file_a, opened_file, output_file, num_words)
    if isinstance(output_file, str):
        with open(output_file, 'w') as opened_file:
            return file_xor(file_a, file_b, opened_file, num_words)
    
    sum_diff_weight = 0

    for _ in range(num_words):
        word_a = file_read_next_hex4(file_a)
        word_b = file_read_next_hex4(file_b)
        word_out = xor(word_a, word_b)
        diff_weight = hamming_weight(word_out)
        sum_diff_weight += diff_weight
        print(hex4(word_a), end=' ', file=output_file)
        print(hex4(word_b), end=' ', file=output_file)
        print(hex4(word_out), end=' ', file=output_file)
        print(bin16(word_out), end=' ', file=output_file)
        print(diff_weight, file=output_file)

    print('total bit difference', sum_diff_weight, file=output_file)


def file_xor_basic(file_a, file_b, output_file, num_words: int):
    '''Read 'num_words' words from 2 files and write the XOR of the words to the output file.'''
    if isinstance(file_a, str):
        with open(file_a, 'rb') as opened_file:
            return file_xor_basic(opened_file, file_b, output_file, num_words)
    if isinstance(file_b, str):
        with open(file_b, 'rb') as opened_file:
            return file_xor_basic(file_a, opened_file, output_file, num_words)
    if isinstance(output_file, str):
        with open(output_file, 'w') as opened_file:
            return file_xor_basic(file_a, file_b, opened_file, num_words)

    for _ in range(num_words):
        word_a = file_read_next_hex4(file_a)
        word_b = file_read_next_hex4(file_b)
        word_out = xor(word_a, word_b)
        print(hex4(word_out), file=output_file)


def file_seek_trial_num(file_in, trial_num: int) -> int:
    '''Go to trial number 'trial_num' and return the count for how long it is (trial_num index starts at 1, not 0)'''
    if trial_num < 1:
        raise ValueError("trial_num must be >= 1")
    if trial_num > 1:
        # skip N - 1 trials
        for _ in range(trial_num - 1):
            file_seek_next_data_dump(file_in)
    num_words = file_seek_next_data_dump_and_count_it(file_in)
    return num_words


def file_load_trial_num(file_name: str, trial_num: int, max_count: int|None = None) -> np.ndarray:
    '''Find and load all data for the given trial index (index starts at 1, not 0)'''
    with open(file_name, 'rb') as file_in:
        num_words = file_seek_trial_num(file_in, trial_num)
        if max_count is not None:
            num_words = min(num_words, max_count)
        return file_load_words(file_in, num_words)
    

def file_seek_trial_for_delay(file_in, delay_ms: float) -> int:
    if delay_ms < 0.0:
        raise ValueError("delay_ms must be >= 0")
    # Look for trial with the given delay and count how long it is
    while (delay := file_seek_next_delay_line(file_in)) is not None:
        epsilon = 0.001 # maximum margin of error for float comparison
        if abs(float(delay) - delay_ms) < epsilon:
            break
    try:
        num_words = file_seek_next_data_dump_and_count_it(file_in)
    except ValueError:
        raise ValueError(f"could not trial for {delay_ms=} in the given file")
    return num_words


def file_load_words(file_in, num_words: int) -> np.ndarray:
    '''Read 'num_words' words from the input file 'file_in' and return it as a numpy array'''
    words = np.empty(num_words, dtype='uint16')
    for i in range(num_words):
        words[i] = file_read_next_hex4(file_in)
    return words


def file_write_words(file_out: TextIO, words: Iterable[int]):
    for i, w in enumerate(words):
        if i and (i % 16 == 0):
            print(file=file_out)
        print(hex4(int(w)), end=" ", file=file_out)


def file_load_trial_for_delay(file_name: str, delay_ms: float, max_count: int|None = None) -> np.ndarray:
    '''Find and load all data for the first trial with the given delay value'''
    with open(file_name, 'rb') as file_in:
        num_words = file_seek_trial_for_delay(file_in, delay_ms)
        if max_count is not None:
            num_words = min(num_words, max_count)
        return file_load_words(file_in, num_words)
    

def file_load_capture(file_in: TextIO, num_words: int) -> np.ndarray:
    """Load one data dump into a numpy array"""
    result = np.empty(num_words, dtype="uint16")

    for j in range(num_words):
        try:
            result[j] = file_read_next_hex4(file_in)
        except ValueError as e:
            print(f"(error in word #{j}) at file position #{file_in.tell()}")
            raise e

    return result
    

def file_load_captures(file_in: TextIO, num_captures: int, num_words: int) -> np.ndarray:
    """Load multiple data dumps of the same size at once, into a numpy array.
    Creates a 2D array of memory values, indexed by capture index and then by word address"""
    result = np.empty((num_captures, num_words), dtype="uint16")

    for i in range(num_captures):
        #print(f"Reading capture {i+1}/{num_captures}")
        file_seek_next_data_dump(file_in)
        for j in range(num_words):
            try:
                result[i, j] = file_read_next_hex4(file_in)
            except ValueError as e:
                print(f"(error in capture #{i}, word #{j}) at file position #{file_in.tell()}")
                raise e

    return result


def create_votes_np(captures: np.ndarray) -> np.ndarray:
    """Convert capture's memory dumps to bit array of votes for that bit being a 1"""

    num_captures = captures.shape[0]
    num_words = captures.shape[1]
    num_bits = num_words * BITS_PER_WORD

    capture_votes = np.zeros((num_captures, num_bits), dtype="uint8")

    # Create a repeating series of bit shift amounts to avoid using a nested for loop
    # (Which would iterate the range(0, BITS_PER_WORD) )
    shifts = np.tile(np.arange(BITS_PER_WORD)[::-1], reps=num_words) # note arange() is then reversed to get the correct bit-ordering

    for c in range(num_captures):
        #print(f"Combing capture {c + 1}/{num_captures}")
        cap = captures[c].repeat(BITS_PER_WORD)
        capture_votes[c] = (cap >> shifts) & 1

    return np.sum(capture_votes, axis=0)


def create_puf_np(bit_votes_for_1: np.ndarray, threshold: int) -> np.ndarray:
    "Take an array of each bit's number of votes for powering-up to a value of 1, and convert it to an array of (multi-bit) words"
    num_bits = bit_votes_for_1.shape[0]
    num_words = num_bits // BITS_PER_WORD

    # Reshape bit votes into an 2D array of N-bit rows
    bits = (bit_votes_for_1 > threshold).reshape((num_words, BITS_PER_WORD))

    # Reference: https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
    #return bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))

    return np.packbits(bits, bitorder="big").view(np.uint16).byteswap(inplace=True)