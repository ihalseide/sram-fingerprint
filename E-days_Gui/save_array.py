import numpy as np
import time
import re

def main(thread_object):

    time.sleep(3)
    print("Made it")
    should_close = False
    if not thread_object.com_port.is_open:
        thread_object.com_port.open()
        should_close = True

    # Estimate how many characters we expect:
    # 262,144 words × 4 hex chars = ~1,048,576 characters
    # Add whitespace (~20% overhead) → read more
    expected_chars = 1_300_000

    print("Reading SRAM dump...")
    hex_dump = ""

    try:
        start_time = time.time()
        while len(hex_dump) < expected_chars:
            chunk = thread_object.com_port.read(1024).decode('utf-8', errors='ignore')
            hex_dump += chunk
            if time.time() - start_time > 10:  # Safety timeout (10 seconds)
                break
    except Exception as e:
        print("Error while reading:", e)

    if should_close:
        thread_object.com_port.close()

    words = re.findall(r"[a-fA-F0-9]{4}", hex_dump)
    word_array = [int(word, 16) for word in words]

    print(f"Captured {len(word_array)} words")
    return np.array(word_array, dtype=np.uint16)
