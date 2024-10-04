// This code is for dumping all the SRAM memory contents, using the Arduino DUE.
// Author: INH


// SRAM memory size (the number of 16-bit words)
constexpr uint32_t NUM_WORDS = 1 << 18; // equals 2^18 = 262144, for 18 address bits

// Below are some SRAM control pin definitions (mappings for the PCB setup).
// The 'n' and '!' means that the pin is active low instead of active high.
constexpr unsigned int
 pin_nCE = 7,  // !chip enable
 pin_nOE = 47,  // !output enable
 pin_nBHE = 46, // !byte high enable
 pin_nBLE = 45, // !byte low enable
 pin_nWE = 8, // !write enable
 pin_nVccEnable = 53; // pin for the MOSFET controlling VCC to the SRAM chip

// Arduino pin numbers for the A0-A17 pins on the Cypress SRAM chip (18 address bits)
constexpr unsigned int AddressPinCount = 18;
constexpr unsigned int addressPins[] = { 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 48, 49, 50, }; // mappings for the PCB setup

// Arduino pin numbers for the IO pins, I/O_0 through I/O_15, on the Cypress SRAM chip
constexpr unsigned int DataPinCount = 16;
constexpr unsigned int dataPins[] = { 22, 23, 24, 25, 26, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, }; // mappings for the PCB setup

constexpr unsigned int unused_analog_pin = A0;
static uint16_t wordStorage[NUM_WORDS/8]; // This is the biggest that I can make it
unsigned long wordStorageCount = 0;
static bool chipOk = false;


// Put SRAM chip control pins in the right mode (input or output)
void setupControlPins() {
  pinMode(pin_nCE, OUTPUT);
  pinMode(pin_nOE, OUTPUT);
  pinMode(pin_nBHE, OUTPUT);
  pinMode(pin_nBLE, OUTPUT);
  pinMode(pin_nWE, OUTPUT);
  pinMode(pin_nVccEnable, OUTPUT);
}


// Make the pins for the SRAM address lines be outputs
void setupAddressPins() {
  for (int i = 0; i < AddressPinCount; i++) {
    pinMode(addressPins[i], OUTPUT);
  }
}


// Set the correct SRAM address pin logic values for a given address into the SRAM memory.
void setAddressPins(uint32_t address) {
  // Error check
  if (address >= NUM_WORDS) {
    Serial.print("[ERROR]: invalid SRAM address: ");
    Serial.println(address, HEX);
  }
  for (uint32_t i = 0; i < AddressPinCount; i++) {
    digitalWrite(addressPins[i], address & (1UL << i) );
  }
}


// Set the SRAM's data I/O pins to input mode and read them.
// Result is the 16-bit data word read from combining the data pins.
uint16_t readDataPins() {
  // Setup control pins
  digitalWrite(pin_nWE, HIGH); // disable write
  digitalWrite(pin_nBHE, LOW); // enable high byte
  digitalWrite(pin_nBLE, LOW); // enable low byte
  digitalWrite(pin_nOE, LOW); // enable output

  // Actually read the pin values
  uint16_t result = getDataPins();
  
  // Cleanup control pins
  digitalWrite(pin_nWE, HIGH); // disable writing (still)
  digitalWrite(pin_nOE, HIGH); // disable output

  return result;
}


// Set the SRAM's I/O pins to output mode and write a value to them.
void writeDataPins(uint16_t value) {
  // Setup control pins
  digitalWrite(pin_nWE, LOW); // enable writing
  digitalWrite(pin_nBHE, LOW); // enable high byte
  digitalWrite(pin_nBLE, LOW); // enable low byte
  digitalWrite(pin_nOE, HIGH); // disable output

  // Actually write the value
  setDataPins(value);

  // Cleanup control pins
  digitalWrite(pin_nWE, HIGH); // disable writing
  digitalWrite(pin_nOE, HIGH); // disable output (still)
}


// Helper function for writeDataPins.
// NOTE: this also sets the data pins I/O mode to be OUTPUT.
// Write the SRAM's I/O data pins to have values that come from the bits of the given 16-bit value.
// This is the opposite function of getDataPins.
void setDataPins(uint16_t value) {
  dataPinMode(OUTPUT);
  for (uint16_t i = 0; i < DataPinCount; i++) {
    digitalWrite(dataPins[i], (value >> i) & 1);
  }
  dataPinMode(INPUT);
}


// Helper function for readDataPins.
// Read the SRAM's I/O data pins and combine the bits from them into a 16-bit value.
// This is the opposite function of setDataPins.
uint16_t getDataPins() {
  dataPinMode(INPUT);
  uint16_t result = 0;
  for (uint16_t i = 0; i < DataPinCount; i++) {
    result |= (!!digitalRead(dataPins[i])) << i; // the '!!' "operator" forces the value to be '0' or '1'
  }
  return result;
}


// Helper function to set all of the data pins to be either INPUT or OUTPUT.
void dataPinMode(uint32_t mode) {
  for (int i = 0; i < DataPinCount; i++) {
    pinMode(dataPins[i], mode);
  }
}


// Read one 16-bit value from an 18-bit address in SRAM
uint16_t readWord(uint32_t address) {
  setAddressPins(address);
  return readDataPins();
}


// Write one 16-bit value to an 18-bit address in SRAM
void writeWord(uint32_t address, uint16_t value) {
  setAddressPins(address);
  writeDataPins(value);
  // Set address pins to be idle at the last address
  // setAddressPins(NUM_WORDS - 1);
}


// Print a 4-digit hex word, padded with leading zeros.
void printWordHex4(uint16_t word) {
  if (word < 0x10) { Serial.print("000"); }
  else if (word < 0x100) { Serial.print("00"); }
  else if (word < 0x1000) { Serial.print("0"); }
  Serial.print(word, HEX);
}


// Get a newline-terminated line of input from the serial stream.
// - Buffer usually will include the terminating '\n' newline character.
// - More than 'bufferSize' characters may be consumed if backspace characters are received.
// - Backspace characters are handled and remove the previous character from the buffer.
// - Allows the 'buffer' to be nullptr, and of course no data will be collected then.
void serialPromptLine(char *buffer, int bufferSize) {
  Serial.flush();

  // Temporarily disable serial timeout
  auto timeoutBefore = Serial.getTimeout();
  Serial.setTimeout(-1);

  int i = 0;
  char lastChar = 0;
  while (((!buffer && !bufferSize) || i < bufferSize - 1) && lastChar != '\n') {
    while (!Serial.available()) { /* pass */ }
    lastChar = Serial.read();

    // backspace
    if (lastChar == '\b' && i > 0) {
      // echo back
      Serial.print(lastChar);
      Serial.flush();
      i--;
      continue;
    }

    if (buffer && bufferSize) {
      buffer[i] = lastChar;
      i++;
    }

    // echo back
    Serial.print(lastChar);
    Serial.flush();
  }

  if (buffer && bufferSize) {
    buffer[i] = '\0';
  }

  // Restore previous serial timeout
  Serial.setTimeout(timeoutBefore);
}


// Get an ASCII character's digit value, for integer parsing, in a given base.
// Returns -1 if the character is invalid for the given base.
int digitValue(char c, unsigned int base = 10) {
  if ('0' <= c && c <= '9') {
    c -= '0';
  }
  else if ('A' <= c && c <= 'Z') {
    c = 10 + (c - 'A');
  }
  else if ('a' <= c && c <= 'z') {
    c = 10 + (c - 'a');
  }
  else {
    // invalid
    return -1;
  }

  if (c < 0 || c >= base) {
    // invalid
    return -1;
  }

  return c;
}


// Convert ASCII string to a positive integer value.
// Returns -1 if given an invalid string or when there is an integer overflow.
// The whole string may have trailing spaces and be valid still.
long parseInt(const char *chars, size_t length, unsigned int base) {
  long value = 0;
  for (size_t i = 0; i < length && chars[i] && !isspace(chars[i]); i++) {
    int x = digitValue(chars[i], base);
    if (x < 0) {
      // invalid digit
      return -1;
    }
    long newValue = (value * base) + x;
    if (newValue < value || (value != 0 && newValue == value)) {
      // overflow
      return -1;
    }
    value = newValue;
  }
  return value;
}


// Set/write all of the addresses in a given region of SRAM to store the given value.
void fillRangeOfSRAM(uint16_t value, uint32_t base_address, uint32_t count, unsigned int step, bool showProgress) {
  auto lastTime = millis();
  constexpr long progressInterval = 1000; // milliseconds

  if (!step) {
    // Prevent infinite loop that would happen if step == 0
    return;
  }

  for (uint32_t i = 0; i < count; i += step) {
    writeWord(base_address + i, value);

    if (showProgress && (millis() - lastTime >= progressInterval)) {
      lastTime = millis();
      Serial.print('.');
      Serial.flush();
    }
  }

  if (showProgress) {
    Serial.println();
  }
}


// Disable all voltage across the SRAM chip's pins (power off)
void turnOffSRAM() {
  // Disable chip entil the end
  digitalWrite(pin_nCE, HIGH);

  // disable VCC to chip (by setting the MOSFET pin for power to high)
  digitalWrite(pin_nVccEnable, HIGH);

  // Make all the SRAM lines are LOW, so there is no voltage sustaining the SRAM memory
  digitalWrite(pin_nOE, LOW);
  digitalWrite(pin_nWE, LOW);
  digitalWrite(pin_nBHE, LOW);
  digitalWrite(pin_nBLE, LOW);
  setAddressPins(0);
  setDataPins(0);

  // Finally, set this last line LOW
  digitalWrite(pin_nCE, LOW);
}


// Turn on the SRAM chip: voltage on, chip enabled, no output enabled.
void turnOnSRAM() {
  // Disable chip until the end
  digitalWrite(pin_nCE, HIGH);

  // enable VCC to chip
  digitalWrite(pin_nVccEnable, LOW);
  // disable chip output (because this is the default expected state for the rest of the code)
  digitalWrite(pin_nOE, HIGH);
  // chip data writing disabled
  digitalWrite(pin_nWE, HIGH);

  // Finally, re-enable the chip
  digitalWrite(pin_nCE, LOW);
}


// Turn the SRAM chip OFF and ON again, with a time delay in-between.
// This version is when the delay is given as two integers: delay in milliseconds (ms) and additional delay in microseconds (us).
void powerCycleSRAM2(uint32_t delay_ms, uint32_t delay_us) {
  // Logging, so that my Python data parsing scipt knows about all power-off events
  Serial.print("Powering cycling the SRAM for "); // "Powering" is a typo, but keep it for backwards-compatibility
  Serial.print(delay_ms);
  Serial.print("ms + ");
  Serial.print(delay_us);
  Serial.println("us");

  if (delay_ms == 0 && delay_us == 0) {
    Serial.println("note: useless 0ms delay!!!");
    return;
  }

  turnOffSRAM();
  delay(delay_ms);
  // Arduino Notes and Warnings for the 'delayMicroseconds' function:
  // - This function works very accurately in the range 3 microseconds and up to 16383.
  delayMicroseconds(delay_us);
  turnOnSRAM();
}


// Turn the SRAM chip OFF and ON again, with a time delay in-between.
// This version is when the delay is given as a real number, in milliseconds (ms).
void powerCycleSRAM1(double delay_ms) {
  if (delay_ms < 0) {
    Serial.println("error: cannot use negative delay value!");
    return;
  }
  uint32_t delay_ms_int = floor(delay_ms);
  uint32_t delay_us_int = ((long)floor(delay_ms * 1000.0)) % 1000;
  uint32_t delay_total_us = delay_ms_int * 1000 + delay_us_int;
  if (delay_total_us < 16383) {
    // Put all delay in microseconds because it is within the Arduino's microseconds accuracy limit
    powerCycleSRAM2(0, delay_total_us);
  }
  else {
    // Split the delay into milliseconds and microseconds because it would be too many microseconds to delay for accurately
    powerCycleSRAM2(delay_ms_int, delay_us_int);
  }
}


// Prompt a human on the other side of the serial monitor to provide an integer.
uint32_t promptInt(unsigned int base) {
  if (!base) { return 0; }
  while (1) {
    char digitBuffer[64] = {0};
    serialPromptLine(digitBuffer, sizeof(digitBuffer));
    auto x = parseInt(digitBuffer, strlen(digitBuffer) - 1, base); // Note: subtracts 1 from strlen because of there is a newline char at the end of the buffer
    if (x < 0) {
      Serial.print("Invalid input for a base-");
      Serial.print(base);
      Serial.println(" integer. Try again >");
      continue;
    }
    return x;
  }
}


uint32_t promptForDecimalNumber(const char *messagePrompt = nullptr) {
  if (messagePrompt && *messagePrompt) { Serial.print(messagePrompt); }
  return promptInt(10);
}


uint32_t promptForHexNumber(const char *messagePrompt = nullptr) {
  if (messagePrompt && *messagePrompt) { Serial.print(messagePrompt); }
  return promptInt(16);
}


// Read a given range of SRAM and print to serial
// NOTE: "interruptable" just means that the function will return early if an input character is received from the Serial monitor.
void dumpRangeOfSRAM(uint32_t base_address, uint32_t count, unsigned int step, bool interruptable) {
  // clear pending serial input data (so we can notice a real input from Serial data that signifies an "interrupt")
  while (interruptable && Serial.available()) {
    Serial.read();
  }

  // Prevent infinite loop that would happen if step == 0
  if (!step) {
    return;
  }
  
  for (uint32_t a = 0; a < count; a += step) {
    // Print word
    printWordHex4(readWord(base_address + a));
    Serial.println();
    // Flush serial because this seems to prevent data being lost in transmission
    Serial.flush();

    // check for user interruption
    if (interruptable && Serial.available()) {
      Serial.println("[Interrupted by serial input]");
      break;
    }
  }
}


// Read a given range of SRAM and print to serial
// NOTE: "interruptable" just means that the function will return early if an input character is received from the Serial monitor.
// (NEW version 2 uses a space as a separator instead of a newline)
void dumpRangeOfSRAM_v2(uint32_t base_address, uint32_t count, unsigned int step, bool interruptable) {
  // clear pending serial input data (so we can notice a real input from Serial data that signifies an "interrupt")
  while (interruptable && Serial.available()) {
    Serial.read();
  }

  // Prevent infinite loop that would happen if step == 0
  if (!step) {
    return;
  }
  
  for (uint32_t i = 0; i < count; i += step) {
    // Print SPACE as separator, and a newline every 16 words
    if (i != 0) {
      if (i % 16 == 0) {
        Serial.print('\n');
        Serial.flush();
      }
      else {
        Serial.print(' ');
      }
    }
    // Print word
    printWordHex4(readWord(base_address + i));
    /*
    // Flush serial because this seems to prevent data being lost in transmission
    Serial.flush();
    */
    
    // check for user interruption
    if (interruptable && Serial.available()) {
      Serial.println("[Interrupted by serial input]");
      break;
    }
  }
  Serial.println();
}


// Run a remanence experiment, lots of trials with differing SRAM power cycle time delays.
// Arguments are the loop's (start, stop, and step) delay values (in milliseconds)
void runRemanenceExperiment(double start, double stop, double step) {
  if (wordStorageCount <= 0) {
    Serial.println("error: no image available in cache");
    return;
  }
  long theWordCount = wordStorageCount;

  Serial.print("Size of memory section to use (words): ");
  Serial.println(theWordCount);
  Serial.print("Delay start (ms): ");
  Serial.println(start, 3);
  Serial.print("Delay stop (ms): ");
  Serial.println(stop, 3);
  Serial.print("Delay step (ms): ");
  Serial.println(step, 3);

  Serial.println("Press enter to continue...");
  serialPromptLine(nullptr, 0);

  for (double t_d = start; t_d <= stop; t_d += step) {
    // These lines are essential for my Python data parsing program to find the delay time for each trial.
    Serial.print("Beginning next trial with delay of ");
    Serial.print(t_d, 3);
    Serial.println("ms");

    // Write the stored/cached remanence-test image to the SRAM
    writeReceivedImageBasic();

    // Extra delay, just in case
    delay(11);

    // Restart SRAM and dump the power-up memory state
    powerCycleSRAM1(t_d);
    printSectionMemoryDump_v2(0, theWordCount, 1);

    // Extra delay, just in case
    delay(11);
  }

  Serial.println("\n\n\nDone with remanence experiment\n\n\n");
}



// Run a remanence experiment, lots of trials with differing SRAM power cycle time delays.
// Arguments are the loop's (start, stop, and step) delay values (in milliseconds)
void runRemanenceExperimentInternal(double start, double stop, double step) {
  if (wordStorageCount <= 0) {
    return;
  }
  long theWordCount = wordStorageCount;

  for (double t_d = start; t_d <= stop; t_d += step) {
    // These lines are essential for my Python data parsing program to find the delay time for each trial.
    Serial.print("Beginning next trial with delay of ");
    Serial.print(t_d, 3);
    Serial.println("ms");

    // Write the stored/cached remanence-test image to the SRAM
    writeReceivedImageBasic();

    // Extra delay, just in case
    delay(11);

    // Restart SRAM and dump the power-up memory state
    powerCycleSRAM1(t_d);
    printSectionMemoryDump_v2(0, theWordCount, 1);

    // Extra delay, just in case
    delay(11);
  }

  Serial.println("\n\n(done with 1 remanence experiment)\n\n");
}


// Run a remanence experiment, lots of trials with differing SRAM power cycle time delays.
// Arguments are the loop's (start, stop, and step) delay values (in milliseconds)
void runCustomRemanenceExperiment(void) {
  if (wordStorageCount == 0) {
    Serial.println("error: no image available in cache");
    return;
  }
  long theWordCount = wordStorageCount;

  int choice = promptForDecimalNumber("Add checks for 90, 150, or 250 (0 for none)?");
  if (choice != 0 && choice != 90 && choice != 150 && choice != 250) {
    Serial.println("error: say 90, 150, or 250");
    return;
  }

  Serial.print("Size of memory section to use (words): ");
  Serial.println(theWordCount);

  Serial.println("Press enter to continue...");
  serialPromptLine(nullptr, 0);

  // double delays[] = {
  //   1.050, 1.010, 0.890, 1.100, 0.020, 1.010,
  //   0.470, 0.910, 0.220,
  //   1, 2, 3, 4, 5, 6, 7, 8, 9,
  //   1.950, 0.510, 0.900, 0.010, 0.990, 1.500,
  //   10, 20, 30, 40, 50, 60, 70, 80, 90,
  //   1.900,  0.300, 0.410, 0.250, 0.200, 0.100,
  //   100, 200, 300, 400, 500, 600, 700, 800,
  //   0.390, 0.030, 0.600, 0.350, 0.500, 0.120, 0.090,
  //   0.100, 0.900,
  // };
  double generalDelaysRange[] = {
    // log scale
    0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090,
    0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900,
    1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 2000, 3000,
  };
  constexpr int count = (sizeof(generalDelaysRange)/sizeof(generalDelaysRange[0]));

  // Iterate the delays[] in a random order
  if (randomSequenceBegin(count)) {
    int i;
    while ((i = randomSequenceNext()) >= 0) {
      double t_d = generalDelaysRange[i];

      // These lines are essential for my Python data parsing program to find the delay time for each trial.
      Serial.print("Beginning next trial with delay of ");
      Serial.print(t_d, 3);
      Serial.println("ms");

      // Write the stored/cached remanence-test image to the SRAM
      writeReceivedImageBasic();

      // Extra delay, just in case
      delay(1000);

      // Restart SRAM and dump the power-up memory state
      powerCycleSRAM1(t_d);
      printSectionMemoryDump(0, theWordCount, 1);

      // Extra delay, just in case
      delay(1000);
    }
  }

  if (choice != 0) {
    // areas of interest for 250nm
    double delaysRange250[] = {
      270, 280, 285, 286, 287, 288, 289, 290, 291, 295, 310, 320,
      1100, 1200, 1300, 1400, 1500,
      -1,//terminator
    };

    // areas of interest for 150nm
    double delaysRange150[] = {
      2.500, 3.500, 4.500, 5.500, 6.500, 7.500, 8.500, 9.500, 10.500,
      -1,//terminator
    };

    // areas of interest for 90nm
    double delaysRange90[] = {
      0.110, 0.550,
      0.150, 0.160, 0.170, 0.180, 0.190, 0.210, 0.250, 0.350, 0.380, 0.390, 0.410, 0.420, 0.450,
      -1,//terminator
    };

    // Now, use one of the node-specific ranges of delays to try next

    double *interestRange = nullptr;
    switch (choice) {
      case 90: interestRange = delaysRange90; break;
      case 150: interestRange = delaysRange150; break;
      case 250: interestRange = delaysRange250; break;
      default: return;
    }
    // Get the chosen array length
    int count2 = 0;
    while (interestRange[count2] > 0) {
      count2++;
    }

    if (randomSequenceBegin(count2)) {
      int i;
      while ((i = randomSequenceNext()) >= 0) {
        double t_d = interestRange[i];

        // These lines are essential for my Python data parsing program to find the delay time for each trial.
        Serial.print("Beginning next trial with delay of ");
        Serial.print(t_d, 3);
        Serial.println("ms");

        // Write the stored/cached remanence-test image to the SRAM
        writeReceivedImageBasic();

        // Extra delay, just in case
        delay(1000);

        // Restart SRAM and dump the power-up memory state
        powerCycleSRAM1(t_d);
        printSectionMemoryDump(0, theWordCount, 1);

        // Extra delay, just in case
        delay(1000);
      }
    }
  }

  Serial.println("\n\n\n\aDone with remanence experiment\n\n\n");
}



// Run a CUMULATIVE remanence experiment, lots of trials with differing SRAM power cycle time delays.
// Arguments are the loop's (start, stop, and step) delay values (in milliseconds)
void runCumulativeRemanenceExperiment(double start, double stop, double step) {
  if (wordStorageCount <= 0) {
    Serial.println("error: no image available in cache");
    return;
  }
  long theWordCount = wordStorageCount;

  Serial.print("Size of memory section to use (words): ");
  Serial.println(theWordCount);
  Serial.print("Delay start (ms): ");
  Serial.println(start, 3);
  Serial.print("Delay stop (ms): ");
  Serial.println(stop, 3);
  Serial.print("Delay step (ms): ");
  Serial.println(step, 3);

  Serial.println("Press enter to continue...");
  serialPromptLine(nullptr, 0);

  // Write the stored/cached remanence-test image to the SRAM
  // ONCE only.
  writeReceivedImageBasic();

  for (double t_d = start; t_d <= stop; t_d += step) {
    // These lines are essential for my Python data parsing program to find the delay time for each trial.
    Serial.print("Beginning next trial with delay of ");
    Serial.print(t_d, 3);
    Serial.println("ms (cumulative)");

    // Extra delay, just in case
    delay(11);

    // Restart SRAM and dump the power-up memory state.
    // Do not re-write the image, and
    // only power-off for the 'step' duration (this is why it is cumulative)
    powerCycleSRAM1(step);
    printSectionMemoryDump(0, theWordCount, 1);

    // Extra delay, just in case
    delay(11);
  }

  Serial.println("\n\n\n\aDone with CUMULATIVE remanence experiment\n\n\n");
}


// RS = random sequence, for iterating an array randomly
bool rs_visited[500];
unsigned int rs_index = 0;
unsigned int rs_length = 0;
bool randomSequenceBegin(unsigned int length) {
  constexpr unsigned int maxLength = sizeof(rs_visited) / sizeof(rs_visited[0]);
  if (length > maxLength) {
    return false;
  }
  rs_length = length;

  // initialize random state
  randomSeed(analogRead(unused_analog_pin));

  rs_index = 0;

  for (int i = 0; i < rs_length; i++) {
    rs_visited[i] = false;
  }

  return true;
}
int randomSequenceNext() {
  if (rs_index >= rs_length) {
    return -1;
  }
  while (1) {
    int i = random(rs_length);
    if (!rs_visited[i]) {
      rs_visited[i] = true;
      rs_index++;
      return i;
    }
  }
}


// Get a 4-digit hex word from the Serial data stream.
// Returns an int that fits in 16-bits, or returns -1 if an exceptional input is encountered.
long serialAwaitWord(void) {
  uint16_t val = 0;
  for (int i = 0; i < 4; i++) {

    // Wait for a character
    while (!Serial.available()) { /* pass */ }
    char c = Serial.read();

    auto part = digitValue(c, 16);
    if (part < 0) {
      // invalid hex digit
      return -1;
    }

    val = (val << 4) | part;
  }
  
  return val;
}


// Get hex image data from over the Serial monitor, and save it to wordStorage, an image buffer on the Arduino.
void receiveBinaryImage(void) {
  Serial.println("Prepare to send over data for a binary image (1 bit per pixel)");

  Serial.println("Enter number of columns = image width:");
  const int numColumns = promptForDecimalNumber();

  Serial.println("Enter number of rows = image height:");
  const int numRows = promptForDecimalNumber();

  const long numPixels = numRows * numColumns;
  long numHexWords = ceil((double) numPixels / 16.0);

  if (numPixels == 0) {
    Serial.println("No data");
    return;
  }

  Serial.print("Number of bit pixels: ");
  Serial.println(numPixels);
  if (numPixels % 16 != 0) {
    Serial.println("NOTE: not a multiple of 16");
    numHexWords++;
    // return;
  }

  if (numHexWords > sizeof(wordStorage)/sizeof(wordStorage[0])) {
    Serial.println("Arduino does not have enough memory for an image of that size!");
    return;
  }

  Serial.print("Enter a total of ");
  Serial.print(numHexWords);
  Serial.println(" hex words (4 hex characters each)");
  
  // empty-out the input
  while (Serial.available()) {
    Serial.read();
  }

  wordStorageCount = 0;
  for (int i = 0; i < numHexWords; i++) {
    long long w = serialAwaitWord();
    if (w < 0) {
      // invalid hex word
      Serial.println("error: invalid hex word in image data!");
      Serial.print("(at index ");
      Serial.print(i);
      Serial.println(")");
      return;
    }
    // printWordHex4(w);
    // writeWord(i, w);
    wordStorage[i] = w;
  }
  wordStorageCount = numHexWords;

  Serial.println("\nDone receiving image data");
}


// Write the image data from this Arduino's wordStorage image buffer to the SRAM memory.
void writeReceivedImage(bool showProgress) {
  if (wordStorageCount <= 0) {
    Serial.println("error: do not have received image to save");
    return;
  }
  Serial.print("Saving a ");
  Serial.print(wordStorageCount);
  Serial.print("-word image to SRAM");
  long lastUpdate = millis(); // keep track of the last time we sent a progress update
  for (int i = 0; i < wordStorageCount; i++) {
    writeWord(i, wordStorage[i]);

    // Display progress updates over the Serial by sending dots
    if (showProgress && (millis() - lastUpdate > 500)) {
      Serial.print('.');
      lastUpdate = millis();
    }
  }

  Serial.println("\nDone saving image to SRAM");
}


// Write the image data from this Arduino's wordStorage image buffer to the SRAM memory.
void writeReceivedImageBasic(void) {
  if (wordStorageCount < 1) {
    Serial.println("error: do not have received image to save");
    return;
  }
  for (int i = 0; i < wordStorageCount; i++) {
    writeWord(i, wordStorage[i]);
  }
}


// Write a set value to some memory and check that the same value reads back.
// (A memory range starts at base, and goes up to base+count, by a step of step).
bool checkWriteAndReadBackValue(uint32_t value, uint16_t base_address, uint16_t count) {
  for (auto i = 0; i < count; ++i) {
    writeWord(base_address + i, value);
  }

  for (auto i = 0; i < count; ++i) {
    if (readWord(base_address + i) != value) {
      Serial.print("checkWriteAndReadBackValue(0x");
      Serial.print(value, 16);
      Serial.print(", 0x");
      Serial.print(base_address, 16);
      Serial.print(", 0x");
      Serial.print(count, 16);
      Serial.print(") failed at address 0x");
      Serial.println(i, 16);
      return false;
    }
  }

  return true;
}


// Test to make sure that the different address bits are properly distinguished in reading and writing.
// Writes the values of 0s, 1s, and 'i' to the address '2^i' and makes sure that it reads back the same.
bool runAddressBitTest(void) {
  // Write 0's
  for (int i = 0; i < AddressPinCount; i++) {
    writeWord(1 << i, 0);
  }
  for (int i = 0; i < AddressPinCount; i++) {
    if (readWord(1 << i) != 0) {
      Serial.print("Address pin #");
      Serial.println(i);
      return false;
    }
  }

  // Write 1's
  for (int i = 0; i < AddressPinCount; i++) {
    writeWord(1 << i, 0xFFFF);
  }
  for (int i = 0; i < AddressPinCount; i++) {
    if (readWord(1 << i) != 0xFFFF) {
      Serial.print("Address pin #");
      Serial.println(i);
      return false;
    }
  }

  // Write value of i
  for (int i = 0; i < AddressPinCount; i++) {
    writeWord(1 << i, i);
  }
  for (int i = 0; i < AddressPinCount; i++) {
    if (readWord(1 << i) != i) {
      Serial.print("Address pin #");
      Serial.println(i);
      return false;
    }
  }
  return true;
}


// Returns false if the chip is definitely not connected or not working right.
// NOTE: this overwrites data in the SRAM, so don't run this as a casual check expecting to preserve the SRAM state!
bool checkConnectedChip(void) {
  constexpr int kb = 1024;

  if (!runAddressBitTest()) {
    Serial.println("runAddressBitTest() failed");
    return false;
  }

  // Check some overlapping regions with different values
  constexpr int count1 = 5; // the first few kb
  constexpr int extent1 = 2; // overlap (in kb)
  for (int i = 0; i < count1; ++i) {
    if (!checkWriteAndReadBackValue(0x0000, i * kb, extent1 * kb)) { // check an all-0s word
      return false;
    } 

    if (!checkWriteAndReadBackValue(0xFFFF, i * kb, extent1 * kb)) { // check an all-0s word
      return false;
    }

    Serial.print('.');

    if (!checkWriteAndReadBackValue(0xAAAA, i * kb, extent1 * kb)) { // check an all-0s word
      return false;
    }

    if (!checkWriteAndReadBackValue(0x5555, i * kb, extent1 * kb)) { // check an all-0s word
      return false;
    }

    Serial.print('.');
  }

  if (!runAddressBitTest()) {
    Serial.println("runAddressBitTest() failed");
    return false;
  }

  // Write an increasing sequence and check that it reads back.
  constexpr int count2 = 2;
  for (int i = 0; i < count2 * kb; ++i) {
    writeWord(i, i);
  }
  Serial.print('.');
  for (int i = 0; i < count2 * kb; ++i) {
    if (readWord(i) != i) {
      Serial.println("Increasing sequence failed");
      return false;
    }
  }
  Serial.print('.');

  if (!runAddressBitTest()) {
    Serial.println("runAddressBitTest() failed");
    return false;
  }

  return true;
}


// Do a memory dump with the required surrounding text that my Python code will look for.
// (A memory range starts at base, and goes up to base+count, by a step of step).
void printSectionMemoryDump(uint32_t baseAddress, uint32_t count, unsigned int step) {
  Serial.println("[begin memory dump]");
  dumpRangeOfSRAM(baseAddress, count, step, false);
  Serial.println("[end memory dump]");
}


// Do a memory dump with the required surrounding text that my Python code will look for.
// (A memory range starts at base, and goes up to base+count, by a step of step).
// (New version 2, which logs the parameters)
void printSectionMemoryDump_v2(uint32_t baseAddress, uint32_t count, unsigned int step) {
  Serial.println("Starting memory dump...");
  Serial.print("* base address = "); Serial.println(baseAddress, 16);
  Serial.print("* length = "); Serial.println(count);
  Serial.print("* step = "); Serial.println(step);
  Serial.println("[begin memory dump]");
  dumpRangeOfSRAM_v2(baseAddress, count, step, false);
  Serial.println("[end memory dump]");
}


// Do multiple dumps of the SRAM memory's startup/power-up values.
void doMultipleDumps(void) {
  int cycles = promptForDecimalNumber("Number of times to power cycle and dump start-up SRAM values: ");
  if (cycles <= 0) {
    Serial.println("Invalid cycles value");
    return;
  }

  // Delay
  int ms_delay = promptForDecimalNumber("Enter the power-off delay (in ms) (which should be > max remanence time for the SRAM chip: ");
  if (ms_delay <= 0) {
    Serial.println("Invalid delay value");
    return;
  }

  // Address length/count
  int count = promptForDecimalNumber("Enter the number of addresses/size/count (starting from base address 0x0) to collect (enter 0 for entire SRAM): ");
  if (count < 0) {
    Serial.println("Invalid address count value");
    return;
  }
  if (count == 0) {
    count = NUM_WORDS;
  }

  for (int i = 0; i < cycles; i++) {
    // Extra delay
    delay(10);

    // Info
    Serial.print("Starting repetition #");
    Serial.print(i + 1);
    Serial.print(" of ");
    Serial.println(cycles);

    powerCycleSRAM2(ms_delay, 0);
    printSectionMemoryDump_v2(0, count, 1);
  }

  Serial.println("\nDone with dumping multiple SRAM startup values \a\a ");
}


// Get the Hamming Weight for an integer (the number of bits set to 1 in the unsigned binary representation)
short int intHammingWeight(unsigned long x) {
    short int hWeight = 0;
    while (x) {
      hWeight++;
      x &= x - 1;
    }
    return hWeight;
}


// Find the Hamming distance between two ints
short int intHammingDistance(unsigned long x, unsigned long y) {
  return intHammingWeight(x ^ y);
}


// Sum up the Hamming weight of a range of SRAM memory.
// (A memory range starts at base, and goes up to base+count, by a step of step).
uint32_t rangeHammingWeight(uint32_t baseAddress, uint32_t count, uint32_t step) {
  if (step == 0) {
    // Prevent entering an infinite loop
    return 0;
  }
  uint32_t hWeight = 0;
  for (uint32_t i = 0; i < count; i += step) {
    hWeight += intHammingWeight(readWord(baseAddress + i));
  }
  return hWeight;
}


// Sum up the Hamming weight of a range of SRAM memory, with progress output.
// (A memory range starts at base, and goes up to base+count, by a step of step).
uint32_t rangeHammingWeightProgress(uint32_t baseAddress, uint32_t count, uint32_t step) {
  if (step == 0) {
    // Prevent entering an infinite loop
    return 0;
  }
  long lastUpdate = millis();
  uint32_t hWeight = 0;
  for (uint32_t i = 0; i < count; i += step) {
    hWeight += intHammingWeight(readWord(baseAddress + i));

    // Print out progress updates
    if (millis() - lastUpdate > 500) {
      Serial.print('.');
      lastUpdate = millis();
    }
  }

  // End the line of progress dots
  Serial.println();

  return hWeight;
}


// Writes all-0s to the SRAM and finds at what time bits are done flipping from 0 to 1
// by doing a binary search
// Parameters `t1` and `t2` are the search's power-off start and end times, in milliseconds.
int findBitFlipStopTime(int t1, int t2) {
  // Validate arguments
  if (t1 >= t2) {
    Serial.println("Argument error: (in findBitFlipStopTime) the start time, t1, must be less than the end time, t2.");
    return 0;
  }

  // constexpr int base = 0, count = NUM_WORDS, step = 1;
  constexpr int base = 4*1024, count = 1024, step = 1; // TODO: after testing, use the whole SRAM range!

  int iterationsCompleted = 0;

  while (t1 < t2) {
    // Get Hamming Weight at off-time = t1
    fillRangeOfSRAM(0x0000, base, count, step, false);
    powerCycleSRAM1(t1);
    auto hw1 = rangeHammingWeight(base, count, step);

    // Get Hamming Weight at the midpoint of t1 and t2
    fillRangeOfSRAM(0x0000, base, count, step, false);
    int t1_2 = (t1 + t2) / 2;
    auto hw1_2 = rangeHammingWeight(base, count, step);

    // Get Hamming Weight at off-time = t2
    fillRangeOfSRAM(0x0000, base, count, step, false);
    powerCycleSRAM1(t2);
    auto hw2 = rangeHammingWeight(base, count, step);

    // I expect the HW is increasing over time, up until a point,
    // and we want to find that point where the HW stays the same...

    // Log the time and Hamming weights where this occurred.
    Serial.print("In iteration #"); Serial.println(iterationsCompleted + 1);
    Serial.print("* t1 = "); Serial.print(t1); Serial.print("hw1 = "); Serial.println(hw1);
    Serial.print("* t1_2 = "); Serial.print(t1_2); Serial.print("hw1_2 = "); Serial.println(hw1_2);
    Serial.print("* t2 = "); Serial.print(t2); Serial.print("hw2 = "); Serial.println(hw2);

    // Check that above-mentioned assumption
    if (!(hw1 <= hw1_2 && hw1_2 <= hw2)) {
      // If this happens, our assumption is wrong in this case.
      Serial.println("Hey, the Hamming weight is not always increasing!");
      return 0;
    }

    // I am unsure about this next logic.... but we'll try it
    // * Surely if (hw1_2 == hw_2), then the bits stopped flipping at or before t1_2.
    // * Otherwise, (hw1_2 < hw_2) and the bits should stop flipping at or after t1_2.

    if (hw1_2 == hw_2) {
      // Search the earlier half next iteration
      Serial.println("Earlier.");
      t2 = hw1_2;
    }
    else {
      // Search the later half in the next iteration
      Serial.println("Later.");
      t1 = hw1_2;
    }

    iterationsCompleted++;
  }

  return t2;
}


void beep(void) {
  Serial.print('\a');
}


void printChoices(void) {
  Serial.println("===== Available command choices =====");
  Serial.println("  1) dump memory range");
  Serial.println("  2) fill memory range a given value");
  Serial.println("  3) power cycle SRAM");
  Serial.println("  4) upload data to Arduino image cache");
  Serial.println("  5) write Arduino image cache to SRAM");
  Serial.println("  6) run remanence experiment 'increasing steps'");
  Serial.println("  7) do multiple power-up memory dumps");
  Serial.println("  8) sum up the Hamming weight on a memory range");
  Serial.println("  9) run remanence experiment 'custom'");
  Serial.println(" 10) run remanence experiment 'cumulative'");
  Serial.println(" 11) Find maximum bit-flip time");
  Serial.println(" 12) [currently unused]");
  Serial.println(" 13) manual power cycle SRAM");
  Serial.println(" 14) do multiple sums of Hamming Weight");
  Serial.println(" 15) do experiment from command #6 multiple times");
  Serial.println(" 16) calculate Hamming distance between two SRAM chips");
  Serial.println(" 17) do a write/power-off/read cycle multiple times");
}


void handleCommandNumber(int choice) {
  switch (choice) {
  case 1:
    {
      // dump memory
      auto base = promptForHexNumber("Base address = 0x");
      auto count = promptForDecimalNumber("Count/length = ");
      auto step = 1; //promptForDecimalNumber("Step/stride = ");
      if (base >= NUM_WORDS || count > NUM_WORDS || step < 1 || step > NUM_WORDS) {
        Serial.println("Invalid base, count, or step");
        return;
      }
      printSectionMemoryDump_v2(base, count, step);
    }  break;
  case 2:
    {
      // fill memory
      uint32_t val = promptForHexNumber("Value to fill SRAM with = 0x");
      auto base = promptForHexNumber("Base/start address = 0x");
      auto count = promptForDecimalNumber("Count/length (0 for full SRAM size) = ");
      auto step = 1; //promptForDecimalNumber("Step/stride = ");
      if (base >= NUM_WORDS || count > NUM_WORDS || step < 1 || step > NUM_WORDS) {
        Serial.println("Invalid base, count, or step");
        return;
      }
      if (count == 0) {
        count = NUM_WORDS;
      }
      fillRangeOfSRAM(val, base, count, step, true);
    } break;
  case 3:
    {
      auto off_ms = promptForDecimalNumber("Enter time to power off the SRAM for (milliseconds):");
      auto off_us = promptForDecimalNumber("Enter time to power off the SRAM for (additional microseconds):");
      powerCycleSRAM2(off_ms, off_us);
    } break;
  case 4:
    {
      receiveBinaryImage();
    } break;
  case 5:
    {
      writeReceivedImage(true);
    } break;
  case 6:
    {
      // Remanence experiment = multiple trials of write-powerCycle-read with different powerCycle delays
      auto start_ms = promptForDecimalNumber("Starting power-off value (ms) ");
      auto stop_ms = promptForDecimalNumber("Stopping power-off value (ms) ");
      auto step_us = promptForDecimalNumber("Time step (us) ");
      if (start_ms > stop_ms || step_us == 0 || stop_ms == 0) {
        Serial.println("Invalid start, stop, or step value");
        break;
      }
      auto step_ms = (double) step_us / 1000.0;
      runRemanenceExperiment(start_ms, stop_ms, step_ms);
      beep();
    } break;
  case 7:
    {
      doMultipleDumps();
    } break;
  case 8:
    {
      // Hamming weight for memory range
      auto base = promptForHexNumber("Base/start address = 0x");
      auto count = promptForDecimalNumber("Count/length = ");
      auto step = 1; //promptForDecimalNumber("Step/stride = ");
      
      if (count == 0) {
        count = NUM_WORDS;
      }

      auto result = rangeHammingWeightProgress(base, count, step);

      Serial.print("Hamming weight = ");
      Serial.println(result);
    } break;
  case 9:
    {
      runCustomRemanenceExperiment();
    } break;
  case 10:
    {
      auto start_ms = promptForDecimalNumber("Starting power-off value (ms) = ");
      auto stop_ms = promptForDecimalNumber("Stopping power-off value (ms) = ");
      auto step_us = promptForDecimalNumber("Time step (us) = ");
      if (start_ms > stop_ms || step_us == 0 || stop_ms == 0) {
        Serial.println("Invalid start, stop, or step value");
        break;
      }
      auto step_ms = (double) step_us / 1000.0;
      runCumulativeRemanenceExperiment(start_ms, stop_ms, step_ms);
    } break;
  case 11:
    {
      // Find maximum time for bits to stop flipping (do a binary search)
      Serial.println("Find maximum time for bits to stop flipping...");
      auto t1 = 1; // millis
      auto t2 = promptForDecimalNumber("Search time endpoint (ms)"); // millis
      auto result = findBitFlipStopTime(t1, t2);
      Serial.print("Bits stop flipping at time = ");
      Serial.print(result);
      Serial.println(" ms");
    } break;
  case 12:
    {
      Serial.println("Command #12 is currently unused");
    } break;
  case 13:
    {
      // Power-off SRAM manually for however long
      turnOffSRAM();
      auto startTime = micros();
      Serial.println("SRAM is now off, enter anything to turn it back on and continue...");
      
      serialPromptLine(nullptr, 0);

      turnOnSRAM();

      auto duration = micros() - startTime;
      Serial.print("Off time = ");
      Serial.print(duration);
      Serial.println(" microseconds");
    } break;
  case 14:
    {
      // Collect power-up Hamming weight multiple times
      auto off_us = promptForDecimalNumber("power-off time (us) = ");
      auto cycles = promptForDecimalNumber("number of cycles = ");
      auto base = promptForHexNumber("base address (hex) = 0x");
      auto length = promptForDecimalNumber("address count (0 for the full memory size) = ");
      if (length == 0) {
        length = NUM_WORDS;
      }
      // Reset the full SRAM to all-zeros just once
      // for (int i = 0; i < NUM_WORDS; i++) {
      //   if (readWord(i) != 0) {
      //     Serial.println("Failed!");
      //     return;
      //   }
      // }
      // Repeat the cycles for Hamming Weight
      for (int i = 0; i < cycles; i++) {
        Serial.println("Filling the SRAM...");
        fillRangeOfSRAM(0xffff, base, NUM_WORDS, 1, true);

        powerCycleSRAM1((double) off_us / 1000.0); // convert [us] to [ms]
        Serial.print(i + 1);
        Serial.println(") measuring...");
        auto result = rangeHammingWeight(base, length, 1);
        auto result2 = rangeHammingWeight(base, length, 1);
        Serial.print("Hamming weight: ");
        Serial.println(result);
        if (result2 != result) {
          Serial.println("Hamming weight #2: ");
          Serial.println(result2);
        }
        // Extra delay
        delay(100);
      }
      beep();
    } break;
  case 15:
    {
      // Run multiple runs of the remenance experiment
      if (wordStorageCount == 0) {
        Serial.println("error: no image available in cache");
        break;
      }
      // Repeat option #6 a few times
      auto start_ms = promptForDecimalNumber("Starting power-off value (ms) ");
      auto stop_ms = promptForDecimalNumber("Stopping power-off value (ms) ");
      auto step_us = promptForDecimalNumber("Time step (us) ");
      auto step_ms = (double) step_us / 1000.0;
      if (start_ms > stop_ms || step_us == 0 || stop_ms == 0) {
        Serial.println("Invalid start, stop, or step value");
        break;
      }
      auto repeatTimes = promptForDecimalNumber("Number of times to repeat this whole process: ");
      for (int i = 0; i < repeatTimes; i++) {
        Serial.print(i + 1);
        Serial.println(")");
        runRemanenceExperimentInternal(start_ms, stop_ms, step_ms);
      }
      beep();
    } break;
  case 16:
    {
      // Read some chip A data, then read another (B) one and calculate the Hamming distance between A and B.
      constexpr int numCompareWords  = 10000; // words
      double maxPercentSimilar = 0.05;

      Serial.println("Reading data from current chip...");
      uint16_t wordsA[numCompareWords];
      for (int i = 0; i < numCompareWords; i++) {
        wordsA[i] = readWord(i);
      }

      Serial.println("Put in another chip to read...");
      promptForDecimalNumber("(enter to continue) > ");

      int hammingDistanceSum = 0;
      for (int i = 0; i < numCompareWords; i++) {
        uint16_t wordB = readWord(i);
        intHammingDistance(wordsA[i], wordB);
      }
      double percentDifference = (double) hammingDistanceSum / numCompareWords;

      Serial.print("Hamming distance = ");
      Serial.print(percentDifference);
      Serial.println("%");

      if (percentDifference > maxPercentSimilar) {
        Serial.println("Are they the same chip? NO");
      }
      else {
        Serial.println("Are they the same chip? YES");
      }
    } break;
  case 17:
    {
      // Repeat a write/power-off/read cycle multiple times.
      // Like a remanence experiment, but with constant power-off time.
      int repeats = promptForDecimalNumber("Set the number of repeats: ");
      if (repeats == 0 || repeats > 999) {
        // Probably not right
        Serial.print("Not accepting repeats = ");
        Serial.println(repeats);
        break;
      }
      int delay = promptForDecimalNumber("Set a power-off time between cycles (ms): ");
      int count = promptForDecimalNumber("Use address range 0 up to: ");
      if (count == 0) {
        Serial.println("(Got 0, replacing with NUM_WORDS)");
        count = NUM_WORDS;
      }
      uint16_t value = promptForHexNumber("Set a 16-bit hex value to fill SRAM with: 0x");
      for (int i = 0; i < repeats; i++) {
        // Log trial/repeat sequence number for easier human-readability of a dump file
        Serial.println();
        Serial.print("=== Repeat ");
        Serial.print(i + 1);
        Serial.print(" of ");
        Serial.print(repeats);
        Serial.println(" ===");

        // Print my "standard" trial marker line (but the delay is always the same)
        Serial.print("Beginning next trial with delay of ");
        Serial.print(delay);
        Serial.println("ms");

        // Write the value to all SRAM addresses
        Serial.print("Filling SRAM with 0x");
        printWordHex4(value);
        fillRangeOfSRAM(value, 0, count, 1, true);

        // Power off and on again
        powerCycleSRAM1(delay);

        // Dump SRAM values
        printSectionMemoryDump_v2(0, count, 1);
      }

      // Done
      Serial.println();
      Serial.println("Done with command #17");
      beep();
    } break;
  default:
    {
      // Received an unhandled choice number
      Serial.print("Invalid command choice '");
      Serial.print(choice);
      Serial.println("'");
    } break;
  }
}


void setup() {
  Serial.begin(115200);
  Serial.println("Hello from Arduino!");

  // Setup pins to SRAM chip
  setupControlPins();
  setupAddressPins();

  // Reset SRAM chip
  turnOffSRAM();
  turnOnSRAM();

  // Check SRAM chip socket connection
  Serial.print("Now checking if the SRAM chip is in the socket correctly...");
  if (chipOk = checkConnectedChip()) {
    Serial.println("\n\n>>>> OK <<<<\n");
  }
  else {
    beep();
    Serial.println("\n\n>>>> NOT ok <<<<\n");
  }
}


void loop() {
  printChoices();

  if (!chipOk) {
    Serial.println("NOTE: chip was NOT ok at the end of setup.");
  }
  constexpr int x = 0x35;
  auto choice = promptForDecimalNumber("Enter choice number: ");
  Serial.println("---");
  handleCommandNumber(choice);
}
