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
 pin_pmosGate = 53; // pin for the MOSFET controlling VCC to the SRAM chip

// Arduino pin numbers for the A0-A17 pins on the Cypress SRAM chip (18 address bits)
constexpr unsigned int AddressPinCount = 18;
constexpr unsigned int addressPins[] = { 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 48, 49, 50, }; // mappings for the PCB setup

// Arduino pin numbers for the IO pins, I/O_0 through I/O_15, on the Cypress SRAM chip
constexpr unsigned int DataPinCount = 16;
constexpr unsigned int dataPins[] = { 22, 23, 24, 25, 26, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, }; // mappings for the PCB setup

constexpr unsigned int unused_analog_pin = A0;
static uint16_t wordStorage[NUM_WORDS/8]; // This is the biggest that I can make it
unsigned long wordStorageCount = 0;


// Put SRAM chip control pins in the right mode (input or output)
void setupControlPins() {
  pinMode(pin_nCE, OUTPUT);
  pinMode(pin_nOE, OUTPUT);
  pinMode(pin_nBHE, OUTPUT);
  pinMode(pin_nBLE, OUTPUT);
  pinMode(pin_nWE, OUTPUT);
  pinMode(pin_pmosGate, OUTPUT);
  
  // Make the pins for the SRAM address lines be outputs
  for (int i = 0; i < AddressPinCount; i++) {
    pinMode(addressPins[i], OUTPUT);
  }
}


void setAllPinsMode(int mode) {
  pinMode(pin_nCE, mode);
  pinMode(pin_nOE, mode);
  pinMode(pin_nBHE, mode);
  pinMode(pin_nBLE, mode);
  pinMode(pin_nWE, mode);

  for (int i = 0; i < AddressPinCount; i++) {
    pinMode(addressPins[i], mode);
  }

  for (uint16_t i = 0; i < DataPinCount; i++) {
    pinMode(dataPins[i], mode);
  }
}


void enableWrite() {
  // Note: is inverted because WE is active low
  digitalWrite(pin_nWE, LOW);
}


void disableWrite() {
  // Note: is inverted because WE is active low
  digitalWrite(pin_nWE, HIGH);
}


void enableOutput() {
  // Note: is inverted because OE is active low
  digitalWrite(pin_nOE, LOW);
}


void disableOutput() {
  // Note: is inverted because OE is active low
  digitalWrite(pin_nOE, HIGH);
}


void disableChip() {
  // Note: is inverted because CE is active low
  digitalWrite(pin_nCE, HIGH);
}


void enableChip() {
  // Note: is inverted because CE is active low
  digitalWrite(pin_nCE, LOW);
}


void turnOffPMOS() {
  digitalWrite(pin_pmosGate, HIGH);
}


void turnOnPMOS() {
  digitalWrite(pin_pmosGate, LOW);
}


// Set the correct SRAM address pin logic values for a given address into the SRAM memory.
void setAddressPins(uint32_t address) {
  for (uint32_t i = 0; i < AddressPinCount; i++) {
    digitalWrite(addressPins[i], address & (1UL << i) );
  }
}


// Set the SRAM's data I/O pins to input mode and read them.
// Result is the 16-bit data word read from combining the data pins.
uint16_t readDataPins() {
  // Setup control pins
  disableWrite();
  // The two calls below aren't needed because they are kept low from the very beginning.
  digitalWrite(pin_nBHE, LOW); // enable high byte
  digitalWrite(pin_nBLE, LOW); // enable low byte
  enableOutput();

  // Actually read the pin values
  uint16_t result = getDataPins();
  
  // Cleanup control pins
  disableOutput();

  return result;
}


// Set the SRAM's I/O pins to output mode and write a value to them.
void writeDataPins(uint16_t value) {
  // Setup control pins
  disableOutput();
  // The two calls below aren't needed because they are kept low from the very beginning.
  digitalWrite(pin_nBHE, LOW); // enable high byte
  digitalWrite(pin_nBLE, LOW); // enable low byte
  enableWrite();

  // Actually write the value
  setDataPins(value);

  // Cleanup control pins
  disableWrite();
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
    result |= (!!digitalRead(dataPins[i])) << i; // the '!!' forces the value to be '0' or '1'
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


// Print a 16-bit word as a 4-digit hex text, padded with leading zeros.
// High byte comes out first.
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
    // Wait for input
    while (!Serial.available()) { /* pass */ }
    lastChar = Serial.read();

    // Handle backspace character specially
    if (lastChar == '\b' && i > 0) {
      // Don't echo back
      // Serial.print(lastChar);
      // Serial.flush();
      i--;
      continue;
    }

    if (buffer && bufferSize) {
      buffer[i] = lastChar;
      i++;
    }

    // Don't echo back
    // Serial.print(lastChar);
    // Serial.flush();
  }

  if (buffer && bufferSize) {
    buffer[i] = '\0';
  }

  // Restore previous serial timeout
  Serial.setTimeout(timeoutBefore);
}


// Get an ASCII character's digit value, for integer parsing, in a given base.
// Returns -1 if the character is invalid for the given base.
int digitValue(char c, unsigned int base) {
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


// This function is needed instead of the Arduino default because I want to use HEX values for some numbers.
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


void setAllPinsLow() {
  // Disable chip to prevent any side-effects from the next changes below
  disableChip();

  // Set all lines to be low
  digitalWrite(pin_nOE, LOW);
  digitalWrite(pin_nWE, LOW);
  digitalWrite(pin_nBHE, LOW);
  digitalWrite(pin_nBLE, LOW);
  setAddressPins(0);
  setDataPins(0);

  // Finally, set the chip enable line to LOW
  digitalWrite(pin_nCE, LOW);
}


// Disable all voltage across the SRAM chip's pins (power off)
void turnOffSRAM() {
  setAllPinsLow();
  turnOffPMOS();
}


// Turn on the SRAM chip: voltage on, chip enabled, no output enabled.
void turnOnSRAM() {
  // Disable chip until the end
  digitalWrite(pin_nCE, HIGH);

  // enable VCC to chip
  turnOnPMOS();

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
  Serial.print("Turning off the SRAM power for "); // "Powering" is a typo, but keep it for backwards-compatibility
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
// Keeps trying until a valid number is entered.
uint32_t promptInt(unsigned int base) {
  if (!base) { return 0; }
  while (1) {
    constexpr size_t BufferLength = 64;
    char digitBuffer[BufferLength] = {0};
    serialPromptLine(digitBuffer, BufferLength);
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


// Read a given range of SRAM and print to serial
// (NEW version 2 uses a space as a separator instead of a newline)
void dumpRangeOfSRAM(uint32_t base_address, uint32_t count, unsigned int step) {
  // Prevent infinite loop that would happen if step == 0
  if (!step) {
    return;
  }

  for (uint32_t i = 0; i < count; i += step) {
    if (i > 0) {
      // In ASCII hex dump, add spaces between words and add newlines after every 16 words
      if (i % 16 == 0) {
        Serial.println();
        // Flush serial because this seems to prevent data being lost in transmission
        Serial.flush();
      }
      else {
        Serial.print(' ');
      }
    }

    printWordHex4(readWord(base_address + i));    
  }
  
  Serial.println();
  Serial.flush();
}


//--------------------------------------------------
// Random sequence (RS) code
//--------------------------------------------------
// This is used for iterating an entire array in a random order.
//
// The "API" is that you call `randomSequenceBegin(N)` once, then `randomSequenceNext` N times, and then call `randomSequenceEnd`.
bool rs_isBegun = false;
unsigned int rs_index = 0; // this is a counter between 0 and rs_length, incremented to keep track of when the end has been reached.
unsigned int rs_length = 0; // indicates the end of the current random sequence
bool rs_visited[500]; // array to lookup whether an index has been visited/returned yet.

bool randomSequenceBegin(unsigned int length) {
  // To allow for this code to be simpler (using global variables) only allow one RS at a time.
  if (rs_isBegun) {
    return false;
  }
  rs_isBegun = true;
  
  constexpr unsigned int maxLength = sizeof(rs_visited) / sizeof(rs_visited[0]);
  if (length > maxLength) {
    return false;
  }
  rs_length = length;

  // initialize random state
  randomSeed(analogRead(unused_analog_pin));

  // Reset index/count to 0
  rs_index = 0;

  // Reset the array for whether each index was visited/returned yet.
  for (int i = 0; i < rs_length; i++) {
    rs_visited[i] = false;
  }

  return true;
}

// Returns the next random index, except when it returns -1 to indicate the end of the sequence.
int randomSequenceNext() {
  if (!rs_isBegun || rs_index >= rs_length) {
    return -1;
  }

  // Search for an unused index to mark and return.
  while (1) {
    int i = random(rs_length);
    if (!rs_visited[i]) {
      rs_visited[i] = true; // mark index `i` as visited
      rs_index++;
      return i;
    }
  }
}

void randomSequenceEnd() {
  if (!rs_isBegun) {
    Serial.println("PROBLEM: randomSequenceEnd() called without first reaching randomSequenceBegin()");
    Serial.end();
  }
  rs_isBegun = false;
}
//--------------------------------------------------


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


// Write a set value to some memory and check that the same value reads back.
// (A memory range starts at base, and goes up to base+count, by a step of step).
bool checkWriteAndReadBackValue(uint32_t value, uint16_t base_address, uint16_t count) {
  for (auto i = 0; i < count; ++i) {
    writeWord(base_address + i, value);
  }

  for (auto i = 0; i < count; ++i) {
    if (readWord(base_address + i) != value) {
      // Serial.print("checkWriteAndReadBackValue(0x");
      // Serial.print(value, 16);
      // Serial.print(", 0x");
      // Serial.print(base_address, 16);
      // Serial.print(", 0x");
      // Serial.print(count, 16);
      // Serial.print(") failed at address 0x");
      // Serial.println(i, 16);
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
      // Serial.print("Address pin #");
      // Serial.println(i);
      return false;
    }
  }

  // Write 1's
  for (int i = 0; i < AddressPinCount; i++) {
    writeWord(1 << i, 0xFFFF);
  }
  for (int i = 0; i < AddressPinCount; i++) {
    if (readWord(1 << i) != 0xFFFF) {
      // Serial.print("Address pin #");
      // Serial.println(i);
      return false;
    }
  }

  // Write value of i
  for (int i = 0; i < AddressPinCount; i++) {
    writeWord(1 << i, i);
  }
  for (int i = 0; i < AddressPinCount; i++) {
    if (readWord(1 << i) != i) {
      // Serial.print("Address pin #");
      // Serial.println(i);
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
    // Serial.println("runAddressBitTest() failed");
    return false;
  }

  // Check some overlapping regions with different values
  constexpr int count1 = 3; // the first few kb
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
    // Serial.println("runAddressBitTest() failed");
    return false;
  }

  // Write an increasing sequence and check that it reads back.
  constexpr int count2 = 1;
  for (int i = 0; i < count2 * kb; ++i) {
    writeWord(i, i);
  }
  Serial.print('.');
  for (int i = 0; i < count2 * kb; ++i) {
    if (readWord(i) != i) {
      // Serial.println("Increasing sequence failed");
      return false;
    }
  }
  Serial.print('.');

  if (!runAddressBitTest()) {
    // Serial.println("runAddressBitTest() failed");
    return false;
  }

  return true;
}


void setup() {
  /* Notes:
     * 250000 Baud (8E1) memory dump 250kWords took 57s (has some data errors even with parity bit)
     * 115200 Baud (8N1) memory dump 250kWords took 109s
  */
  //Serial.begin(200000, SERIAL_8E1); // some errors occurr at 200kBaud
  Serial.begin(115200, SERIAL_8E1);
  Serial.println("Hello from Arduino!");

  // Setup pins to SRAM chip
  setupControlPins();

  // These two calls ensure that the pins are in a known state.
  // * Specifically, the BLE and BHE pins should always be LOW!
  turnOffSRAM();
  turnOnSRAM();
}


void loop() {
  auto offTimeSeconds = promptForDecimalNumber();

  turnOnSRAM();
  
  if (checkConnectedChip()) {
    // If chip is connected: print "t" for True, power down for offTime, and then do data dump.
    Serial.println("t");
    powerCycleSRAM1(offTimeSeconds * 1000);
    dumpRangeOfSRAM(0, NUM_WORDS, 1);
  }
  else {
    // Chip not connected: print "f" for False
    Serial.println("f");
  }

  turnOffSRAM();
}