// This code is for dumping all the SRAM memory contents, using the Arduino DUE.
// Author: INH

// SRAM memory size (the number of 16-bit words)
constexpr uint32_t numWords = 1 << 18; // equals 2^18 = 262144, for 18 address bits

// Below are some SRAM control pin definitions (mappings for the PCB setup).
// The 'n' and '!' means that the pin is active low instead of active high.
constexpr unsigned int
 pin_nCE = 7,  // !chip enable
 pin_nOE = 47,  // !output enable
 pin_nBHE = 46, // !byte high enable
 pin_nBLE = 45, // !byte low enable
 pin_nWE = 8; // !write enable
constexpr unsigned int pin_n_vccEnable = 53; // pin for the MOSFET controlling VCC to the SRAM chip

// Arduino pin numbers for the A0-A17 pins on the Cypress SRAM chip (18 address bits)
constexpr unsigned int num_addressPins = 18;
//constexpr int addressPins[] = { 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, }; // mappings for my manual wired setup
constexpr unsigned int addressPins[] = { 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 48, 49, 50, }; // mappings for the PCB setup

// Arduino pin numbers for the IO pins, I/O0-I/O15, on the Cypress SRAM chip
constexpr unsigned int num_dataPins = 16;
//constexpr int dataPins[] = { 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 8, 9, 10, 11, }; // mappings for my manual wired setup
constexpr unsigned int dataPins[] = { 22, 23, 24, 25, 26, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, }; // mappings for the PCB setup

void setupControlPins() {
  pinMode(pin_nCE, OUTPUT);
  pinMode(pin_nOE, OUTPUT);
  pinMode(pin_nBHE, OUTPUT);
  pinMode(pin_nBLE, OUTPUT);
  pinMode(pin_nWE, OUTPUT);
  pinMode(pin_n_vccEnable, OUTPUT);
}

// Make the pins for the SRAM address lines be outputs
void setupAddressPins() {
  for (int i = 0; i < num_addressPins; i++) {
    pinMode(addressPins[i], OUTPUT);
  }
}

// Set the correct pin values for a given SRAM address
void setAddressPins(uint32_t address) {
  if (address > numWords) {
    Serial.print("[ERROR]: invalid SRAM address: ");
    Serial.println(address, HEX);
  }
  for (uint32_t i = 0; i < num_addressPins; i++) {
    digitalWrite(addressPins[i], address & (1UL << i) );
  }
}

// Get the data word read from combining the data pins.
uint16_t readDataPins() {
  for (int i = 0; i < num_dataPins; i++) {
    pinMode(dataPins[i], INPUT);
  }

  // this may or may not be necessary to make the default pin read value be 0...
  for (uint16_t i = 0; i < num_dataPins; i++) {
    digitalWrite(dataPins[i], LOW);
  }

  digitalWrite(pin_nCE, LOW); // enable chip (this line of code new since 2024.03.06)
  digitalWrite(pin_nBHE, LOW); // enable high byte
  digitalWrite(pin_nBLE, LOW); // enable low byte
  digitalWrite(pin_nWE, HIGH); // disable write

  digitalWrite(pin_nOE, LOW); // enable output
  delayMicroseconds(1); // (extra delay, not necessary)

  uint16_t result = 0;
  for (uint16_t i = 0; i < num_dataPins; i++) {
    result |= digitalRead(dataPins[i]) << i;
  }

  digitalWrite(pin_nOE, HIGH); // disable output
  digitalWrite(pin_nCE, HIGH); // disable chip (this line of code new since 2024.03.06)

  return result;
}

// Read one value from SRAM
uint16_t readWord(uint32_t address) {
  setAddressPins(address);
  return readDataPins();
}

// Print a 4-digit hex word, padded with leading zeros.
void printWordHex4(uint16_t word) {
  if (word < 0x10) { Serial.print("000"); }
  else if (word < 0x100) { Serial.print("00"); }
  else if (word < 0x1000) { Serial.print("0"); }
  Serial.print(word, HEX);
}

void setup() {
  Serial.begin(400000);

  setupControlPins();
  setupAddressPins();

  digitalWrite(pin_n_vccEnable, LOW); // enable VCC to chip

  delayMicroseconds(1);

  Serial.println();
  Serial.println("[start dump]");
  for (uint32_t i = 0; i < numWords; i++) {
    printWordHex4(readWord(i));
    Serial.println();
  }
  Serial.println("[end dump]");
  Serial.end();
}

void loop() {
  // pass
}
