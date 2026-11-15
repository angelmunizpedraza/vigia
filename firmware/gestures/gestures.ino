// vigia hands: five LEDs, one per finger, driven by the gesture demo.
// NOT compiled or tested on a board (no Arduino was available).
//
// Serial, 9600 baud, one line per message from the computer:
//   LEDS 10110   thumb..pinky, 1 = LED on, 0 = off
// The computer sends the line when the fingers change and repeats it every 0.5 s as a heartbeat.
// No valid line for 2 s (program closed, cable pulled, computer frozen): every LED turns off.
//
// Wiring: pins 2-6 (thumb, index, middle, ring, pinky) -> 220 ohm resistor -> LED long leg; LED short leg -> GND.

const int PINS[5] = {2, 3, 4, 5, 6};
const unsigned long LINK_TIMEOUT_MS = 2000;

char line[16];
byte len = 0;
unsigned long lastLine = 0;
bool everHeard = false;

void allOff() {
  for (int i = 0; i < 5; i++) digitalWrite(PINS[i], LOW);
}

// Accepts exactly "LEDS " followed by five 0/1 characters; anything else is ignored.
void handle(const char *msg) {
  if (strlen(msg) != 10 || strncmp(msg, "LEDS ", 5) != 0) return;
  for (int i = 0; i < 5; i++) {
    if (msg[5 + i] != '0' && msg[5 + i] != '1') return;
  }
  for (int i = 0; i < 5; i++) digitalWrite(PINS[i], msg[5 + i] == '1' ? HIGH : LOW);
  lastLine = millis();
  everHeard = true;
}

void setup() {
  for (int i = 0; i < 5; i++) pinMode(PINS[i], OUTPUT);
  allOff();
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = '\0';
      handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    } else {
      len = 0;  // overlong garbage: drop it
    }
  }

  // unsigned subtraction stays correct when millis() wraps after ~49 days
  if (everHeard && millis() - lastLine > LINK_TIMEOUT_MS) {
    allOff();
    everHeard = false;  // stay off until a valid line arrives
  }
}
