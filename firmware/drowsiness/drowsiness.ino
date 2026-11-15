// vigia drowsy: alarm board for the eyes-closed demo.
// NOT compiled or tested on a board (no Arduino was available). Not a certified safety device.
//
// Serial, 9600 baud, one line per message from the computer:
//   OK     green LED on, red LED and buzzer off
//   ALARM  red LED and buzzer on, green LED off
//   HB     heartbeat, sent every 0.5 s (the computer repeats the state after it)
// No line for 2 s: both LEDs blink and the buzzer stays off, so the driver knows nothing is watching.

const int GREEN_LED = 7;
const int RED_LED = 8;
const int BUZZER = 9;          // active buzzer: HIGH = sound
const unsigned long LINK_TIMEOUT_MS = 2000;
const unsigned long BLINK_MS = 250;

char line[16];
byte len = 0;
bool alarm = false;
unsigned long lastLine = 0;
bool everHeard = false;

void show(bool green, bool red, bool buzz) {
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
  digitalWrite(BUZZER, buzz ? HIGH : LOW);
}

void handle(const char *msg) {
  lastLine = millis();
  everHeard = true;
  if (strcmp(msg, "ALARM") == 0) alarm = true;
  else if (strcmp(msg, "OK") == 0) alarm = false;
  // "HB" (or anything else) only proves the link is alive
}

void setup() {
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  show(false, false, false);
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
  if (!everHeard || millis() - lastLine > LINK_TIMEOUT_MS) {
    bool on = (millis() / BLINK_MS) % 2 == 0;
    show(on, on, false);  // link lost: never sound the buzzer on a guess
  } else {
    show(!alarm, alarm, alarm);
  }
}
