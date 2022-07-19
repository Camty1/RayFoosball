int LED_PIN = 13;

void start() {
    pinMode(LED_PIN, OUTPUT);
}

void update() {
    for (int i = 1; i <= 8; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(500*i);
        digitalWrite(LED_PIN, LOW);
        delay(500*i);
    }
}