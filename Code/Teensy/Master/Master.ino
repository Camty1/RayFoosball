int LED_PIN = 13;
int pwm = 0;

void start() {
    pinMode(LED_PIN, OUTPUT);
    Serial.begin(9600);
}

void update() {
    int byte = 0;
    if (Serial.available()) {
        byte = Serial.read();
    }

    if (byte >= 0 && byte < 256) {
        pwm = byte;
    }
    analogWrite(LED_PIN, pwm);
}
