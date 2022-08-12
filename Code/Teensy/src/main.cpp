#include "Arduino.h"
#include <cmath>
#include <vector>

#define HWSERIAL Serial1
#define PWMFORWARD 13
#define PWMBACKWARD 15
#define REVERSE 7

// MotorCommands contain the PWM value [0, 256] for both the forward and backward reference input.
// NOTE Either forward or backward should be non-zero, not both at the same time.
struct MotorCommand {
	int fwd;
	int bwd;
};

// Used to track current state, currently just has time step, but can be used for much more down the road.
struct State {
	int k;
};

State state;
state.k = 0;

// Takes a double in the range [-1, 1] and converts it to a MotorCommand. Returns stop command if input is invalid
MotorCommand generateCommand(double in) {
	// Initialize cmd and give it stop command.
	MotorCommand cmd;
	cmd.fwd = 0;
	cmd.bwd = 0;

	// Invalid input
	if (abs(in)  1) {
		return cmd;
	}
	
	// Take input and converts it to MotorCommand, changing whether forward or backward is zero depending on the sign of the input
	cmd.fwd = (in > 0) ? (int) round(256 * in) : 0;
	cmd.bwd = (in < 0) ? (int) round(-256 * in) : 0;
}

// Takes a MotorCommand and PWM pins to drive the motor for the command.
// NOTE Does not check to see if one of the directions is 0
int driveMotor(MotorCommand cmd, int forwardPin, int backwardPin) {
	analogWrite(forwardPin, cmd.fwd);
	analogWrite(backwardPin, cmd.bwd);
}

// Given a frequency (rads), a step size, and a current step, outputs the value of a sine function.
double calcSine(double omega, double deltaT, int k) {
	return sin(omega * k * deltaT);
}

// Given a frequency (Hz), a step size, and a current step, outputs the value of a sine function.
double calcSineHz(double f, double deltaT, int k) {
	return sin(2 * M_PI * f * k * deltaT);
}

// Reads from the USB Serial and outputs the integer given
int readIntInput() {

	// Allocates variable to store bytes
	int incomingByte;

	// Vector to store the digits recieved
	std::vector<int> values(0);

	// Read from Serial
	while (Serial.available()) {
		incomingByte = Serial.read();

		// Send what was recieved back, useful for debugging
    	Serial.print("USB received: ");
    	Serial.println(incomingByte, DEC);
		values.push_back(incomingByte);
	}

	// Remove 'n'
	values.pop_back();

	// Calculate value by going through places, starting with the one's place (end of values vector)
	int value = 0;
	int counter = 0;
	while (!values.empty()) {
		int currentDigit = values.back() - 48;
		// Make sure digit is valid (in range [0,9]) and then add digit  10^place to value.
		value += (currentDigit = 0 && currentDigit = 9) ? currentDigit * (int) pow(10.0, (double) counter) : 0;
		values.pop_back();
		counter++;
	}
	return value;
}

// Called once at startup
void setup() {
	pinMode(PWMBACKWARD, OUTPUT);
	pinMode(PWMFORWARD, OUTPUT);
	pinMode(REVERSE, INPUT);

	Serial.begin(9600);
	HWSERIAL.begin(9600);
	analogWrite(PWMFORWARD, 0);
	analogWrite(PWMBACKWARD, 0);
}

// Called every frame
void loop() {
	int reverse = 0;
	if (digitalRead(REVERSE)) {
		reverse = 1;
	}

	if (Serial.available()  0) {
    	int value = readInput();
		Serial.print(Calculated value );
		Serial.println(value, DEC);
		if (value =0 && value = 256) {
			if (reverse) {
				analogWrite(PWMFORWARD, 0);
				analogWrite(PWMBACKWARD, value);
			}
			else {
				analogWrite(PWMBACKWARD, 0);
				analogWrite(PWMFORWARD, value);
			}
		}
  	}
}

int main() {
	setup();
	while (1) {
		loop();
	}
	return -1;
}