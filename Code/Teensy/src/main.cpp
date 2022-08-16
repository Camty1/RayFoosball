#include "Arduino.h"
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <random>

#define PWMFORWARD 14
#define PWMBACKWARD 15
#define BALL_CS 16
#define ROT_CS 17
#define TS_US 20000

// MotorCommands contain the PWM value [0, 256] for both the forward and backward reference input.
// NOTE: Either forward or backward should be non-zero, not both at the same time.
struct MotorCommand {
	int fwd;
	int bwd;
};

// Contains tick counts for both ball screw and rotation encoders
struct EncoderData {
	long ball;
	long rot;

	void zero(EndoderData *initialState) {
		this->ball -= initialState->ball;
		this->rot -= initialState->rot;
	}

};

// Contains a linear and angular position doubles
// TODO: Determine if quadrature counts are 4 * number of slits or not
struct PositionData {
	double ball;
	double rot;

	void convert(EncoderData *encoderData) {
		this->ball = encoderData->ball/250;
		this->rot = encoderData->rot/720;
	}
}

// Function definitions
MotorCommand generateCommand(double in);
void driveMotor(MotorCommand cmd, int forwardPin, int backwardPin);
double calcSine(double omega, double deltaT, int k);
double calcSineHz(double f, double deltaT, int k);
double getRandomInput();
int readIntInput();
void inputMotorControl();
void sineMotorControl(double omega, int us);
double randMotorControl();

// Random number generator
std::default_random_engine randGen;
std::normal_distribution<double> distribution(0.0, 1/3.0);

// Interrupt timer and time step
IntervalTimer timer;
volatile int time = 0;
void stepTime() {
	time++;
	double u = randMotorControl();
	Serial.print(time);
	Serial.print(", ");
	Serial.println(u);
}

// Used to track current state, currently just has time step, but can be used for much more down the road.
// struct State {
// 	int k;
// };

void LS7366_Init(void) {
   
    // SPI initialization
    SPI.begin();
    delay(10);
   
   // Configure ball screw encoder
   digitalWrite(BALL_CS,LOW);
   SPI.transfer(0x88); 
   SPI.transfer(0x03);
   digitalWrite(BALL_CS,HIGH);

   // Configure rotation encoder
   digitalWrite(ROT_CS,LOW);
   SPI.transfer(0x88); 
   SPI.transfer(0x03);
   digitalWrite(ROT_CS,HIGH); 
}

// Uses SPI to get tick count from encoders and updates EncoderData struct
void getEncoderData(EncoderData* encoders) {

	// Get data from ball screw encoder
    digitalWrite(BALL_CS, LOW);
	encoders->ball = encoderRead();
	digitalWrite(BALL_CS, HIGH);

	// Get data from rotation encoder
	digitalWrite(ROT_CS. LOW);
	encoders->rot = encoderRead();
	digitalWrite(ROT_CS, HIGH);
    
}

// Handles communication with encoder by sending 4 empty bytes.
// NOTE: CS pin is not handled in this function.
long encoderRead() {
	// Initialize data
	long data = 0;

	// Read from encoder, combining the 4 bytes of signal into one value.
	for (int i = 0; i < 4; i++) {
		data << 8;
		data += SPI.transfer(0xFF); // Dummy bytes to read from MISO line
	}

	return data;
}

// Takes a double in the range [-1, 1] and converts it to a MotorCommand. Enforces saturation if command is too big
MotorCommand generateCommand(double in) {
	// Initialize cmd and give it stop command.
	MotorCommand cmd;
	cmd.fwd = 0;
	cmd.bwd = 0;

	// Invalid input
	if (abs(in) > 1) {
		in = in/abs(in);
	}
	
	// Take input and converts it to MotorCommand, changing whether forward or backward is zero depending on the sign of the input
	cmd.fwd = (in > 0) ? (int) round(256 * in) : 0;
	cmd.bwd = (in < 0) ? (int) round(-256 * in) : 0;

	return cmd;
}

// Takes a MotorCommand and PWM pins to drive the motor for the command.
// NOTE Does not check to see if one of the directions is 0
void driveMotor(MotorCommand cmd, int forwardPin, int backwardPin) {
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

// Returns a normally distributed input around 0 with sigma = 1/6;
double getRandomInput() {
	return distribution(randGen);
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

	// Initialize variables for value calculation
	int value = 0;
	int counter = 0;

	// Handle negative inputs
	int negative = values[0] == 45;

	// Calculate value by going through places, starting with the one's place (end of values vector)
	while (!values.empty()) {
		int currentDigit = values.back() - 48;
		// Make sure digit is valid (in range [0,9]) and then add digit  10^place to value.
		value += (currentDigit >= 0 && currentDigit <= 9) ? currentDigit * (int) pow(10.0, (double) counter) : 0;
		values.pop_back();
		counter++;
	}
	return !negative ? value : -value;
}

// Function to control motor based on user input
void inputMotorControl() {

	if (Serial.available()) {
    	int value = readIntInput();
		Serial.print("Calculated value: ");
		Serial.println(value, DEC);
		double norm = value/256;
		MotorCommand cmd = generateCommand(norm);
		driveMotor(cmd, PWMFORWARD, PWMBACKWARD);
  	}
}

// Function to control motor with sine wave input
void sineMotorControl(double omega, int us) {
	// Copying of time step will get messed up if interrrupt occurs at same time, therefore no interrupt is called
	noInterrupts();
	int k = time;
	interrupts();

	// Get time step size from microseconds
	double deltaT = us/1000000.0;
	
	// Calculate sin value
	double value = calcSine(omega, deltaT, k);

	// Drive motor
	MotorCommand cmd = generateCommand(value);
	driveMotor(cmd, PWMFORWARD, PWMBACKWARD);
}

// Function to control motor with random input (useful for SYS-ID)
double randMotorControl() {
	// Get random input for motor
	double rand = getRandomInput();

	// Drive motor
	MotorCommand cmd = generateCommand(rand);
	driveMotor(cmd, PWMFORWARD, PWMBACKWARD);
	return rand;
}

// Called once at startup
void setup() {

	// Digital pins
	pinMode(PWMBACKWARD, OUTPUT);
	pinMode(PWMFORWARD, OUTPUT);
	pinMode(BALL_CS, OUTPUT);
	pinMode(ROT_CS, OUTPUT);

	// SPI setup
	digitalWrite(BALL_CS, HIGH);
	digitalWrite(ROT_CS, HIGH);


	// USB serial
	Serial.begin(9600);

	// Turn off motors at start
	analogWrite(PWMFORWARD, 0);
	analogWrite(PWMBACKWARD, 0);

	// Time step
	// TODO: Attach interrupt to external pin for system clock signal
	timer.begin(stepTime, TS_US);
}

// Called every frame
void loop() {
	//inputMotorControl();
	//sineMotorControl(.5 * M_PI * 2, TS_US);
}

int main(void) {
	setup();
	while (1) {
		loop();
	}
	return -1;
}