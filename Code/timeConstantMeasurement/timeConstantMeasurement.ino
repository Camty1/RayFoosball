#include <SPI.h>

#define NUM_SAMPLES 10
#define TIME_STEPS 300

IntervalTimer clock100;

double u;
int counter = 0;
int state = 0;
int measurement = 0;
int clkFlag = 0;

int motorCW = 0;
int motorCCW = 1;
int encoderCS = 26;

double motorMagnitudes[NUM_SAMPLES];

void setup() {

    for (int i = 0; i < NUM_SAMPLES; i++) {
        motorMagnitudes[i] = ((double) (i+1))/((double) NUM_SAMPLES);
    }
    
    // Pin setup
    pinMode(encoderCS, OUTPUT);
    digitalWrite(encoderCS, HIGH);
    
    // Communication protocol init
    SPI.begin();
    Serial.begin(115200);

    // Setup encoder chip
    digitalWrite(encoderCS, LOW);
    SPI.transfer(0x88);
    SPI.transfer(0x03);
    digitalWrite(encoderCS, HIGH);

    // Begin interval timer
    clock100.begin(clk, 1000);
}

void loop() {
    driveMotor(u);
    if (measurement == NUM_SAMPLES) {
        state = 6;
    }
    if (clkFlag) {
        switch(state) {
            case 0:
                u = 0.0;
                printState();
                counter ++;
                break;
            case 1:
                u = motorMagnitudes[measurement];
                printState();
                counter ++;
                break;
            case 2: 
                u = 0.0;
                printState();
                counter ++;
                break;
            case 3:
                u = -motorMagnitudes[measurement];
                printState();
                counter ++;
                break;
            case 4:
                u = 0.0;
                printState();
                counter ++;
                break;
            case 5:
                u = 0.0;
                measurement ++;
                state = 1;
                counter = 0;
            default:
                u = 0.0;
                break;
        }
        if (counter >= TIME_STEPS) {
            state ++;
            counter = 0;
        }
        clkFlag = 0;
    }
}

void clk() {
    clkFlag = 1;
}

void driveMotor(double u) {
    
    int magnitude = (int)(fabs(u) * 255);
    
    if (u < 0) {
        analogWrite(motorCW, 0);
        analogWrite(motorCCW, magnitude);
    }
    else if (u > 0) {
        analogWrite(motorCW, magnitude);
        analogWrite(motorCCW, 0);
    }
    else {
        analogWrite(motorCW, 0);
        analogWrite(motorCCW, 0);
    }

}

void printState() {
    unsigned long t = micros();
    long count = getEncoderData(encoderCS);
    Serial.print(t);
    Serial.print(",");
    Serial.print(u, 3);
    Serial.print(",");
    Serial.println(count);
}

long getEncoderData(int CS) {
    long dataOut = 0;
    uint8_t dataByte; 
    // Handle SPI CS
    digitalWrite(CS, LOW);
    
    // Get encoder data
    SPI.transfer(0x60);
    for (int i = 0; i < 4; i++) {
        dataByte = SPI.transfer(0x00);
        dataOut = dataOut << 8;
        dataOut = dataOut | (long)dataByte;
    }
    digitalWrite(CS, HIGH);
    
    return dataOut;
}
