#include "Arduino.h"
#include "encoder.h"
#include "motor.h"
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <random>

#define TS_US 20000


// Function definitions
double calcSine(double omega, double deltaT, int k);
double calcSineHz(double f, double deltaT, int k);
double getRandomInput();
int readIntInput();
void inputMotorControl();
void sineMotorControl(double omega, int us);
double randMotorControl();