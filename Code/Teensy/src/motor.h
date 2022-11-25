#ifndef MOTOR_H
#define MOTOR_H

#include "Arduino.h"

#define BALL_FWD 14
#define BALL_BWD 15
#define ROT_FWD 23
#define ROT_BWD 24

namespace motor {
    struct MotorCommand {
        int fwd;
        int bwd;
    };

    MotorCommand generateCommand(double in);
    void driveMotor(MotorCommand cmd, int forwardPin, int backwardPin);


}
#endif