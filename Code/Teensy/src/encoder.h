#ifndef ENCODER_H
#define ENCODER_H
#include "Arduino.h"

#define BALL_CS 10
#define ROT_CS 12

namespace encoder {
    struct EncoderData {
        long ball;
        long rot;

        void zero(EncoderData *zeroPos);
    };

    struct PositionData {
        double ball;
        double rot;

        void convert(EncoderData *encoderData);
    };

    void LS7366_Init(void);
    void getEncoderData(EncoderData* encoders);
    long encoderRead();
}

#endif