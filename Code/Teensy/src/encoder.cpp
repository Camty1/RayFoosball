#include "encoder.h"
namespace encoder
{
    // Contains tick counts for both ball screw and rotation encoders
    struct EncoderData {
        long ball;
        long rot;

        void zero(EncoderData *zeroPos) {
            this->ball -= zeroPos->ball;
            this->rot -= zeroPos->rot;
        }

    };

    // Contains a linear and angular position doubles
    // TODO: Determine if quadrature counts are 4 * number of slits or not
    struct PositionData {
        double ball;
        double rot;

        void convert(encoder::EncoderData *encoderData) {
            this->ball = encoderData->ball/250;
            this->rot = encoderData->rot/720;
        }
    }

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
}