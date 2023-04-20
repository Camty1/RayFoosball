#include "motor.h"
namespace motor {

    // MotorCommands contain the PWM value [0, 256] for both the forward and backward reference input.
    // NOTE: Either forward or backward should be non-zero, not both at the same time.
    struct MotorCommand {
	    int fwd;
	    int bwd;
    };

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
}
