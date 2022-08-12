# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /teensyduino

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /teensyduino/build

# Include any dependencies generated for this target.
include libraries/TimerOne/CMakeFiles/TimerOne.dir/depend.make

# Include the progress variables for this target.
include libraries/TimerOne/CMakeFiles/TimerOne.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/TimerOne/CMakeFiles/TimerOne.dir/flags.make

libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.obj: libraries/TimerOne/CMakeFiles/TimerOne.dir/flags.make
libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.obj: ../libraries/TimerOne/TimerOne.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.obj"
	cd /teensyduino/build/libraries/TimerOne && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TimerOne.dir/TimerOne.cpp.obj -c /teensyduino/libraries/TimerOne/TimerOne.cpp

libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TimerOne.dir/TimerOne.cpp.i"
	cd /teensyduino/build/libraries/TimerOne && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/TimerOne/TimerOne.cpp > CMakeFiles/TimerOne.dir/TimerOne.cpp.i

libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TimerOne.dir/TimerOne.cpp.s"
	cd /teensyduino/build/libraries/TimerOne && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/TimerOne/TimerOne.cpp -o CMakeFiles/TimerOne.dir/TimerOne.cpp.s

# Object files for target TimerOne
TimerOne_OBJECTS = \
"CMakeFiles/TimerOne.dir/TimerOne.cpp.obj"

# External object files for target TimerOne
TimerOne_EXTERNAL_OBJECTS =

libraries/TimerOne/libTimerOne.a: libraries/TimerOne/CMakeFiles/TimerOne.dir/TimerOne.cpp.obj
libraries/TimerOne/libTimerOne.a: libraries/TimerOne/CMakeFiles/TimerOne.dir/build.make
libraries/TimerOne/libTimerOne.a: libraries/TimerOne/CMakeFiles/TimerOne.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libTimerOne.a"
	cd /teensyduino/build/libraries/TimerOne && $(CMAKE_COMMAND) -P CMakeFiles/TimerOne.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/TimerOne && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TimerOne.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/TimerOne/CMakeFiles/TimerOne.dir/build: libraries/TimerOne/libTimerOne.a

.PHONY : libraries/TimerOne/CMakeFiles/TimerOne.dir/build

libraries/TimerOne/CMakeFiles/TimerOne.dir/clean:
	cd /teensyduino/build/libraries/TimerOne && $(CMAKE_COMMAND) -P CMakeFiles/TimerOne.dir/cmake_clean.cmake
.PHONY : libraries/TimerOne/CMakeFiles/TimerOne.dir/clean

libraries/TimerOne/CMakeFiles/TimerOne.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/TimerOne /teensyduino/build /teensyduino/build/libraries/TimerOne /teensyduino/build/libraries/TimerOne/CMakeFiles/TimerOne.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/TimerOne/CMakeFiles/TimerOne.dir/depend

