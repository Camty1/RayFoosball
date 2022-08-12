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
include libraries/LedControl/CMakeFiles/LedControl.dir/depend.make

# Include the progress variables for this target.
include libraries/LedControl/CMakeFiles/LedControl.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/LedControl/CMakeFiles/LedControl.dir/flags.make

libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.obj: libraries/LedControl/CMakeFiles/LedControl.dir/flags.make
libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.obj: ../libraries/LedControl/src/LedControl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.obj"
	cd /teensyduino/build/libraries/LedControl && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LedControl.dir/src/LedControl.cpp.obj -c /teensyduino/libraries/LedControl/src/LedControl.cpp

libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LedControl.dir/src/LedControl.cpp.i"
	cd /teensyduino/build/libraries/LedControl && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/LedControl/src/LedControl.cpp > CMakeFiles/LedControl.dir/src/LedControl.cpp.i

libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LedControl.dir/src/LedControl.cpp.s"
	cd /teensyduino/build/libraries/LedControl && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/LedControl/src/LedControl.cpp -o CMakeFiles/LedControl.dir/src/LedControl.cpp.s

# Object files for target LedControl
LedControl_OBJECTS = \
"CMakeFiles/LedControl.dir/src/LedControl.cpp.obj"

# External object files for target LedControl
LedControl_EXTERNAL_OBJECTS =

libraries/LedControl/libLedControl.a: libraries/LedControl/CMakeFiles/LedControl.dir/src/LedControl.cpp.obj
libraries/LedControl/libLedControl.a: libraries/LedControl/CMakeFiles/LedControl.dir/build.make
libraries/LedControl/libLedControl.a: libraries/LedControl/CMakeFiles/LedControl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libLedControl.a"
	cd /teensyduino/build/libraries/LedControl && $(CMAKE_COMMAND) -P CMakeFiles/LedControl.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/LedControl && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LedControl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/LedControl/CMakeFiles/LedControl.dir/build: libraries/LedControl/libLedControl.a

.PHONY : libraries/LedControl/CMakeFiles/LedControl.dir/build

libraries/LedControl/CMakeFiles/LedControl.dir/clean:
	cd /teensyduino/build/libraries/LedControl && $(CMAKE_COMMAND) -P CMakeFiles/LedControl.dir/cmake_clean.cmake
.PHONY : libraries/LedControl/CMakeFiles/LedControl.dir/clean

libraries/LedControl/CMakeFiles/LedControl.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/LedControl /teensyduino/build /teensyduino/build/libraries/LedControl /teensyduino/build/libraries/LedControl/CMakeFiles/LedControl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/LedControl/CMakeFiles/LedControl.dir/depend

