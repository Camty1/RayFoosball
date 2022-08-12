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
include libraries/LedDisplay/CMakeFiles/LedDisplay.dir/depend.make

# Include the progress variables for this target.
include libraries/LedDisplay/CMakeFiles/LedDisplay.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/LedDisplay/CMakeFiles/LedDisplay.dir/flags.make

libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj: libraries/LedDisplay/CMakeFiles/LedDisplay.dir/flags.make
libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj: ../libraries/LedDisplay/LedDisplay.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj"
	cd /teensyduino/build/libraries/LedDisplay && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj -c /teensyduino/libraries/LedDisplay/LedDisplay.cpp

libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LedDisplay.dir/LedDisplay.cpp.i"
	cd /teensyduino/build/libraries/LedDisplay && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/LedDisplay/LedDisplay.cpp > CMakeFiles/LedDisplay.dir/LedDisplay.cpp.i

libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LedDisplay.dir/LedDisplay.cpp.s"
	cd /teensyduino/build/libraries/LedDisplay && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/LedDisplay/LedDisplay.cpp -o CMakeFiles/LedDisplay.dir/LedDisplay.cpp.s

# Object files for target LedDisplay
LedDisplay_OBJECTS = \
"CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj"

# External object files for target LedDisplay
LedDisplay_EXTERNAL_OBJECTS =

libraries/LedDisplay/libLedDisplay.a: libraries/LedDisplay/CMakeFiles/LedDisplay.dir/LedDisplay.cpp.obj
libraries/LedDisplay/libLedDisplay.a: libraries/LedDisplay/CMakeFiles/LedDisplay.dir/build.make
libraries/LedDisplay/libLedDisplay.a: libraries/LedDisplay/CMakeFiles/LedDisplay.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libLedDisplay.a"
	cd /teensyduino/build/libraries/LedDisplay && $(CMAKE_COMMAND) -P CMakeFiles/LedDisplay.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/LedDisplay && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LedDisplay.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/LedDisplay/CMakeFiles/LedDisplay.dir/build: libraries/LedDisplay/libLedDisplay.a

.PHONY : libraries/LedDisplay/CMakeFiles/LedDisplay.dir/build

libraries/LedDisplay/CMakeFiles/LedDisplay.dir/clean:
	cd /teensyduino/build/libraries/LedDisplay && $(CMAKE_COMMAND) -P CMakeFiles/LedDisplay.dir/cmake_clean.cmake
.PHONY : libraries/LedDisplay/CMakeFiles/LedDisplay.dir/clean

libraries/LedDisplay/CMakeFiles/LedDisplay.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/LedDisplay /teensyduino/build /teensyduino/build/libraries/LedDisplay /teensyduino/build/libraries/LedDisplay/CMakeFiles/LedDisplay.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/LedDisplay/CMakeFiles/LedDisplay.dir/depend

