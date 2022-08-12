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
include libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/depend.make

# Include the progress variables for this target.
include libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/flags.make

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/flags.make
libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj: ../libraries/LiquidCrystal/LiquidCrystal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj -c /teensyduino/libraries/LiquidCrystal/LiquidCrystal.cpp

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.i"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/LiquidCrystal/LiquidCrystal.cpp > CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.i

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.s"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/LiquidCrystal/LiquidCrystal.cpp -o CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.s

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/flags.make
libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj: ../libraries/LiquidCrystal/src/LiquidCrystal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj -c /teensyduino/libraries/LiquidCrystal/src/LiquidCrystal.cpp

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.i"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/LiquidCrystal/src/LiquidCrystal.cpp > CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.i

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.s"
	cd /teensyduino/build/libraries/LiquidCrystal && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/LiquidCrystal/src/LiquidCrystal.cpp -o CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.s

# Object files for target LiquidCrystal
LiquidCrystal_OBJECTS = \
"CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj" \
"CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj"

# External object files for target LiquidCrystal
LiquidCrystal_EXTERNAL_OBJECTS =

libraries/LiquidCrystal/libLiquidCrystal.a: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/LiquidCrystal.cpp.obj
libraries/LiquidCrystal/libLiquidCrystal.a: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/src/LiquidCrystal.cpp.obj
libraries/LiquidCrystal/libLiquidCrystal.a: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/build.make
libraries/LiquidCrystal/libLiquidCrystal.a: libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libLiquidCrystal.a"
	cd /teensyduino/build/libraries/LiquidCrystal && $(CMAKE_COMMAND) -P CMakeFiles/LiquidCrystal.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/LiquidCrystal && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LiquidCrystal.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/build: libraries/LiquidCrystal/libLiquidCrystal.a

.PHONY : libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/build

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/clean:
	cd /teensyduino/build/libraries/LiquidCrystal && $(CMAKE_COMMAND) -P CMakeFiles/LiquidCrystal.dir/cmake_clean.cmake
.PHONY : libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/clean

libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/LiquidCrystal /teensyduino/build /teensyduino/build/libraries/LiquidCrystal /teensyduino/build/libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/LiquidCrystal/CMakeFiles/LiquidCrystal.dir/depend

