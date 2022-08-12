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
include libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/depend.make

# Include the progress variables for this target.
include libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/flags.make

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/flags.make
libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj: ../libraries/FreqMeasureMulti/FreqMeasureMulti.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj -c /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMulti.cpp

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.i"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMulti.cpp > CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.i

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.s"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMulti.cpp -o CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.s

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/flags.make
libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj: ../libraries/FreqMeasureMulti/FreqMeasureMultiIMXRT.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj -c /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMultiIMXRT.cpp

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.i"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMultiIMXRT.cpp > CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.i

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.s"
	cd /teensyduino/build/libraries/FreqMeasureMulti && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/FreqMeasureMulti/FreqMeasureMultiIMXRT.cpp -o CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.s

# Object files for target FreqMeasureMulti
FreqMeasureMulti_OBJECTS = \
"CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj" \
"CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj"

# External object files for target FreqMeasureMulti
FreqMeasureMulti_EXTERNAL_OBJECTS =

libraries/FreqMeasureMulti/libFreqMeasureMulti.a: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMulti.cpp.obj
libraries/FreqMeasureMulti/libFreqMeasureMulti.a: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/FreqMeasureMultiIMXRT.cpp.obj
libraries/FreqMeasureMulti/libFreqMeasureMulti.a: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/build.make
libraries/FreqMeasureMulti/libFreqMeasureMulti.a: libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libFreqMeasureMulti.a"
	cd /teensyduino/build/libraries/FreqMeasureMulti && $(CMAKE_COMMAND) -P CMakeFiles/FreqMeasureMulti.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/FreqMeasureMulti && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FreqMeasureMulti.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/build: libraries/FreqMeasureMulti/libFreqMeasureMulti.a

.PHONY : libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/build

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/clean:
	cd /teensyduino/build/libraries/FreqMeasureMulti && $(CMAKE_COMMAND) -P CMakeFiles/FreqMeasureMulti.dir/cmake_clean.cmake
.PHONY : libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/clean

libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/FreqMeasureMulti /teensyduino/build /teensyduino/build/libraries/FreqMeasureMulti /teensyduino/build/libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/FreqMeasureMulti/CMakeFiles/FreqMeasureMulti.dir/depend
