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
include libraries/FreqCount/CMakeFiles/FreqCount.dir/depend.make

# Include the progress variables for this target.
include libraries/FreqCount/CMakeFiles/FreqCount.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/FreqCount/CMakeFiles/FreqCount.dir/flags.make

libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.obj: libraries/FreqCount/CMakeFiles/FreqCount.dir/flags.make
libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.obj: ../libraries/FreqCount/FreqCount.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.obj"
	cd /teensyduino/build/libraries/FreqCount && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FreqCount.dir/FreqCount.cpp.obj -c /teensyduino/libraries/FreqCount/FreqCount.cpp

libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FreqCount.dir/FreqCount.cpp.i"
	cd /teensyduino/build/libraries/FreqCount && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/FreqCount/FreqCount.cpp > CMakeFiles/FreqCount.dir/FreqCount.cpp.i

libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FreqCount.dir/FreqCount.cpp.s"
	cd /teensyduino/build/libraries/FreqCount && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/FreqCount/FreqCount.cpp -o CMakeFiles/FreqCount.dir/FreqCount.cpp.s

# Object files for target FreqCount
FreqCount_OBJECTS = \
"CMakeFiles/FreqCount.dir/FreqCount.cpp.obj"

# External object files for target FreqCount
FreqCount_EXTERNAL_OBJECTS =

libraries/FreqCount/libFreqCount.a: libraries/FreqCount/CMakeFiles/FreqCount.dir/FreqCount.cpp.obj
libraries/FreqCount/libFreqCount.a: libraries/FreqCount/CMakeFiles/FreqCount.dir/build.make
libraries/FreqCount/libFreqCount.a: libraries/FreqCount/CMakeFiles/FreqCount.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libFreqCount.a"
	cd /teensyduino/build/libraries/FreqCount && $(CMAKE_COMMAND) -P CMakeFiles/FreqCount.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/FreqCount && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FreqCount.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/FreqCount/CMakeFiles/FreqCount.dir/build: libraries/FreqCount/libFreqCount.a

.PHONY : libraries/FreqCount/CMakeFiles/FreqCount.dir/build

libraries/FreqCount/CMakeFiles/FreqCount.dir/clean:
	cd /teensyduino/build/libraries/FreqCount && $(CMAKE_COMMAND) -P CMakeFiles/FreqCount.dir/cmake_clean.cmake
.PHONY : libraries/FreqCount/CMakeFiles/FreqCount.dir/clean

libraries/FreqCount/CMakeFiles/FreqCount.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/FreqCount /teensyduino/build /teensyduino/build/libraries/FreqCount /teensyduino/build/libraries/FreqCount/CMakeFiles/FreqCount.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/FreqCount/CMakeFiles/FreqCount.dir/depend
