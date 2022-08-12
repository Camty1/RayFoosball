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
include libraries/x10/CMakeFiles/x10.dir/depend.make

# Include the progress variables for this target.
include libraries/x10/CMakeFiles/x10.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/x10/CMakeFiles/x10.dir/flags.make

libraries/x10/CMakeFiles/x10.dir/x10.cpp.obj: libraries/x10/CMakeFiles/x10.dir/flags.make
libraries/x10/CMakeFiles/x10.dir/x10.cpp.obj: ../libraries/x10/x10.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/x10/CMakeFiles/x10.dir/x10.cpp.obj"
	cd /teensyduino/build/libraries/x10 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/x10.dir/x10.cpp.obj -c /teensyduino/libraries/x10/x10.cpp

libraries/x10/CMakeFiles/x10.dir/x10.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/x10.dir/x10.cpp.i"
	cd /teensyduino/build/libraries/x10 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/x10/x10.cpp > CMakeFiles/x10.dir/x10.cpp.i

libraries/x10/CMakeFiles/x10.dir/x10.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/x10.dir/x10.cpp.s"
	cd /teensyduino/build/libraries/x10 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/x10/x10.cpp -o CMakeFiles/x10.dir/x10.cpp.s

# Object files for target x10
x10_OBJECTS = \
"CMakeFiles/x10.dir/x10.cpp.obj"

# External object files for target x10
x10_EXTERNAL_OBJECTS =

libraries/x10/libx10.a: libraries/x10/CMakeFiles/x10.dir/x10.cpp.obj
libraries/x10/libx10.a: libraries/x10/CMakeFiles/x10.dir/build.make
libraries/x10/libx10.a: libraries/x10/CMakeFiles/x10.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libx10.a"
	cd /teensyduino/build/libraries/x10 && $(CMAKE_COMMAND) -P CMakeFiles/x10.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/x10 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/x10.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/x10/CMakeFiles/x10.dir/build: libraries/x10/libx10.a

.PHONY : libraries/x10/CMakeFiles/x10.dir/build

libraries/x10/CMakeFiles/x10.dir/clean:
	cd /teensyduino/build/libraries/x10 && $(CMAKE_COMMAND) -P CMakeFiles/x10.dir/cmake_clean.cmake
.PHONY : libraries/x10/CMakeFiles/x10.dir/clean

libraries/x10/CMakeFiles/x10.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/x10 /teensyduino/build /teensyduino/build/libraries/x10 /teensyduino/build/libraries/x10/CMakeFiles/x10.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/x10/CMakeFiles/x10.dir/depend

