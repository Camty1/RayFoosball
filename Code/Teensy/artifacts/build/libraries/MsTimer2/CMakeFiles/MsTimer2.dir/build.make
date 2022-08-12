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
include libraries/MsTimer2/CMakeFiles/MsTimer2.dir/depend.make

# Include the progress variables for this target.
include libraries/MsTimer2/CMakeFiles/MsTimer2.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/MsTimer2/CMakeFiles/MsTimer2.dir/flags.make

libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj: libraries/MsTimer2/CMakeFiles/MsTimer2.dir/flags.make
libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj: ../libraries/MsTimer2/MsTimer2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj"
	cd /teensyduino/build/libraries/MsTimer2 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj -c /teensyduino/libraries/MsTimer2/MsTimer2.cpp

libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MsTimer2.dir/MsTimer2.cpp.i"
	cd /teensyduino/build/libraries/MsTimer2 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/MsTimer2/MsTimer2.cpp > CMakeFiles/MsTimer2.dir/MsTimer2.cpp.i

libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MsTimer2.dir/MsTimer2.cpp.s"
	cd /teensyduino/build/libraries/MsTimer2 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/MsTimer2/MsTimer2.cpp -o CMakeFiles/MsTimer2.dir/MsTimer2.cpp.s

# Object files for target MsTimer2
MsTimer2_OBJECTS = \
"CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj"

# External object files for target MsTimer2
MsTimer2_EXTERNAL_OBJECTS =

libraries/MsTimer2/libMsTimer2.a: libraries/MsTimer2/CMakeFiles/MsTimer2.dir/MsTimer2.cpp.obj
libraries/MsTimer2/libMsTimer2.a: libraries/MsTimer2/CMakeFiles/MsTimer2.dir/build.make
libraries/MsTimer2/libMsTimer2.a: libraries/MsTimer2/CMakeFiles/MsTimer2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libMsTimer2.a"
	cd /teensyduino/build/libraries/MsTimer2 && $(CMAKE_COMMAND) -P CMakeFiles/MsTimer2.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/MsTimer2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MsTimer2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/MsTimer2/CMakeFiles/MsTimer2.dir/build: libraries/MsTimer2/libMsTimer2.a

.PHONY : libraries/MsTimer2/CMakeFiles/MsTimer2.dir/build

libraries/MsTimer2/CMakeFiles/MsTimer2.dir/clean:
	cd /teensyduino/build/libraries/MsTimer2 && $(CMAKE_COMMAND) -P CMakeFiles/MsTimer2.dir/cmake_clean.cmake
.PHONY : libraries/MsTimer2/CMakeFiles/MsTimer2.dir/clean

libraries/MsTimer2/CMakeFiles/MsTimer2.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/MsTimer2 /teensyduino/build /teensyduino/build/libraries/MsTimer2 /teensyduino/build/libraries/MsTimer2/CMakeFiles/MsTimer2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/MsTimer2/CMakeFiles/MsTimer2.dir/depend

