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
include libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/depend.make

# Include the progress variables for this target.
include libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/flags.make

libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj: libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/flags.make
libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj: ../libraries/XPT2046_Touchscreen/XPT2046_Touchscreen.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj"
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj -c /teensyduino/libraries/XPT2046_Touchscreen/XPT2046_Touchscreen.cpp

libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.i"
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/XPT2046_Touchscreen/XPT2046_Touchscreen.cpp > CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.i

libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.s"
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/XPT2046_Touchscreen/XPT2046_Touchscreen.cpp -o CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.s

# Object files for target XPT2046_Touchscreen
XPT2046_Touchscreen_OBJECTS = \
"CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj"

# External object files for target XPT2046_Touchscreen
XPT2046_Touchscreen_EXTERNAL_OBJECTS =

libraries/XPT2046_Touchscreen/libXPT2046_Touchscreen.a: libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/XPT2046_Touchscreen.cpp.obj
libraries/XPT2046_Touchscreen/libXPT2046_Touchscreen.a: libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/build.make
libraries/XPT2046_Touchscreen/libXPT2046_Touchscreen.a: libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libXPT2046_Touchscreen.a"
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && $(CMAKE_COMMAND) -P CMakeFiles/XPT2046_Touchscreen.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/XPT2046_Touchscreen.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/build: libraries/XPT2046_Touchscreen/libXPT2046_Touchscreen.a

.PHONY : libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/build

libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/clean:
	cd /teensyduino/build/libraries/XPT2046_Touchscreen && $(CMAKE_COMMAND) -P CMakeFiles/XPT2046_Touchscreen.dir/cmake_clean.cmake
.PHONY : libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/clean

libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/XPT2046_Touchscreen /teensyduino/build /teensyduino/build/libraries/XPT2046_Touchscreen /teensyduino/build/libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/XPT2046_Touchscreen/CMakeFiles/XPT2046_Touchscreen.dir/depend

