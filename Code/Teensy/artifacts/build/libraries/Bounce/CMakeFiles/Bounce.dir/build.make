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
include libraries/Bounce/CMakeFiles/Bounce.dir/depend.make

# Include the progress variables for this target.
include libraries/Bounce/CMakeFiles/Bounce.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/Bounce/CMakeFiles/Bounce.dir/flags.make

libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.obj: libraries/Bounce/CMakeFiles/Bounce.dir/flags.make
libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.obj: ../libraries/Bounce/Bounce.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.obj"
	cd /teensyduino/build/libraries/Bounce && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Bounce.dir/Bounce.cpp.obj -c /teensyduino/libraries/Bounce/Bounce.cpp

libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Bounce.dir/Bounce.cpp.i"
	cd /teensyduino/build/libraries/Bounce && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Bounce/Bounce.cpp > CMakeFiles/Bounce.dir/Bounce.cpp.i

libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Bounce.dir/Bounce.cpp.s"
	cd /teensyduino/build/libraries/Bounce && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Bounce/Bounce.cpp -o CMakeFiles/Bounce.dir/Bounce.cpp.s

# Object files for target Bounce
Bounce_OBJECTS = \
"CMakeFiles/Bounce.dir/Bounce.cpp.obj"

# External object files for target Bounce
Bounce_EXTERNAL_OBJECTS =

libraries/Bounce/libBounce.a: libraries/Bounce/CMakeFiles/Bounce.dir/Bounce.cpp.obj
libraries/Bounce/libBounce.a: libraries/Bounce/CMakeFiles/Bounce.dir/build.make
libraries/Bounce/libBounce.a: libraries/Bounce/CMakeFiles/Bounce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libBounce.a"
	cd /teensyduino/build/libraries/Bounce && $(CMAKE_COMMAND) -P CMakeFiles/Bounce.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/Bounce && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Bounce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/Bounce/CMakeFiles/Bounce.dir/build: libraries/Bounce/libBounce.a

.PHONY : libraries/Bounce/CMakeFiles/Bounce.dir/build

libraries/Bounce/CMakeFiles/Bounce.dir/clean:
	cd /teensyduino/build/libraries/Bounce && $(CMAKE_COMMAND) -P CMakeFiles/Bounce.dir/cmake_clean.cmake
.PHONY : libraries/Bounce/CMakeFiles/Bounce.dir/clean

libraries/Bounce/CMakeFiles/Bounce.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/Bounce /teensyduino/build /teensyduino/build/libraries/Bounce /teensyduino/build/libraries/Bounce/CMakeFiles/Bounce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/Bounce/CMakeFiles/Bounce.dir/depend
