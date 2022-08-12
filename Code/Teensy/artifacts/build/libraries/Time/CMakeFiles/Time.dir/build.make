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
include libraries/Time/CMakeFiles/Time.dir/depend.make

# Include the progress variables for this target.
include libraries/Time/CMakeFiles/Time.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/Time/CMakeFiles/Time.dir/flags.make

libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.obj: libraries/Time/CMakeFiles/Time.dir/flags.make
libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.obj: ../libraries/Time/DateStrings.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.obj"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Time.dir/DateStrings.cpp.obj -c /teensyduino/libraries/Time/DateStrings.cpp

libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Time.dir/DateStrings.cpp.i"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Time/DateStrings.cpp > CMakeFiles/Time.dir/DateStrings.cpp.i

libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Time.dir/DateStrings.cpp.s"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Time/DateStrings.cpp -o CMakeFiles/Time.dir/DateStrings.cpp.s

libraries/Time/CMakeFiles/Time.dir/Time.cpp.obj: libraries/Time/CMakeFiles/Time.dir/flags.make
libraries/Time/CMakeFiles/Time.dir/Time.cpp.obj: ../libraries/Time/Time.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/Time/CMakeFiles/Time.dir/Time.cpp.obj"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Time.dir/Time.cpp.obj -c /teensyduino/libraries/Time/Time.cpp

libraries/Time/CMakeFiles/Time.dir/Time.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Time.dir/Time.cpp.i"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Time/Time.cpp > CMakeFiles/Time.dir/Time.cpp.i

libraries/Time/CMakeFiles/Time.dir/Time.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Time.dir/Time.cpp.s"
	cd /teensyduino/build/libraries/Time && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Time/Time.cpp -o CMakeFiles/Time.dir/Time.cpp.s

# Object files for target Time
Time_OBJECTS = \
"CMakeFiles/Time.dir/DateStrings.cpp.obj" \
"CMakeFiles/Time.dir/Time.cpp.obj"

# External object files for target Time
Time_EXTERNAL_OBJECTS =

libraries/Time/libTime.a: libraries/Time/CMakeFiles/Time.dir/DateStrings.cpp.obj
libraries/Time/libTime.a: libraries/Time/CMakeFiles/Time.dir/Time.cpp.obj
libraries/Time/libTime.a: libraries/Time/CMakeFiles/Time.dir/build.make
libraries/Time/libTime.a: libraries/Time/CMakeFiles/Time.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libTime.a"
	cd /teensyduino/build/libraries/Time && $(CMAKE_COMMAND) -P CMakeFiles/Time.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/Time && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Time.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/Time/CMakeFiles/Time.dir/build: libraries/Time/libTime.a

.PHONY : libraries/Time/CMakeFiles/Time.dir/build

libraries/Time/CMakeFiles/Time.dir/clean:
	cd /teensyduino/build/libraries/Time && $(CMAKE_COMMAND) -P CMakeFiles/Time.dir/cmake_clean.cmake
.PHONY : libraries/Time/CMakeFiles/Time.dir/clean

libraries/Time/CMakeFiles/Time.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/Time /teensyduino/build /teensyduino/build/libraries/Time /teensyduino/build/libraries/Time/CMakeFiles/Time.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/Time/CMakeFiles/Time.dir/depend
