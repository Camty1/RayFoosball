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
include libraries/SerialFlash/CMakeFiles/SerialFlash.dir/depend.make

# Include the progress variables for this target.
include libraries/SerialFlash/CMakeFiles/SerialFlash.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/SerialFlash/CMakeFiles/SerialFlash.dir/flags.make

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/flags.make
libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj: ../libraries/SerialFlash/SerialFlashChip.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj -c /teensyduino/libraries/SerialFlash/SerialFlashChip.cpp

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.i"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/SerialFlash/SerialFlashChip.cpp > CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.i

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.s"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/SerialFlash/SerialFlashChip.cpp -o CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.s

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/flags.make
libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj: ../libraries/SerialFlash/SerialFlashDirectory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj -c /teensyduino/libraries/SerialFlash/SerialFlashDirectory.cpp

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.i"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/SerialFlash/SerialFlashDirectory.cpp > CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.i

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.s"
	cd /teensyduino/build/libraries/SerialFlash && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/SerialFlash/SerialFlashDirectory.cpp -o CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.s

# Object files for target SerialFlash
SerialFlash_OBJECTS = \
"CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj" \
"CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj"

# External object files for target SerialFlash
SerialFlash_EXTERNAL_OBJECTS =

libraries/SerialFlash/libSerialFlash.a: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashChip.cpp.obj
libraries/SerialFlash/libSerialFlash.a: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/SerialFlashDirectory.cpp.obj
libraries/SerialFlash/libSerialFlash.a: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/build.make
libraries/SerialFlash/libSerialFlash.a: libraries/SerialFlash/CMakeFiles/SerialFlash.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libSerialFlash.a"
	cd /teensyduino/build/libraries/SerialFlash && $(CMAKE_COMMAND) -P CMakeFiles/SerialFlash.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/SerialFlash && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SerialFlash.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/SerialFlash/CMakeFiles/SerialFlash.dir/build: libraries/SerialFlash/libSerialFlash.a

.PHONY : libraries/SerialFlash/CMakeFiles/SerialFlash.dir/build

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/clean:
	cd /teensyduino/build/libraries/SerialFlash && $(CMAKE_COMMAND) -P CMakeFiles/SerialFlash.dir/cmake_clean.cmake
.PHONY : libraries/SerialFlash/CMakeFiles/SerialFlash.dir/clean

libraries/SerialFlash/CMakeFiles/SerialFlash.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/SerialFlash /teensyduino/build /teensyduino/build/libraries/SerialFlash /teensyduino/build/libraries/SerialFlash/CMakeFiles/SerialFlash.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/SerialFlash/CMakeFiles/SerialFlash.dir/depend

