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
include libraries/MFRC522/CMakeFiles/MFRC522.dir/depend.make

# Include the progress variables for this target.
include libraries/MFRC522/CMakeFiles/MFRC522.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/MFRC522/CMakeFiles/MFRC522.dir/flags.make

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj: libraries/MFRC522/CMakeFiles/MFRC522.dir/flags.make
libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj: ../libraries/MFRC522/src/MFRC522.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj -c /teensyduino/libraries/MFRC522/src/MFRC522.cpp

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MFRC522.dir/src/MFRC522.cpp.i"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/MFRC522/src/MFRC522.cpp > CMakeFiles/MFRC522.dir/src/MFRC522.cpp.i

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MFRC522.dir/src/MFRC522.cpp.s"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/MFRC522/src/MFRC522.cpp -o CMakeFiles/MFRC522.dir/src/MFRC522.cpp.s

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj: libraries/MFRC522/CMakeFiles/MFRC522.dir/flags.make
libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj: ../libraries/MFRC522/src/MFRC522Debug.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj -c /teensyduino/libraries/MFRC522/src/MFRC522Debug.cpp

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.i"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/MFRC522/src/MFRC522Debug.cpp > CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.i

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.s"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/MFRC522/src/MFRC522Debug.cpp -o CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.s

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj: libraries/MFRC522/CMakeFiles/MFRC522.dir/flags.make
libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj: ../libraries/MFRC522/src/MFRC522Extended.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj -c /teensyduino/libraries/MFRC522/src/MFRC522Extended.cpp

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.i"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/MFRC522/src/MFRC522Extended.cpp > CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.i

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.s"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/MFRC522/src/MFRC522Extended.cpp -o CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.s

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj: libraries/MFRC522/CMakeFiles/MFRC522.dir/flags.make
libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj: ../libraries/MFRC522/src/MFRC522Hack.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj -c /teensyduino/libraries/MFRC522/src/MFRC522Hack.cpp

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.i"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/MFRC522/src/MFRC522Hack.cpp > CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.i

libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.s"
	cd /teensyduino/build/libraries/MFRC522 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/MFRC522/src/MFRC522Hack.cpp -o CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.s

# Object files for target MFRC522
MFRC522_OBJECTS = \
"CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj" \
"CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj" \
"CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj" \
"CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj"

# External object files for target MFRC522
MFRC522_EXTERNAL_OBJECTS =

libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522.cpp.obj
libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Debug.cpp.obj
libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Extended.cpp.obj
libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/src/MFRC522Hack.cpp.obj
libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/build.make
libraries/MFRC522/libMFRC522.a: libraries/MFRC522/CMakeFiles/MFRC522.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libMFRC522.a"
	cd /teensyduino/build/libraries/MFRC522 && $(CMAKE_COMMAND) -P CMakeFiles/MFRC522.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/MFRC522 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MFRC522.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/MFRC522/CMakeFiles/MFRC522.dir/build: libraries/MFRC522/libMFRC522.a

.PHONY : libraries/MFRC522/CMakeFiles/MFRC522.dir/build

libraries/MFRC522/CMakeFiles/MFRC522.dir/clean:
	cd /teensyduino/build/libraries/MFRC522 && $(CMAKE_COMMAND) -P CMakeFiles/MFRC522.dir/cmake_clean.cmake
.PHONY : libraries/MFRC522/CMakeFiles/MFRC522.dir/clean

libraries/MFRC522/CMakeFiles/MFRC522.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/MFRC522 /teensyduino/build /teensyduino/build/libraries/MFRC522 /teensyduino/build/libraries/MFRC522/CMakeFiles/MFRC522.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/MFRC522/CMakeFiles/MFRC522.dir/depend

