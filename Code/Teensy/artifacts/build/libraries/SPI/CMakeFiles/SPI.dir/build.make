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
include libraries/SPI/CMakeFiles/SPI.dir/depend.make

# Include the progress variables for this target.
include libraries/SPI/CMakeFiles/SPI.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/SPI/CMakeFiles/SPI.dir/flags.make

libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.obj: libraries/SPI/CMakeFiles/SPI.dir/flags.make
libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.obj: ../libraries/SPI/SPI.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.obj"
	cd /teensyduino/build/libraries/SPI && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SPI.dir/SPI.cpp.obj -c /teensyduino/libraries/SPI/SPI.cpp

libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SPI.dir/SPI.cpp.i"
	cd /teensyduino/build/libraries/SPI && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/SPI/SPI.cpp > CMakeFiles/SPI.dir/SPI.cpp.i

libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SPI.dir/SPI.cpp.s"
	cd /teensyduino/build/libraries/SPI && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/SPI/SPI.cpp -o CMakeFiles/SPI.dir/SPI.cpp.s

# Object files for target SPI
SPI_OBJECTS = \
"CMakeFiles/SPI.dir/SPI.cpp.obj"

# External object files for target SPI
SPI_EXTERNAL_OBJECTS =

libraries/SPI/libSPI.a: libraries/SPI/CMakeFiles/SPI.dir/SPI.cpp.obj
libraries/SPI/libSPI.a: libraries/SPI/CMakeFiles/SPI.dir/build.make
libraries/SPI/libSPI.a: libraries/SPI/CMakeFiles/SPI.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libSPI.a"
	cd /teensyduino/build/libraries/SPI && $(CMAKE_COMMAND) -P CMakeFiles/SPI.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/SPI && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SPI.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/SPI/CMakeFiles/SPI.dir/build: libraries/SPI/libSPI.a

.PHONY : libraries/SPI/CMakeFiles/SPI.dir/build

libraries/SPI/CMakeFiles/SPI.dir/clean:
	cd /teensyduino/build/libraries/SPI && $(CMAKE_COMMAND) -P CMakeFiles/SPI.dir/cmake_clean.cmake
.PHONY : libraries/SPI/CMakeFiles/SPI.dir/clean

libraries/SPI/CMakeFiles/SPI.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/SPI /teensyduino/build /teensyduino/build/libraries/SPI /teensyduino/build/libraries/SPI/CMakeFiles/SPI.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/SPI/CMakeFiles/SPI.dir/depend
