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
include libraries/RA8875/CMakeFiles/RA8875.dir/depend.make

# Include the progress variables for this target.
include libraries/RA8875/CMakeFiles/RA8875.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/RA8875/CMakeFiles/RA8875.dir/flags.make

libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.obj: libraries/RA8875/CMakeFiles/RA8875.dir/flags.make
libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.obj: ../libraries/RA8875/RA8875.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.obj"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RA8875.dir/RA8875.cpp.obj -c /teensyduino/libraries/RA8875/RA8875.cpp

libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RA8875.dir/RA8875.cpp.i"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/RA8875/RA8875.cpp > CMakeFiles/RA8875.dir/RA8875.cpp.i

libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RA8875.dir/RA8875.cpp.s"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/RA8875/RA8875.cpp -o CMakeFiles/RA8875.dir/RA8875.cpp.s

libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.obj: libraries/RA8875/CMakeFiles/RA8875.dir/flags.make
libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.obj: ../libraries/RA8875/glcdfont.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.obj"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/RA8875.dir/glcdfont.c.obj   -c /teensyduino/libraries/RA8875/glcdfont.c

libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/RA8875.dir/glcdfont.c.i"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /teensyduino/libraries/RA8875/glcdfont.c > CMakeFiles/RA8875.dir/glcdfont.c.i

libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/RA8875.dir/glcdfont.c.s"
	cd /teensyduino/build/libraries/RA8875 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /teensyduino/libraries/RA8875/glcdfont.c -o CMakeFiles/RA8875.dir/glcdfont.c.s

# Object files for target RA8875
RA8875_OBJECTS = \
"CMakeFiles/RA8875.dir/RA8875.cpp.obj" \
"CMakeFiles/RA8875.dir/glcdfont.c.obj"

# External object files for target RA8875
RA8875_EXTERNAL_OBJECTS =

libraries/RA8875/libRA8875.a: libraries/RA8875/CMakeFiles/RA8875.dir/RA8875.cpp.obj
libraries/RA8875/libRA8875.a: libraries/RA8875/CMakeFiles/RA8875.dir/glcdfont.c.obj
libraries/RA8875/libRA8875.a: libraries/RA8875/CMakeFiles/RA8875.dir/build.make
libraries/RA8875/libRA8875.a: libraries/RA8875/CMakeFiles/RA8875.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libRA8875.a"
	cd /teensyduino/build/libraries/RA8875 && $(CMAKE_COMMAND) -P CMakeFiles/RA8875.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/RA8875 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RA8875.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/RA8875/CMakeFiles/RA8875.dir/build: libraries/RA8875/libRA8875.a

.PHONY : libraries/RA8875/CMakeFiles/RA8875.dir/build

libraries/RA8875/CMakeFiles/RA8875.dir/clean:
	cd /teensyduino/build/libraries/RA8875 && $(CMAKE_COMMAND) -P CMakeFiles/RA8875.dir/cmake_clean.cmake
.PHONY : libraries/RA8875/CMakeFiles/RA8875.dir/clean

libraries/RA8875/CMakeFiles/RA8875.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/RA8875 /teensyduino/build /teensyduino/build/libraries/RA8875 /teensyduino/build/libraries/RA8875/CMakeFiles/RA8875.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/RA8875/CMakeFiles/RA8875.dir/depend
