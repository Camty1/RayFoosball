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
include libraries/ssd1351/CMakeFiles/ssd1351.dir/depend.make

# Include the progress variables for this target.
include libraries/ssd1351/CMakeFiles/ssd1351.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/ssd1351/CMakeFiles/ssd1351.dir/flags.make

libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.obj: libraries/ssd1351/CMakeFiles/ssd1351.dir/flags.make
libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.obj: ../libraries/ssd1351/glcdfont.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.obj"
	cd /teensyduino/build/libraries/ssd1351 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ssd1351.dir/glcdfont.c.obj   -c /teensyduino/libraries/ssd1351/glcdfont.c

libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ssd1351.dir/glcdfont.c.i"
	cd /teensyduino/build/libraries/ssd1351 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /teensyduino/libraries/ssd1351/glcdfont.c > CMakeFiles/ssd1351.dir/glcdfont.c.i

libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ssd1351.dir/glcdfont.c.s"
	cd /teensyduino/build/libraries/ssd1351 && /teensyduino/bin/arm-none-eabi-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /teensyduino/libraries/ssd1351/glcdfont.c -o CMakeFiles/ssd1351.dir/glcdfont.c.s

# Object files for target ssd1351
ssd1351_OBJECTS = \
"CMakeFiles/ssd1351.dir/glcdfont.c.obj"

# External object files for target ssd1351
ssd1351_EXTERNAL_OBJECTS =

libraries/ssd1351/libssd1351.a: libraries/ssd1351/CMakeFiles/ssd1351.dir/glcdfont.c.obj
libraries/ssd1351/libssd1351.a: libraries/ssd1351/CMakeFiles/ssd1351.dir/build.make
libraries/ssd1351/libssd1351.a: libraries/ssd1351/CMakeFiles/ssd1351.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libssd1351.a"
	cd /teensyduino/build/libraries/ssd1351 && $(CMAKE_COMMAND) -P CMakeFiles/ssd1351.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/ssd1351 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ssd1351.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/ssd1351/CMakeFiles/ssd1351.dir/build: libraries/ssd1351/libssd1351.a

.PHONY : libraries/ssd1351/CMakeFiles/ssd1351.dir/build

libraries/ssd1351/CMakeFiles/ssd1351.dir/clean:
	cd /teensyduino/build/libraries/ssd1351 && $(CMAKE_COMMAND) -P CMakeFiles/ssd1351.dir/cmake_clean.cmake
.PHONY : libraries/ssd1351/CMakeFiles/ssd1351.dir/clean

libraries/ssd1351/CMakeFiles/ssd1351.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/ssd1351 /teensyduino/build /teensyduino/build/libraries/ssd1351 /teensyduino/build/libraries/ssd1351/CMakeFiles/ssd1351.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/ssd1351/CMakeFiles/ssd1351.dir/depend

