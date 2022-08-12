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
include libraries/Ethernet/CMakeFiles/Ethernet.dir/depend.make

# Include the progress variables for this target.
include libraries/Ethernet/CMakeFiles/Ethernet.dir/progress.make

# Include the compile flags for this target's objects.
include libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj: ../libraries/Ethernet/src/Dhcp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj -c /teensyduino/libraries/Ethernet/src/Dhcp.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/Dhcp.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/Dhcp.cpp > CMakeFiles/Ethernet.dir/src/Dhcp.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/Dhcp.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/Dhcp.cpp -o CMakeFiles/Ethernet.dir/src/Dhcp.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.obj: ../libraries/Ethernet/src/Dns.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/Dns.cpp.obj -c /teensyduino/libraries/Ethernet/src/Dns.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/Dns.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/Dns.cpp > CMakeFiles/Ethernet.dir/src/Dns.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/Dns.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/Dns.cpp -o CMakeFiles/Ethernet.dir/src/Dns.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj: ../libraries/Ethernet/src/Ethernet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj -c /teensyduino/libraries/Ethernet/src/Ethernet.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/Ethernet.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/Ethernet.cpp > CMakeFiles/Ethernet.dir/src/Ethernet.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/Ethernet.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/Ethernet.cpp -o CMakeFiles/Ethernet.dir/src/Ethernet.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj: ../libraries/Ethernet/src/EthernetClient.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj -c /teensyduino/libraries/Ethernet/src/EthernetClient.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/EthernetClient.cpp > CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/EthernetClient.cpp -o CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj: ../libraries/Ethernet/src/EthernetServer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj -c /teensyduino/libraries/Ethernet/src/EthernetServer.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/EthernetServer.cpp > CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/EthernetServer.cpp -o CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj: ../libraries/Ethernet/src/EthernetUdp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj -c /teensyduino/libraries/Ethernet/src/EthernetUdp.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/EthernetUdp.cpp > CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/EthernetUdp.cpp -o CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.obj: ../libraries/Ethernet/src/socket.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/socket.cpp.obj -c /teensyduino/libraries/Ethernet/src/socket.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/socket.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/socket.cpp > CMakeFiles/Ethernet.dir/src/socket.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/socket.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/socket.cpp -o CMakeFiles/Ethernet.dir/src/socket.cpp.s

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj: libraries/Ethernet/CMakeFiles/Ethernet.dir/flags.make
libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj: ../libraries/Ethernet/src/utility/w5100.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj -c /teensyduino/libraries/Ethernet/src/utility/w5100.cpp

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.i"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /teensyduino/libraries/Ethernet/src/utility/w5100.cpp > CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.i

libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.s"
	cd /teensyduino/build/libraries/Ethernet && /teensyduino/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /teensyduino/libraries/Ethernet/src/utility/w5100.cpp -o CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.s

# Object files for target Ethernet
Ethernet_OBJECTS = \
"CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/Dns.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/socket.cpp.obj" \
"CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj"

# External object files for target Ethernet
Ethernet_EXTERNAL_OBJECTS =

libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dhcp.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Dns.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/Ethernet.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetClient.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetServer.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/EthernetUdp.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/socket.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/src/utility/w5100.cpp.obj
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/build.make
libraries/Ethernet/libEthernet.a: libraries/Ethernet/CMakeFiles/Ethernet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/teensyduino/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library libEthernet.a"
	cd /teensyduino/build/libraries/Ethernet && $(CMAKE_COMMAND) -P CMakeFiles/Ethernet.dir/cmake_clean_target.cmake
	cd /teensyduino/build/libraries/Ethernet && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Ethernet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
libraries/Ethernet/CMakeFiles/Ethernet.dir/build: libraries/Ethernet/libEthernet.a

.PHONY : libraries/Ethernet/CMakeFiles/Ethernet.dir/build

libraries/Ethernet/CMakeFiles/Ethernet.dir/clean:
	cd /teensyduino/build/libraries/Ethernet && $(CMAKE_COMMAND) -P CMakeFiles/Ethernet.dir/cmake_clean.cmake
.PHONY : libraries/Ethernet/CMakeFiles/Ethernet.dir/clean

libraries/Ethernet/CMakeFiles/Ethernet.dir/depend:
	cd /teensyduino/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /teensyduino /teensyduino/libraries/Ethernet /teensyduino/build /teensyduino/build/libraries/Ethernet /teensyduino/build/libraries/Ethernet/CMakeFiles/Ethernet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : libraries/Ethernet/CMakeFiles/Ethernet.dir/depend

