CALL %~dp0\build_docker_container.bat

if not [%1]==[] goto program
echo
echo "Must use one of 30,31,32,35,35,40,41,LC as argument, exiting"
goto exit


:program
cd %~dp0..

CALL docker run -t -i ^
     -v %CD%\src:/src ^
     -v %CD%\libs:/libs ^
     -v %CD%\artifacts\build:/teensyduino/build ^
     -v %CD%\artifacts\install:/teensyduino/install ^
     --env TEENSY_VERSION=%1 ^
     --env PROGRAM_ON_BUILD=0 ^
     teensy_dev:latest

:exit