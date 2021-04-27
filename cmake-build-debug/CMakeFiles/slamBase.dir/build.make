# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /opt/clion-2020.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion-2020.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/slamBase.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/slamBase.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slamBase.dir/flags.make

CMakeFiles/slamBase.dir/src/slamBase.cpp.o: CMakeFiles/slamBase.dir/flags.make
CMakeFiles/slamBase.dir/src/slamBase.cpp.o: ../src/slamBase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slamBase.dir/src/slamBase.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slamBase.dir/src/slamBase.cpp.o -c /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/src/slamBase.cpp

CMakeFiles/slamBase.dir/src/slamBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slamBase.dir/src/slamBase.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/src/slamBase.cpp > CMakeFiles/slamBase.dir/src/slamBase.cpp.i

CMakeFiles/slamBase.dir/src/slamBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slamBase.dir/src/slamBase.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/src/slamBase.cpp -o CMakeFiles/slamBase.dir/src/slamBase.cpp.s

# Object files for target slamBase
slamBase_OBJECTS = \
"CMakeFiles/slamBase.dir/src/slamBase.cpp.o"

# External object files for target slamBase
slamBase_EXTERNAL_OBJECTS =

slamBase: CMakeFiles/slamBase.dir/src/slamBase.cpp.o
slamBase: CMakeFiles/slamBase.dir/build.make
slamBase: libRGBD_SLAM.so
slamBase: /usr/local/lib/libopencv_superres.so.3.4.4
slamBase: /usr/local/lib/libopencv_stitching.so.3.4.4
slamBase: /usr/local/lib/libopencv_viz.so.3.4.4
slamBase: /usr/local/lib/libopencv_ml.so.3.4.4
slamBase: /usr/local/lib/libopencv_shape.so.3.4.4
slamBase: /usr/local/lib/libopencv_dnn.so.3.4.4
slamBase: /usr/local/lib/libopencv_videostab.so.3.4.4
slamBase: /usr/local/lib/libopencv_video.so.3.4.4
slamBase: /usr/local/lib/libopencv_photo.so.3.4.4
slamBase: /usr/local/lib/libopencv_objdetect.so.3.4.4
slamBase: /usr/local/lib/libopencv_calib3d.so.3.4.4
slamBase: /usr/local/lib/libopencv_features2d.so.3.4.4
slamBase: /usr/local/lib/libopencv_flann.so.3.4.4
slamBase: /usr/local/lib/libopencv_highgui.so.3.4.4
slamBase: /usr/local/lib/libopencv_videoio.so.3.4.4
slamBase: /usr/local/lib/libopencv_imgcodecs.so.3.4.4
slamBase: /usr/local/lib/libopencv_imgproc.so.3.4.4
slamBase: /usr/local/lib/libopencv_core.so.3.4.4
slamBase: /usr/local/lib/libpcl_io.so
slamBase: /usr/local/lib/libpcl_octree.so
slamBase: /usr/local/lib/libpcl_common.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_system.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
slamBase: /usr/lib/x86_64-linux-gnu/libboost_regex.so
slamBase: /usr/lib/libvtkGenericFiltering.so.5.10.1
slamBase: /usr/lib/libvtkGeovis.so.5.10.1
slamBase: /usr/lib/libvtkCharts.so.5.10.1
slamBase: /usr/lib/libvtkViews.so.5.10.1
slamBase: /usr/lib/libvtkInfovis.so.5.10.1
slamBase: /usr/lib/libvtkWidgets.so.5.10.1
slamBase: /usr/lib/libvtkVolumeRendering.so.5.10.1
slamBase: /usr/lib/libvtkHybrid.so.5.10.1
slamBase: /usr/lib/libvtkParallel.so.5.10.1
slamBase: /usr/lib/libvtkRendering.so.5.10.1
slamBase: /usr/lib/libvtkImaging.so.5.10.1
slamBase: /usr/lib/libvtkGraphics.so.5.10.1
slamBase: /usr/lib/libvtkIO.so.5.10.1
slamBase: /usr/lib/libvtkFiltering.so.5.10.1
slamBase: /usr/lib/libvtkCommon.so.5.10.1
slamBase: /usr/lib/libvtksys.so.5.10.1
slamBase: CMakeFiles/slamBase.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable slamBase"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slamBase.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slamBase.dir/build: slamBase

.PHONY : CMakeFiles/slamBase.dir/build

CMakeFiles/slamBase.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slamBase.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slamBase.dir/clean

CMakeFiles/slamBase.dir/depend:
	cd /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug /media/xtcsun/e58c5409-4d15-4c21-b7a0-623f13653d53/Learn/rgbd_slam/cmake-build-debug/CMakeFiles/slamBase.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slamBase.dir/depend
