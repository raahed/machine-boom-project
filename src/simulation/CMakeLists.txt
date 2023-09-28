#
# Sample CMakeLists.txt for building an application using AGX.
#
project(agxHose)


cmake_minimum_required(VERSION 3.3)

# Check so that the setup_env environment file has been executed:
if ("$ENV{AGX_DIR}"  STREQUAL "")
  message(WARNING "\n*** The environment initialization file setup_env.bat/bash has not been executed\n*** Errors might occur during runtime!")
endif()

set(CMAKE_MAP_IMPORTED_CONFIG_DEBUG Debug Release)

#
# Set this to the path of the installed agx version. For example set(AGX_INSTALL_DIRECTORY "C:/Program Files/Algoryx/AgX 2.10.0.0")
# By default it will look for AGX in the same directory as this CMakeLists.txt
set(AGX_INSTALL_DIRECTORY $ENV{AGX_DIR})

# Get flags that were used when AGX was built.
include(${AGX_INSTALL_DIRECTORY}/AGX_build_settings.txt)


# Make build system 2.4.7 compliant
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

set (AGX_DEBUG_POSTFIX d)
set(CMAKE_DEBUG_POSTFIX ${AGX_DEBUG_POSTFIX} )

# Set the INCLUDE environment variable that will help CMake locate AgX, OSG etc with find_package(AGX)
set(ENV{INCLUDE} "$ENV{INCLUDE};${INCLUDE_PATH}")

file(TO_NATIVE_PATH ${AGX_INSTALL_DIRECTORY}/lib/ LIB_PATH)
set(ENV{LIB} "$ENV{LIB};${LIB_PATH}")

# Need to be set, either as an environment variable or in CMake
# Here we take the current directory (of this file)
set (AGX_DIR ${AGX_INSTALL_DIRECTORY})

#
# Setup CMake and environment
#
set( CMAKE_MODULE_PATH "${AGX_INSTALL_DIRECTORY}/CMakeModules;${CMAKE_MODULE_PATH}")

#
#
#
include(AGXUtils)
include(AGX${CMAKE_SYSTEM_NAME}Specific)
include(AGXCompilerFlags )

# To make some macros compatible
set(AGX_DYNAMIC TRUE)
set(AGX_BINARY_DIR ${agxBuild_BINARY_DIR})

#Libraries for debug has a postfix to separate them from release
set(CMAKE_DEBUG_POSTFIX "d" )

#
# Build Options
#
option(AGX_CMAKE_VERBOSE "Enable to get verbose information from the CMake scripts" ON)


############ Find AGX #####################
# Now locate AGX
find_package(AGX REQUIRED)

if (NOT AGX_FOUND)
  message(FATAL_ERROR "Unable to locate AGX headers and libraries. Cannot proceed.")
endif()


############ find OSG ####################
find_package(OSG REQUIRED)
if (NOT OSG_FOUND)
	message("Could not find OSG!")
else ()
	include_directories (${OSG_INCLUDE_DIR})
	link_directories(${OSG_LIBRARY_DIR})
endif ()

#
# Include and link directories.
#
include_directories (
  ${AGX_INCLUDE_DIR}
)

link_directories(
  ${AGX_LIBRARY_DIR}
)


# We want paths embedded in binaries (e.g tutorials) that are built on OSes that supports
# this feature so that required libraries are found more easily.
# CMake will embed paths by default in the build tree, this block is needed to also
# have paths in the install tree.
#
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${AGX_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(TARGET_NAME hoses_test)

include_directories ( ${AGXOSG_DIR} include/ )

# Add the application
agx_add_executable(${TARGET_NAME}  src/data_collect.cpp  )

# Link to AGX targets
target_link_libraries(${TARGET_NAME} AGX::agxCore AGX::agxPhysics AGX::agxModel AGX::agxCable AGX::agxTerrain AGX::agxHydraulics AGX::agxOSG AGX::agxVehicle)

# Link to OSG targets
target_link_libraries(${TARGET_NAME} OSG::osg OSG::osgUtil OSG::osgDB OSG::osgText OSG::osgViewer OSG::osgShadow OSG::OpenThreads)
