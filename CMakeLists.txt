cmake_minimum_required(VERSION 2.6)
project(seeded-region-growing)
include_directories(include)
file(GLOB headers include/andres/*.hxx include/andres/vision/*.hxx)

##############################################################################
# MATLAB
##############################################################################
find_package(Matlab QUIET)
if(MATLAB_FOUND)
    INCLUDE_DIRECTORIES(${Matlab_INCLUDE_DIRS})
    message(STATUS "MATLAB found")
else()
    message("MATLAB not found")
endif()

##############################################################################
# targets: vision, mex: seeded region growing
##############################################################################
if(MATLAB_FOUND)
    add_library(andres_seeded_region_growing SHARED src/mex/andres_seeded_region_growing.cxx src/mex/andres_seeded_region_growing.def ${headers})
    target_link_libraries(andres_seeded_region_growing ${MATLAB_LIBRARIES})
endif()
