# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

# Configure the clSPARSE package to be used in another cmake project.
#
# Defines the following variables:
#
#  clSPARSE_INCLUDE_DIRS - include directories for clSPARSE
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
# Accepts the following variables as input:
#
#-----------------------
# Example Usage:
#
# find_package( clSPARSE REQUIRED CONFIG
#     HINTS <CLSPARSE_ROOT>/package )
#
#    add_executable( foo foo.cc )

#    # uses imported targets from package, including setting header paths
#    target_link_libraries( foo clSPARSE )
#
#-----------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was clSPARSEConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set_and_check( clSPARSE_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include" )
set_and_check( clSPARSE_INCLUDE_DIRS "${clSPARSE_INCLUDE_DIR}" )
set_and_check( clSPARSE_LIB_INSTALL_DIR "${PACKAGE_PREFIX_DIR}/lib64" )

include( "${CMAKE_CURRENT_LIST_DIR}/clSPARSE-Targets.cmake" )
