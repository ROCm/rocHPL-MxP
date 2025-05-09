# Modifications (c) 2019-2022 Advanced Micro Devices, Inc.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Dependencies

# Git
find_package(Git REQUIRED)

# MPI
set(MPI_HOME ${HPLMXP_MPI_DIR})
find_package(MPI REQUIRED)

# Add some paths
list(APPEND CMAKE_PREFIX_PATH ${ROCSOLVER_PATH} ${ROCBLAS_PATH} ${ROCM_PATH})
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip )

if(HPLMXP_TRACING)
  find_library(ROCTRACER NAMES roctracer64
               PATHS ${ROCM_PATH}/lib
               NO_DEFAULT_PATH)
  find_library(ROCTX NAMES roctx64
               PATHS ${ROCM_PATH}/lib
               NO_DEFAULT_PATH)

  message("-- roctracer:  ${ROCTRACER}")
  message("-- roctx:      ${ROCTX}")

  add_library(roc::roctracer SHARED IMPORTED)
  set_target_properties(roc::roctracer PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCM_PATH}/include"
    INTERFACE_LINK_LIBRARIES "hip::host"
    IMPORTED_LOCATION "${ROCTRACER}"
    IMPORTED_SONAME "libroctracer.so")
  add_library(roc::roctx SHARED IMPORTED)
  set_target_properties(roc::roctx PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ROCM_PATH}/include"
    INTERFACE_LINK_LIBRARIES "hip::host"
    IMPORTED_LOCATION "${ROCTX}"
    IMPORTED_SONAME "libroctx64.so")
endif()

# Find HIP package
find_package(HIP REQUIRED)

# rocblas
find_package(rocblas REQUIRED)

get_target_property(rocblas_LIBRARIES roc::rocblas IMPORTED_LOCATION_RELEASE)

message("-- rocBLAS version:      ${rocblas_VERSION}")
message("-- rocBLAS include dirs: ${rocblas_INCLUDE_DIRS}")
message("-- rocBLAS libraries:    ${rocblas_LIBRARIES}")

get_filename_component(ROCBLAS_LIB_PATH ${rocblas_LIBRARIES} DIRECTORY)

# rocsolver
find_package(rocsolver REQUIRED)

get_target_property(rocsolver_LIBRARIES roc::rocsolver IMPORTED_LOCATION_RELEASE)

message("-- rocSOLVER version:      ${rocsolver_VERSION}")
message("-- rocSOLVER include dirs: ${rocsolver_INCLUDE_DIRS}")
message("-- rocSOLVER libraries:    ${rocsolver_LIBRARIES}")

get_filename_component(ROCSOLVER_LIB_PATH ${rocsolver_LIBRARIES} DIRECTORY)

# ROCm cmake package
find_package(ROCmCMakeBuildTools QUIET CONFIG PATHS ${CMAKE_PREFIX_PATH})
if(NOT ROCM_FOUND)
  set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package(ROCmCMakeBuildTools REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds OPTIONAL)
