cmake_minimum_required(VERSION 3.23)

project(bwarp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake's MSVC default adds /EHsc; strip it so our own /EHs-c- (no exceptions --
# bwarp never uses C++ exceptions) isn't overridden on the command line.
if(MSVC)
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

option(BWARP_CUDA "Build the GPU variant if a CUDA compiler is found" ON)

# The GPU variant is optional: without a CUDA compiler bwarp builds the CPU parts only, and
# gpubwarp_stub.cpp reports every GPU call as unavailable. With the ClangCL toolset the CUDA 13.5
# Visual Studio integration fails CMake's compiler check, so the clang preset builds the CPU parts
# only; the cuda preset (Ninja, clang-cl + nvcc) and the msvc preset (cl.exe + nvcc) build both.
if(BWARP_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "bwarp: no CUDA compiler found, building without the GPU variant")
    endif()
endif()

# =============================================================================
# Source groups -- organized by concept (see README.md, "Where things live").
# =============================================================================

# Shared by all parts: the BWT checker from sadcs (checker) and file I/O (fileio).
set(COMMON_SOURCES
    src/common/checker.h
    src/common/checker.cpp
    src/common/fileio.h
    src/common/fileio.cpp
)

# The bwarp.exe command line: commands (main) and the built-in test suite (selftest).
set(CLI_SOURCES
    src/cli/main.cpp
    src/cli/selftest.h
    src/cli/selftest.cpp
)

set(BWARP_SOURCES
    ${COMMON_SOURCES}
    ${CLI_SOURCES}
    src/cuda/gpubwarp.h
)

if(NOT CMAKE_CUDA_COMPILER)
    list(APPEND BWARP_SOURCES src/cuda/gpubwarp_stub.cpp)
endif()

add_executable(${PROJECT_NAME} ${BWARP_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE src/common src/cuda src/cli)
source_group(TREE "${CMAKE_SOURCE_DIR}/src" FILES ${BWARP_SOURCES})

# The GPU variant: device code compiled by nvcc, with MSVC's cl.exe as host compiler on
# Windows. sm_120 (Blackwell) plus PTX for newer GPUs; CUB ships with the toolkit.
# The .cuh files are sadcs's GPU BWT construction (BWT path only), all in gpubwarp.cu's translation unit.
if(CMAKE_CUDA_COMPILER)
    add_library(bwarp_cuda STATIC src/cuda/gpubwarp.h src/cuda/gpubwarp.cu src/cuda/differencecover.h
        src/cuda/devicecommon.cuh src/cuda/samplerank.cuh src/cuda/classsort.cuh src/cuda/merge.cuh)
    set_target_properties(bwarp_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "120-real;120-virtual"
    )
    target_include_directories(bwarp_cuda PRIVATE src/common)
    if(MSVC)
        # CCCL 3 refuses cl.exe's traditional preprocessor.
        target_compile_options(bwarp_cuda PRIVATE -Xcompiler=/Zc:preprocessor,/utf-8)
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE bwarp_cuda)
endif()

if(MSVC)
    # Link-time code generation. The Visual Studio generator turns /GL into its
    # WholeProgramOptimization property, for clang-cl too; elsewhere clang-cl ignores /GL.
    if(CMAKE_GENERATOR MATCHES "Visual Studio" OR NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(BWARP_LTCG ON)
    endif()

    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
        /EHs-c-
        /utf-8
        $<$<CONFIG:Release>:/O2>
        $<$<AND:$<CONFIG:Release>,$<BOOL:${BWARP_LTCG}>>:/GL>
    )
    # _HAS_EXCEPTIONS=0: the MSVC STL's own switch matching /EHs-c- (otherwise C4530).
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _HAS_EXCEPTIONS=0
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Release>,$<BOOL:${BWARP_LTCG}>>:/LTCG>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -fno-exceptions $<$<CONFIG:Release>:-O2>)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
install(FILES LICENSE THIRD-PARTY-NOTICES.txt DESTINATION .)
