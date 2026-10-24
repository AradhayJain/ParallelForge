cmake_minimum_required(VERSION 3.20)
project(ParallelForge
    VERSION 1.0.0
    DESCRIPTION "C++ Parallel Computing & CPU Performance Engineering Lab"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(PARALLELFORGE_ENABLE_TESTS "Build test suite" ON)
option(PARALLELFORGE_ENABLE_BENCHMARKS "Build benchmark suite" ON)
option(PARALLELFORGE_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(PARALLELFORGE_ENABLE_AVX2 "Enable AVX2 vectorization where supported" OFF)
option(PARALLELFORGE_ENABLE_ASAN "Enable AddressSanitizer and LeakSanitizer" OFF)
option(PARALLELFORGE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(PARALLELFORGE_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(PARALLELFORGE_STRICT_WARNINGS "Treat warnings as errors" OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler Warnings Configuration
set(PARALLELFORGE_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wnull-dereference
)

if(PARALLELFORGE_STRICT_WARNINGS)
    list(APPEND PARALLELFORGE_WARNING_FLAGS -Werror)
endif()

# Sanitizer Configuration
set(PARALLELFORGE_SANITIZER_FLAGS "")
if(PARALLELFORGE_ENABLE_TSAN)
    if(PARALLELFORGE_ENABLE_ASAN OR PARALLELFORGE_ENABLE_UBSAN)
        message(FATAL_ERROR "ThreadSanitizer cannot be combined with AddressSanitizer or UBSan.")
    endif()
    message(STATUS "ThreadSanitizer enabled")
    list(APPEND PARALLELFORGE_SANITIZER_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
else()
    if(PARALLELFORGE_ENABLE_ASAN)
        message(STATUS "AddressSanitizer enabled")
        list(APPEND PARALLELFORGE_SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer)
    endif()
    if(PARALLELFORGE_ENABLE_UBSAN)
        message(STATUS "UndefinedBehaviorSanitizer enabled")
        list(APPEND PARALLELFORGE_SANITIZER_FLAGS -fsanitize=undefined -fno-omit-frame-pointer)
    endif()
endif()

# Core Library Sources
set(PARALLELFORGE_LIB_SOURCES
    src/core/matrix.cpp
    src/algorithms/matrix_multiply.cpp
    src/algorithms/convolution.cpp
    src/algorithms/vector_ops.cpp
    src/algorithms/parallel_sort.cpp
    src/threading/bounded_queue.cpp
    src/threading/thread_pool.cpp
    src/threading/parallel_for.cpp
    src/memory/memory_benchmarks.cpp
    src/memory/cache_locality.cpp
    src/memory/false_sharing.cpp
    src/benchmark/benchmark_runner.cpp
    src/benchmark/metric_calculator.cpp
    src/benchmark/exporter.cpp
    src/diagnostics/cpu_info.cpp
    src/diagnostics/system_info.cpp
    src/diagnostics/hardware_features.cpp
    src/cli/command_parser.cpp
    src/cli/commands.cpp
)

add_library(parallelforge_lib STATIC ${PARALLELFORGE_LIB_SOURCES})

target_include_directories(parallelforge_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(parallelforge_lib
    PUBLIC
        ${PARALLELFORGE_WARNING_FLAGS}
        ${PARALLELFORGE_SANITIZER_FLAGS}
)

target_link_options(parallelforge_lib
    PUBLIC
        ${PARALLELFORGE_SANITIZER_FLAGS}
)

# OpenMP Support
if(PARALLELFORGE_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
        target_link_libraries(parallelforge_lib PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(parallelforge_lib PUBLIC PARALLELFORGE_HAS_OPENMP=1)
    else()
        message(WARNING "OpenMP not found. OpenMP implementations will use standard thread fallbacks.")
        target_compile_definitions(parallelforge_lib PUBLIC PARALLELFORGE_HAS_OPENMP=0)
    endif()
else()
    target_compile_definitions(parallelforge_lib PUBLIC PARALLELFORGE_HAS_OPENMP=0)
endif()

# Threads
find_package(Threads REQUIRED)
target_link_libraries(parallelforge_lib PUBLIC Threads::Threads)

# Architecture & SIMD checks
if(PARALLELFORGE_ENABLE_AVX2)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_definitions(parallelforge_lib PUBLIC PARALLELFORGE_ENABLE_AVX2=1)
        # Apply AVX2 specifically to vectorization compilation units
        set_source_files_properties(
            src/algorithms/matrix_multiply.cpp
            src/algorithms/vector_ops.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
        )
        message(STATUS "AVX2 SIMD enabled for vector and matrix units")
    else()
        message(WARNING "AVX2 compiler flag not supported; continuing with scalar fallback")
    endif()
endif()

# Main CLI Executable
add_executable(parallelforge src/cli/main.cpp)
target_link_libraries(parallelforge PRIVATE parallelforge_lib)
target_compile_options(parallelforge PRIVATE ${PARALLELFORGE_WARNING_FLAGS} ${PARALLELFORGE_SANITIZER_FLAGS})
target_link_options(parallelforge PRIVATE ${PARALLELFORGE_SANITIZER_FLAGS})

# Benchmark Executable
if(PARALLELFORGE_ENABLE_BENCHMARKS)
    set(PARALLELFORGE_BENCH_SOURCES
        benchmarks/benchmark_main.cpp
        benchmarks/workloads/bench_matrix.cpp
        benchmarks/workloads/bench_convolution.cpp
        benchmarks/workloads/bench_vector.cpp
        benchmarks/workloads/bench_memory.cpp
    )
    add_executable(parallelforge_benchmarks ${PARALLELFORGE_BENCH_SOURCES})
    target_link_libraries(parallelforge_benchmarks PRIVATE parallelforge_lib)
    target_compile_options(parallelforge_benchmarks PRIVATE ${PARALLELFORGE_WARNING_FLAGS} ${PARALLELFORGE_SANITIZER_FLAGS})
    target_link_options(parallelforge_benchmarks PRIVATE ${PARALLELFORGE_SANITIZER_FLAGS})
endif()

# Testing with GoogleTest
if(PARALLELFORGE_ENABLE_TESTS)
    enable_testing()
    include(FetchContent)
    
    # Declare GoogleTest dependency
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    # Prevent installing GoogleTest artifacts with the main project
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    set(PARALLELFORGE_TEST_SOURCES
        tests/unit/test_matrix.cpp
        tests/unit/test_matrix_multiply.cpp
        tests/unit/test_convolution.cpp
        tests/unit/test_vector_ops.cpp
        tests/unit/test_parallel_sort.cpp
        tests/unit/test_diagnostics.cpp
        tests/unit/test_benchmark_metrics.cpp
        tests/concurrency/test_bounded_queue.cpp
        tests/concurrency/test_thread_pool.cpp
        tests/integration/test_cli.cpp
    )
    
    add_executable(parallelforge_tests ${PARALLELFORGE_TEST_SOURCES})
    target_link_libraries(parallelforge_tests PRIVATE parallelforge_lib GTest::gtest_main)
    target_compile_options(parallelforge_tests PRIVATE ${PARALLELFORGE_WARNING_FLAGS} ${PARALLELFORGE_SANITIZER_FLAGS})
    target_link_options(parallelforge_tests PRIVATE ${PARALLELFORGE_SANITIZER_FLAGS})
    
    include(GoogleTest)
    gtest_discover_tests(parallelforge_tests)
endif()

# Installation rules
install(TARGETS parallelforge parallelforge_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
