cmake_minimum_required(VERSION 3.20)
project(vram_cpp VERSION 0.1.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VRAM_BUILD_WASM "Build the WebAssembly predictor target" ON)
option(VRAM_BUILD_TESTS "Build local unit tests" ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/CMakeLists.txt")
    message(FATAL_ERROR "vendor/llama-cpp is required for all builds")
endif()

if(EMSCRIPTEN)
    add_compile_options("-fexceptions")
    add_compile_options("-sMEMORY64=1")
    add_link_options("-sNO_DISABLE_EXCEPTION_CATCHING")
    add_link_options("-sMEMORY64=1")
endif()

# Keep llama.cpp available for future linking while avoiding full builds by default.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama common library" FORCE)
add_subdirectory(vendor/llama-cpp EXCLUDE_FROM_ALL)

if(EMSCRIPTEN AND TARGET llama-common)
    # Keep llama/common fit logging on a synchronous path in wasm builds.
    # The default logger path can try to spawn a worker thread, which is not
    # available in the non-pthreads browser build used here.
    target_compile_definitions(llama-common PRIVATE VRAM_EMSCRIPTEN_SYNC_FIT_LOGS=1)
endif()

add_library(vram_predictor_core STATIC
    cpp/src/predictor_api.cpp
    cpp/src/fit_executor.cpp
    cpp/src/sim_backend.cpp
)

target_include_directories(vram_predictor_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/ggml/src
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/vendor/nlohmann/json.hpp")
    target_include_directories(vram_predictor_core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/vendor
    )
endif()

target_link_libraries(vram_predictor_core
    PRIVATE
        llama-common
        llama
        Threads::Threads
)

if(VRAM_BUILD_FIT_HARNESS AND NOT EMSCRIPTEN)
    add_executable(vram_fit_harness
        cpp/tools/vram_fit_harness.cpp
    )

    target_link_libraries(vram_fit_harness
        PRIVATE
            vram_predictor_core
    )
endif()

if(EMSCRIPTEN AND VRAM_BUILD_WASM)
    add_executable(vram_predictor_wasm
        cpp/src/predictor_wasm_main.cpp
    )

    target_link_libraries(vram_predictor_wasm
        PRIVATE
            vram_predictor_core
    )

    # Emit a JS loader + WASM pair for browser/worker usage.
    set_target_properties(vram_predictor_wasm PROPERTIES SUFFIX ".js")

    target_link_options(vram_predictor_wasm PRIVATE
        "-sALLOW_MEMORY_GROWTH=1"
        "-sFETCH=1"
        "-sENVIRONMENT=web,worker"
        "-sMODULARIZE=1"
        "-sEXPORT_NAME=createVRAMPredictorModule"
        "-sEXPORTED_FUNCTIONS=['_malloc','_free','_vram_predictor_predict_json']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS']"
    )
else()
    add_executable(vram_predictor_dev
        cpp/src/predictor_wasm_main.cpp
    )

    target_link_libraries(vram_predictor_dev
        PRIVATE
            vram_predictor_core
    )
endif()

if(VRAM_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(vram_api_tests
            cpp/tests/predictor_api_tests.cpp
        )

        target_link_libraries(vram_api_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_api_tests COMMAND vram_api_tests)

        add_executable(vram_llama_fit_parity_tests
            cpp/tests/llama_fit_parity_tests.cpp
        )

        target_link_libraries(vram_llama_fit_parity_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_llama_fit_parity_tests COMMAND vram_llama_fit_parity_tests)

        if(TARGET vram_fit_harness)
            add_dependencies(vram_llama_fit_parity_tests vram_fit_harness)
        endif()

        add_executable(vram_api_fit_execution_tests
            cpp/tests/predictor_api_fit_execution_tests.cpp
        )

        target_link_libraries(vram_api_fit_execution_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_api_fit_execution_tests COMMAND vram_api_fit_execution_tests)
    endif()
endif()
