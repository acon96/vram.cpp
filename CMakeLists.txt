cmake_minimum_required(VERSION 3.20)
project(vram_cpp VERSION 0.1.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VRAM_BUILD_WASM "Build the WebAssembly predictor target" ON)
option(VRAM_ENABLE_VENDOR_LLAMA "Configure vendor/llama-cpp as an external subproject" OFF)
option(VRAM_BUILD_TESTS "Build local unit tests" ON)
option(VRAM_BUILD_FIT_HARNESS "Build native in-process llama-fit harness (requires vendor llama enabled)" ON)

if(EMSCRIPTEN AND VRAM_ENABLE_VENDOR_LLAMA)
    add_compile_options("-fexceptions")
    add_compile_options("-sMEMORY64=1")
    add_link_options("-sNO_DISABLE_EXCEPTION_CATCHING")
    add_link_options("-sMEMORY64=1")
endif()

if(VRAM_ENABLE_VENDOR_LLAMA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/CMakeLists.txt")
    # Keep llama.cpp available for future linking while avoiding full builds by default.
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama common library" FORCE)
    add_subdirectory(vendor/llama-cpp EXCLUDE_FROM_ALL)
endif()

if(VRAM_ENABLE_VENDOR_LLAMA AND VRAM_BUILD_FIT_HARNESS AND NOT EMSCRIPTEN)
    add_executable(vram_fit_harness
        tools/vram_fit_harness.cpp
    )

    target_include_directories(vram_fit_harness
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/common
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/ggml/include
    )

    target_link_libraries(vram_fit_harness
        PRIVATE
            llama-common
            llama
            Threads::Threads
    )
endif()

add_library(vram_predictor_core STATIC
    src/predictor_api.cpp
    src/fit_executor.cpp
    src/gguf_prefix_parser.cpp
    src/hf_range_plan.cpp
    src/hf_range_fetch_helper.cpp
)

target_include_directories(vram_predictor_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/vendor/nlohmann/json.hpp")
    target_include_directories(vram_predictor_core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/vendor
    )
endif()

if(VRAM_ENABLE_VENDOR_LLAMA)
    target_compile_definitions(vram_predictor_core
        PRIVATE
            VRAM_HAS_LLAMA_FIT_EXECUTION=1
    )

    target_include_directories(vram_predictor_core
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/common
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama-cpp/ggml/include
    )

    target_link_libraries(vram_predictor_core
        PRIVATE
            llama-common
            llama
            Threads::Threads
    )
endif()

if(EMSCRIPTEN AND VRAM_BUILD_WASM)
    add_executable(vram_predictor_wasm
        src/predictor_wasm_main.cpp
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
        "-sEXPORTED_FUNCTIONS=['_malloc','_free','_vram_predictor_get_system_info_json','_vram_predictor_predict_json']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS']"
    )
else()
    add_executable(vram_predictor_dev
        src/predictor_wasm_main.cpp
    )

    target_link_libraries(vram_predictor_dev
        PRIVATE
            vram_predictor_core
    )
endif()

if(VRAM_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_executable(vram_parser_tests
            tests/parser_tests.cpp
        )

        target_link_libraries(vram_parser_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_parser_tests COMMAND vram_parser_tests)

        add_executable(vram_api_tests
            tests/predictor_api_tests.cpp
        )

        target_link_libraries(vram_api_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_api_tests COMMAND vram_api_tests)

        add_executable(vram_hf_helper_tests
            tests/hf_range_fetch_helper_tests.cpp
        )

        target_link_libraries(vram_hf_helper_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_hf_helper_tests COMMAND vram_hf_helper_tests)

        add_executable(vram_llama_fit_parity_tests
            tests/llama_fit_parity_tests.cpp
        )

        target_link_libraries(vram_llama_fit_parity_tests
            PRIVATE
                vram_predictor_core
        )

        add_test(NAME vram_llama_fit_parity_tests COMMAND vram_llama_fit_parity_tests)

        if(TARGET vram_fit_harness)
            add_dependencies(vram_llama_fit_parity_tests vram_fit_harness)
        endif()

        if(VRAM_ENABLE_VENDOR_LLAMA)
            add_executable(vram_api_fit_execution_tests
                tests/predictor_api_fit_execution_tests.cpp
            )

            target_link_libraries(vram_api_fit_execution_tests
                PRIVATE
                    vram_predictor_core
            )

            add_test(NAME vram_api_fit_execution_tests COMMAND vram_api_fit_execution_tests)
        endif()
    endif()
endif()
