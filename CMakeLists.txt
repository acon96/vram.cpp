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

    # Minimize vendored llama.cpp/ggml surface for wasm predictor builds.
    # Keep only the CPU path required by in-process fit execution.
    set(LLAMA_BUILD_HTML OFF CACHE BOOL "Disable llama html target in wasm predictor builds" FORCE)
    set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "Disable llama webui assets in wasm predictor builds" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "Disable openssl in wasm predictor builds" FORCE)
    set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "Disable llama test install targets" FORCE)

    set(GGML_ACCELERATE OFF CACHE BOOL "Disable accelerate backend for wasm predictor builds" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP for wasm predictor builds" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile kernels for wasm predictor builds" FORCE)
    set(GGML_CPU_REPACK OFF CACHE BOOL "Disable CPU repack path for wasm predictor builds" FORCE)

    # Explicitly pin non-CPU backends and optional backend extras off.
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA backend" FORCE)
    set(GGML_HIP OFF CACHE BOOL "Disable HIP backend" FORCE)
    set(GGML_MUSA OFF CACHE BOOL "Disable MUSA backend" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan backend" FORCE)
    set(GGML_WEBGPU OFF CACHE BOOL "Disable WebGPU backend" FORCE)
    set(GGML_METAL OFF CACHE BOOL "Disable Metal backend" FORCE)
    set(GGML_OPENCL OFF CACHE BOOL "Disable OpenCL backend" FORCE)
    set(GGML_SYCL OFF CACHE BOOL "Disable SYCL backend" FORCE)
    set(GGML_RPC OFF CACHE BOOL "Disable RPC backend" FORCE)
    set(GGML_OPENVINO OFF CACHE BOOL "Disable OpenVINO backend" FORCE)
    set(GGML_ZDNN OFF CACHE BOOL "Disable zDNN backend" FORCE)
    set(GGML_ZENDNN OFF CACHE BOOL "Disable ZenDNN backend" FORCE)
    set(GGML_HEXAGON OFF CACHE BOOL "Disable Hexagon backend" FORCE)
    set(GGML_VIRTGPU OFF CACHE BOOL "Disable VirtGPU backend" FORCE)
    set(GGML_VIRTGPU_BACKEND OFF CACHE BOOL "Disable VirtGPU backend server" FORCE)

    # Keep architecture-specific CPU feature toggles off for wasm cross-builds.
    set(GGML_LASX OFF CACHE BOOL "Disable LASX in wasm predictor builds" FORCE)
    set(GGML_LSX OFF CACHE BOOL "Disable LSX in wasm predictor builds" FORCE)
    set(GGML_RVV OFF CACHE BOOL "Disable RVV in wasm predictor builds" FORCE)
    set(GGML_RV_ZFH OFF CACHE BOOL "Disable RISC-V ZFH in wasm predictor builds" FORCE)
    set(GGML_RV_ZVFH OFF CACHE BOOL "Disable RISC-V ZVFH in wasm predictor builds" FORCE)
    set(GGML_RV_ZICBOP OFF CACHE BOOL "Disable RISC-V ZICBOP in wasm predictor builds" FORCE)
    set(GGML_RV_ZIHINTPAUSE OFF CACHE BOOL "Disable RISC-V ZIHINTPAUSE in wasm predictor builds" FORCE)
    set(GGML_WEBGPU_JSPI OFF CACHE BOOL "Disable WebGPU JSPI mode when WebGPU is off" FORCE)
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

if(EMSCRIPTEN)
    if(VRAM_BUILD_WASM)
        add_executable(vram_predictor
            cpp/src/predictor_main.cpp
        )

        target_link_libraries(vram_predictor
            PRIVATE
                vram_predictor_core
        )

        # Emit a JS loader + WASM pair for browser/worker usage.
        set_target_properties(vram_predictor PROPERTIES SUFFIX ".js")

        target_link_options(vram_predictor PRIVATE
            "-sALLOW_MEMORY_GROWTH=1"
            "-sFETCH=1"
            "-sENVIRONMENT=web,worker"
            "-sMODULARIZE=1"
            "-sEXPORT_NAME=createVRAMPredictorModule"
            "-sEXPORTED_FUNCTIONS=['_malloc','_free','_vram_predictor_predict_json']"
            "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','FS']"
        )
    endif()
else()
    add_executable(vram_predictor
        cpp/src/predictor_main.cpp
    )

    target_link_libraries(vram_predictor
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

        if(TARGET vram_predictor)
            add_dependencies(vram_llama_fit_parity_tests vram_predictor)
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
