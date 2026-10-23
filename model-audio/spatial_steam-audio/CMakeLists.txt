cmake_minimum_required(VERSION 3.17)
project(llm_local_steam_audio_renderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STEAM_AUDIO_SDK_DIR "$ENV{STEAM_AUDIO_SDK_DIR}" CACHE PATH "Steam Audio SDK root")
if(NOT STEAM_AUDIO_SDK_DIR)
    message(FATAL_ERROR "Set STEAM_AUDIO_SDK_DIR to an extracted Steam Audio SDK")
endif()

if(WIN32)
    set(STEAM_AUDIO_LIBRARY_DIR "${STEAM_AUDIO_SDK_DIR}/lib/windows-x64")
elseif(APPLE)
    set(STEAM_AUDIO_LIBRARY_DIR "${STEAM_AUDIO_SDK_DIR}/lib/osx")
else()
    set(STEAM_AUDIO_LIBRARY_DIR "${STEAM_AUDIO_SDK_DIR}/lib/linux-x64")
endif()

find_path(STEAM_AUDIO_INCLUDE_DIR phonon.h HINTS "${STEAM_AUDIO_SDK_DIR}/include" REQUIRED)
find_library(STEAM_AUDIO_LIBRARY NAMES phonon libphonon HINTS "${STEAM_AUDIO_LIBRARY_DIR}" REQUIRED)

add_library(llm_local_spatial_core
    src/audio_io.cpp
    src/binaural_renderer.cpp
    src/json_value.cpp
    src/render_manifest.cpp
    src/spatial_mapping.cpp
)
target_include_directories(llm_local_spatial_core PUBLIC include "${STEAM_AUDIO_INCLUDE_DIR}")
target_link_libraries(llm_local_spatial_core PUBLIC "${STEAM_AUDIO_LIBRARY}")
target_compile_options(llm_local_spatial_core PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>
)

add_executable(steam-audio-render src/main.cpp)
target_link_libraries(steam-audio-render PRIVATE llm_local_spatial_core)

if(UNIX AND NOT APPLE)
    set_target_properties(steam-audio-render PROPERTIES
        BUILD_RPATH "${STEAM_AUDIO_LIBRARY_DIR}"
        INSTALL_RPATH "${STEAM_AUDIO_LIBRARY_DIR}"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(spatial-mapping-tests tests/spatial_mapping_test.cpp src/spatial_mapping.cpp)
    target_include_directories(spatial-mapping-tests PRIVATE include)
    add_test(NAME spatial-mapping-tests COMMAND spatial-mapping-tests)

    add_executable(
        render-manifest-tests
        tests/render_manifest_test.cpp
        src/json_value.cpp
        src/render_manifest.cpp
    )
    target_include_directories(render-manifest-tests PRIVATE include)
    add_test(NAME render-manifest-tests COMMAND render-manifest-tests)

    add_executable(audio-io-tests tests/audio_io_test.cpp src/audio_io.cpp)
    target_include_directories(audio-io-tests PRIVATE include)
    add_test(NAME audio-io-tests COMMAND audio-io-tests)
endif()
