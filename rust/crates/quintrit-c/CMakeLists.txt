# Builds libquintrit_c with cargo and the C / C++ tests of the C API against it.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#   ctest --test-dir build -C Release --output-on-failure
#
# Other projects can use this file as a package: add_subdirectory(<path to this directory>) and link
# quintrit::quintrit (shared) or quintrit::quintrit_static. Set QUINTRIT_BUILD_TESTS=OFF to skip the test programs.
cmake_minimum_required(VERSION 3.14)
project(quintrit_c LANGUAGES C CXX)

option(QUINTRIT_BUILD_TESTS "Build and register the C and C++ test programs" ON)
set(QUINTRIT_CARGO_PROFILE "" CACHE STRING
    "Cargo profile: release, debug, or release-unwind (panics become QUINTRIT_ERR_INTERNAL). Default: from CMAKE_BUILD_TYPE")

find_program(QUINTRIT_CARGO cargo REQUIRED)

# The cargo workspace is two directories up (crates/quintrit-c -> crates -> workspace root).
get_filename_component(QUINTRIT_WORKSPACE "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(QUINTRIT_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target" ABSOLUTE)

if(QUINTRIT_CARGO_PROFILE STREQUAL "")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(QUINTRIT_CARGO_PROFILE debug)
  else()
    set(QUINTRIT_CARGO_PROFILE release)
  endif()
endif()
if(QUINTRIT_CARGO_PROFILE STREQUAL "debug")
  set(_cargo_profile_args)
elseif(QUINTRIT_CARGO_PROFILE STREQUAL "release")
  set(_cargo_profile_args --release)
else()
  set(_cargo_profile_args --profile ${QUINTRIT_CARGO_PROFILE})
endif()
set(_lib_dir "${QUINTRIT_TARGET_DIR}/${QUINTRIT_CARGO_PROFILE}")

if(WIN32)
  set(_shared "${_lib_dir}/quintrit_c.dll")
  set(_implib "${_lib_dir}/quintrit_c.dll.lib")
  set(_static "${_lib_dir}/quintrit_c.lib")
elseif(APPLE)
  set(_shared "${_lib_dir}/libquintrit_c.dylib")
  set(_static "${_lib_dir}/libquintrit_c.a")
else()
  set(_shared "${_lib_dir}/libquintrit_c.so")
  set(_static "${_lib_dir}/libquintrit_c.a")
endif()

# On Unix, ensure the shared object embeds an ELF SONAME (or Mach-O install_name) so that
# downstreams linking against it record only the library name in DT_NEEDED instead of a relative path.
if(APPLE)
  set(_extra_rustflags "-C link-arg=-Wl,-install_name,@rpath/libquintrit_c.dylib")
elseif(UNIX)
  set(_extra_rustflags "-C link-arg=-Wl,-soname,libquintrit_c.so")
else()
  set(_extra_rustflags "")
endif()

if(_extra_rustflags)
  if(DEFINED ENV{RUSTFLAGS} AND NOT "$ENV{RUSTFLAGS}" STREQUAL "")
    set(_cargo_env ${CMAKE_COMMAND} -E env "RUSTFLAGS=$ENV{RUSTFLAGS} ${_extra_rustflags}")
  else()
    set(_cargo_env ${CMAKE_COMMAND} -E env "RUSTFLAGS=${_extra_rustflags}")
  endif()
else()
  set(_cargo_env)
endif()

# Always run cargo; it is a no-op when nothing changed. Both library kinds are produced by one invocation.
add_custom_target(quintrit_cargo ALL
  COMMAND ${_cargo_env} ${QUINTRIT_CARGO} build -p quintrit-c ${_cargo_profile_args} --target-dir "${QUINTRIT_TARGET_DIR}"
  WORKING_DIRECTORY "${QUINTRIT_WORKSPACE}"
  BYPRODUCTS "${_shared}" "${_static}"
  COMMENT "cargo build -p quintrit-c (${QUINTRIT_CARGO_PROFILE})"
  VERBATIM)

add_library(quintrit SHARED IMPORTED GLOBAL)
set_target_properties(quintrit PROPERTIES
  IMPORTED_LOCATION "${_shared}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
if(APPLE)
  set_target_properties(quintrit PROPERTIES IMPORTED_SONAME "libquintrit_c.dylib")
elseif(UNIX)
  set_target_properties(quintrit PROPERTIES IMPORTED_SONAME "libquintrit_c.so")
endif()
if(WIN32)
  set_target_properties(quintrit PROPERTIES IMPORTED_IMPLIB "${_implib}")
endif()
add_dependencies(quintrit quintrit_cargo)
add_library(quintrit::quintrit ALIAS quintrit)

# The static library carries the Rust standard library; the system libraries below are what it needs
# (`rustc --print native-static-libs` prints the authoritative list for your target).
add_library(quintrit_static STATIC IMPORTED GLOBAL)
set_target_properties(quintrit_static PROPERTIES
  IMPORTED_LOCATION "${_static}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
if(WIN32)
  set_property(TARGET quintrit_static PROPERTY INTERFACE_LINK_LIBRARIES kernel32 ntdll userenv ws2_32 dbghelp)
elseif(APPLE)
  set_property(TARGET quintrit_static PROPERTY INTERFACE_LINK_LIBRARIES System c m)
else()
  set_property(TARGET quintrit_static PROPERTY INTERFACE_LINK_LIBRARIES pthread dl m c gcc_s)
endif()
add_dependencies(quintrit_static quintrit_cargo)
add_library(quintrit::quintrit_static ALIAS quintrit_static)

if(QUINTRIT_BUILD_TESTS)
  enable_testing()
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  foreach(name IN ITEMS test_kernel_c test_kernel_cpp)
    if(name MATCHES "_cpp$")
      add_executable(${name} tests/cpp/test_kernel.cpp)
    else()
      add_executable(${name} tests/c/test_kernel.c)
    endif()
    target_link_libraries(${name} PRIVATE quintrit::quintrit)
    if(MSVC)
      target_compile_options(${name} PRIVATE /W4)
    else()
      target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic)
    endif()
    # The DLL / shared object must be found at run time: copy it next to the executable on Windows, set an rpath elsewhere.
    if(WIN32)
      add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_shared}" "$<TARGET_FILE_DIR:${name}>")
    else()
      set_target_properties(${name} PROPERTIES BUILD_RPATH "${_lib_dir}")
    endif()
    add_test(NAME ${name} COMMAND ${name})
  endforeach()

  # The same C test linked statically, to check that the static library is complete.
  add_executable(test_kernel_c_static tests/c/test_kernel.c)
  target_link_libraries(test_kernel_c_static PRIVATE quintrit::quintrit_static)
  add_test(NAME test_kernel_c_static COMMAND test_kernel_c_static)
endif()
