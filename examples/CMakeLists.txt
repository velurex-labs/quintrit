# Builds the C and C++ examples against the Quintrit C library (built with cargo by the quintrit-c CMake project).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
#   ctest --test-dir build -C Release --output-on-failure      # runs the two gemv examples
#   build/c_generate model.gguf                                 # the generate examples need a GGUF model
cmake_minimum_required(VERSION 3.14)
project(quintrit_examples LANGUAGES C CXX)

set(QUINTRIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../rust/crates/quintrit-c" quintrit_c)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

foreach(name gemv generate)
  add_executable(c_${name} c/${name}.c)
  target_link_libraries(c_${name} PRIVATE quintrit::quintrit)
  add_executable(cpp_${name} cpp/${name}.cpp)
  target_link_libraries(cpp_${name} PRIVATE quintrit::quintrit)
endforeach()

enable_testing()
add_test(NAME c_gemv COMMAND c_gemv)
add_test(NAME cpp_gemv COMMAND cpp_gemv)

# The shared library must be found when the examples run.
get_target_property(_quintrit_lib quintrit::quintrit IMPORTED_LOCATION)
get_filename_component(_quintrit_lib_dir "${_quintrit_lib}" DIRECTORY)
if(WIN32)
  foreach(name c_gemv c_generate cpp_gemv cpp_generate)
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_quintrit_lib}" "$<TARGET_FILE_DIR:${name}>")
  endforeach()
else()
  foreach(name c_gemv c_generate cpp_gemv cpp_generate)
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${_quintrit_lib_dir}")
  endforeach()
endif()
