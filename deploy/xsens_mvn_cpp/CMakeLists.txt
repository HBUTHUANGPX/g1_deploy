cmake_minimum_required(VERSION 3.16)
project(xsens_mvn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(XSENS_MVN_CPP_BUILD_TESTS "Build xsens_mvn_cpp C++ tests" ON)
option(XSENS_MVN_CPP_BUILD_PYTHON "Build xsens_mvn_cpp pybind11 module" ON)

set(XSENS_MVN_SDK_VENDOR_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsens_mvn_sdk"
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread chrono)

file(GLOB XSENS_MVN_SDK_SOURCES
  "${XSENS_MVN_SDK_VENDOR_ROOT}/src/*.cpp"
)

add_library(xsens_mvn_sdk_reused STATIC
  ${XSENS_MVN_SDK_SOURCES}
)
target_include_directories(xsens_mvn_sdk_reused PUBLIC
  "${XSENS_MVN_SDK_VENDOR_ROOT}/include"
)
target_link_libraries(xsens_mvn_sdk_reused PUBLIC
  Boost::system
  Boost::thread
  Boost::chrono
  Eigen3::Eigen
)

add_library(xsens_mvn_cpp_core STATIC
  src/latest_frame_store.cpp
  src/udp_datagram_socket.cpp
  src/xsens_datagram_parser.cpp
  src/xsens_raw_frame_assembler.cpp
  src/xsens_mvn_stream_reader.cpp
)
target_include_directories(xsens_mvn_cpp_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${XSENS_MVN_SDK_VENDOR_ROOT}/include"
)
target_link_libraries(xsens_mvn_cpp_core PUBLIC
  xsens_mvn_sdk_reused
  Boost::system
  Boost::thread
  Boost::chrono
  Eigen3::Eigen
  pthread
)

if(XSENS_MVN_CPP_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE XSENS_MVN_CPP_PYBIND11_CMAKEDIR_RESULT
    OUTPUT_VARIABLE XSENS_MVN_CPP_PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(XSENS_MVN_CPP_PYBIND11_CMAKEDIR_RESULT EQUAL 0 AND
      EXISTS "${XSENS_MVN_CPP_PYBIND11_CMAKEDIR}/pybind11Config.cmake")
    set(pybind11_DIR
      "${XSENS_MVN_CPP_PYBIND11_CMAKEDIR}"
      CACHE PATH "pybind11 CMake package from the active Python environment" FORCE
    )
  endif()

  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(xsens_mvn_cpp_py
    bindings/xsens_mvn_pybind.cpp
  )
  target_link_libraries(xsens_mvn_cpp_py PRIVATE
    xsens_mvn_cpp_core
  )
endif()

if(XSENS_MVN_CPP_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(test_latest_frame_store
    tests/test_latest_frame_store.cpp
  )
  target_link_libraries(test_latest_frame_store PRIVATE
    xsens_mvn_cpp_core
    GTest::GTest
    GTest::Main
  )
  add_test(NAME test_latest_frame_store COMMAND test_latest_frame_store)
endif()
