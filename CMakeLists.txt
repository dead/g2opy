cmake_minimum_required(VERSION 3.1)
project(g2opy)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${g2opy_SOURCE_DIR}/cmake_modules)

find_package(Cholmod REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CSparse REQUIRED)
find_package(BLAS)
find_package(LAPACK)

add_subdirectory(pybind11)

set(G2O_OPENGL_FOUND OFF)
set(G2O_HAVE_CHOLMOD ${CHOLMOD_FOUND})
set(G2O_HAVE_CSPARSE ${CSPARSE_FOUND})
set(G2O_SHARED_LIBS  OFF)
set(G2O_LGPL_SHARED_LIBS OFF)
set(G2O_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
configure_file(g2o/config.h.in "${PROJECT_BINARY_DIR}/g2o/g2o/config.h")

file(GLOB G2O_CORE_FILES
    "g2o/g2o/core/*.cpp"
)

list(FILTER G2O_CORE_FILES EXCLUDE REGEX ".*_test.cpp$")

file(GLOB_RECURSE G2O_SOLVERS_FILES
    "g2o/g2o/solvers/*.cpp"
)

file(GLOB_RECURSE G2O_STUFF_FILES
	"g2o/g2o/stuff/*.cpp"
	"g2o/g2o/stuff/*.c"
)

list(FILTER G2O_STUFF_FILES EXCLUDE REGEX ".*opengl_.*$")

file(GLOB G2O_TYPES_FILES
	"g2o/g2o/types/data/*.cpp"
	"g2o/g2o/types/icp/*.cpp"
	"g2o/g2o/types/sba/*.cpp"
	"g2o/g2o/types/sclam2d/*.cpp"
	"g2o/g2o/types/sim3/*.cpp"
	"g2o/g2o/types/slam2d/*.cpp"
	"g2o/g2o/types/slam2d_addons/*.cpp"
	"g2o/g2o/types/slam3d/*.cpp"
    "g2o/g2o/types/slam3d_addons/*.cpp"
)

list(FILTER G2O_TYPES_FILES EXCLUDE REGEX ".*/slam3d/dquat2mat_maxima_generated.cpp$")
list(FILTER G2O_TYPES_FILES EXCLUDE REGEX ".*/slam3d/test_.*.cpp$")

if(WIN32)
  add_definitions(-DWINDOWS)
  add_definitions(/bigobj /nologo /EHsc /GF /MP)
  add_definitions(/wd4267 /wd4244 /wd4996)
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox /Oi")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Ox /Oi")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

  if(G2O_FAST_MATH)
  	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /fp:fast")
  endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
	option(BUILD_WITH_MARCH_NATIVE "Build with \"-march native\"" ON)

	if(BUILD_WITH_MARCH_NATIVE AND NOT "${CMAKE_SYSTEM_PROCESSOR}" MATCHES "arm" AND "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
	endif()

	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
endif()

include_directories(${g2opy_SOURCE_DIR}/g2o
					${PROJECT_BINARY_DIR}/g2o
					${EIGEN3_INCLUDE_DIR}
					${CHOLMOD_INCLUDE_DIR}
					${BLAS_INCLUDE_DIR}
					${LAPACK_INCLUDE_DIR})

pybind11_add_module(g2opy src/g2o.cpp
						  src/contrib/estimate_propagator/SmoothEstimatePropagator.cpp
						  src/contrib/sba_addons/types_stereo_sba.cpp
						  ${G2O_CORE_FILES}
						  ${G2O_SOLVERS_FILES}
						  ${G2O_STUFF_FILES}
						  ${G2O_TYPES_FILES})


target_link_libraries(g2opy PRIVATE
	${BLAS_LIBRARIES}
	${LAPACK_LIBRARIES}
	${CHOLMOD_LIBRARIES}
	${CSPARSE_LIBRARY}
)
