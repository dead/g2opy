cmake_minimum_required(VERSION 3.1)
project(g2opy)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${g2opy_SOURCE_DIR}/cmake_modules)

find_package(Cholmod REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CSparse REQUIRED)
find_package(BLAS)
find_package(LAPACK)

OPTION(G2O_FAST_MATH "Enable fast math operations" OFF)

ExternalProject_Add(
	g2o
	GIT_REPOSITORY "https://github.com/dead/g2o.git"
	GIT_TAG "master"
	INSTALL_COMMAND
		cmake
		--build .
		--target install
		--config Release
	CMAKE_ARGS
		-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/g2oInstall
		-DG2O_USE_CHOLMOD=ON
		-DG2O_USE_CSPARSE=ON
		-DG2O_USE_OPENGL=OFF
		-DG2O_BUILD_EXAMPLES=OFF
		-DG2O_FAST_MATH=${G2O_FAST_MATH}
		-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
)

if(MSVC)
	MESSAGE(STATUS "Compiling with MSVC")
	
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox /Oi")
	SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Ox /Oi")

	SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
	SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
	
	add_definitions("/EHsc") # exception handling
	add_definitions("/Ob2")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS=1)  # Avoid deprecated fprintf(), etc
	add_definitions("/nologo")
	add_definitions("/wd4251") # disable warning on missing DLL interfaces
	add_definitions("/wd4244") # Conversion from number_t -> int
	add_definitions("/wd4267") # Conversion during return
	add_definitions("/wd4522") # Duplicated operator=() in Eigen headers
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR}/g2oInstall/include)
include_directories(${CHOLMOD_INCLUDE_DIR})

link_directories(${CMAKE_BINARY_DIR}/g2oInstall/lib)

add_subdirectory(pybind11)
add_subdirectory(contrib)

add_dependencies(contrib g2o)

pybind11_add_module(g2opy src/g2o.cpp)

MACRO(G2O_LIBRARY MYLIBRARY MYLIBRARYNAME)
	SET(${MYLIBRARY} "${CMAKE_SHARED_MODULE_PREFIX}g2o_${MYLIBRARYNAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
	SET(${MYLIBRARY}_DEBUG "${CMAKE_SHARED_MODULE_PREFIX}g2o_${MYLIBRARYNAME}_d${CMAKE_SHARED_LIBRARY_SUFFIX}")
ENDMACRO(G2O_LIBRARY LIBRARY LIBRARYNAME)

G2O_LIBRARY(G2O_STUFF_LIBRARY stuff)
G2O_LIBRARY(G2O_CORE_LIBRARY core)
G2O_LIBRARY(G2O_CLI_LIBRARY cli)
G2O_LIBRARY(G2O_SOLVER_CHOLMOD solver_cholmod)
G2O_LIBRARY(G2O_SOLVER_CSPARSE solver_csparse)
G2O_LIBRARY(G2O_SOLVER_CSPARSE_EXTENSION csparse_extension)
G2O_LIBRARY(G2O_SOLVER_DENSE solver_dense)
G2O_LIBRARY(G2O_SOLVER_PCG solver_pcg)
G2O_LIBRARY(G2O_SOLVER_SLAM2D_LINEAR solver_slam2d_linear)
G2O_LIBRARY(G2O_SOLVER_STRUCTURE_ONLY solver_structure_only)
G2O_LIBRARY(G2O_SOLVER_EIGEN solver_eigen)
G2O_LIBRARY(G2O_TYPES_DATA types_data)
G2O_LIBRARY(G2O_TYPES_ICP types_icp)
G2O_LIBRARY(G2O_TYPES_SBA types_sba)
G2O_LIBRARY(G2O_TYPES_SCLAM2D types_sclam2d)
G2O_LIBRARY(G2O_TYPES_SIM3 types_sim3)
G2O_LIBRARY(G2O_TYPES_SLAM2D types_slam2d)
G2O_LIBRARY(G2O_TYPES_SLAM3D types_slam3d)

target_link_libraries(g2opy PRIVATE
	${BLAS_LIBRARIES}
	${LAPACK_LIBRARIES}
	${CHOLMOD_LIBRARIES}
	${CSPARSE_LIBRARY}
	${G2O_CORE_LIBRARY}
	${G2O_STUFF_LIBRARY}
	${G2O_SOLVER_CHOLMOD}
	${G2O_SOLVER_CSPARSE}
	${G2O_SOLVER_CSPARSE_EXTENSION}
	${G2O_SOLVER_DENSE}
	${G2O_SOLVER_PCG}
	${G2O_SOLVER_SLAM2D_LINEAR}
	${G2O_SOLVER_STRUCTURE_ONLY}
	${G2O_SOLVER_EIGEN}
	${G2O_TYPES_DATA}
	${G2O_TYPES_ICP}
	${G2O_TYPES_SBA}
	${G2O_TYPES_SCLAM2D}
	${G2O_TYPES_SIM3}
	${G2O_TYPES_SLAM2D}
	${G2O_TYPES_SLAM3D}
	contrib
)

add_dependencies(g2opy g2o contrib)
