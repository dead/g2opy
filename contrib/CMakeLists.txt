include_directories(${G2O_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(contrib STATIC
    sba_addons/types_stereo_sba.cpp
    estimate_propagator/SmoothEstimatePropagator.hpp 
    estimate_propagator/SmoothEstimatePropagator.cpp
)

set_target_properties(contrib PROPERTIES OUTPUT_NAME g2opy_contrib)
target_link_libraries(contrib 
    ${BLAS_LIBRARIES}
	${LAPACK_LIBRARIES}
	${CHOLMOD_LIBRARIES}
	${CSPARSE_LIBRARY}
	${G2O_CORE_LIBRARY}
	${G2O_TYPES_SBA}
)