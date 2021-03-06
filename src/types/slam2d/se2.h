#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <g2o/types/slam2d/se2.h>

#include "core/base_vertex.h"
#include "core/base_edge.h"



namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareSE2(py::module & m) {

    py::class_<SE2>(m, "SE2")
        .def(py::init<>())
        .def(py::init<const Isometry2&>())
        .def(py::init<const Vector3&>())
        .def(py::init<double, double, double>())

        .def("translation", &SE2::translation)
        .def("rotation", &SE2::rotation)

        .def(py::self * py::self)
        .def(py::self * Vector2())
        .def(py::self *= py::self)

        .def("inverse", &SE2::inverse)
        .def("__getitem__", &SE2::operator[])
        .def("from_vector", &SE2::fromVector)
        .def("to_vector", &SE2::toVector)
        .def("vector", &SE2::toVector)
        .def("to_isometry", &SE2::toIsometry)
        .def("Isometry2d", &SE2::toIsometry)
    ;


    templatedBaseVertex<3, SE2>(m, "_3_SE2");
    templatedBaseEdge<3, SE2>(m, "_3_SE2");
    templatedBaseMultiEdge<3, SE2>(m, "_3_SE2");

}

}