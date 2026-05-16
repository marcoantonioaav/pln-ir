#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "initialization.h"

namespace py = pybind11;

PYBIND11_MODULE(initialization_cpp, m) {
    m.doc() = "C++ module for initialization approaches with distance tracking";
    
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("index", &SearchResult::index)
        .def_readonly("distance", &SearchResult::distance);

    py::class_<InitializationApproach>(m, "InitializationApproach")
        .def("get_distance_computations", &InitializationApproach::get_distance_computations)
        .def("reset_distance_computations", &InitializationApproach::reset_distance_computations);
    
    py::class_<RandomPointsInit, InitializationApproach>(m, "RandomPointsInit")
        .def(py::init<uint32_t>(), py::arg("seed") = 42)
        .def("build", &RandomPointsInit::build)
        .def("search", &RandomPointsInit::search);
        
    py::class_<MedoidInit, InitializationApproach>(m, "MedoidInit")
        .def(py::init<>())
        .def("build", &MedoidInit::build)
        .def("search", &MedoidInit::search);
}
