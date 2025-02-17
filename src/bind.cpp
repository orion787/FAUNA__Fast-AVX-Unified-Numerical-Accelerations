#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "avx2_operations.hpp"
#include "avx_filter_normalize.cpp"
#include "avx2_convolution.cpp"
#include "avx2_image_processor.cpp"

namespace py = pybind11;

PYBIND11_MODULE(fauna, m) {
    py::class_<AVX2Operations>(m, "AVX2Operations")
        .def_static("add", &AVX2Operations::add)
        .def_static("multiply", &AVX2Operations::multiply)
        .def_static("transpose", &AVX2Operations::transpose)
        .def_static("scalar_multiply", &AVX2Operations::scalar_multiply)
        .def_static("add_matrices", &AVX2Operations::add_matrices)
        .def_static("multiply_matrices", &AVX2Operations::multiply_matrices)
        .def_static("cross_product", &AVX2Operations::cross_product)
        .def_static("vector_matrix_multiply", &AVX2Operations::vector_matrix_multiply)
        .def_static("add_4", &AVX2Operations::add_4);
    
    py::class_<AVXFilterNormalize>(m, "AVXFilterNormalize")
        .def_static("filter_and_normalize", &AVXFilterNormalize::filter_and_normalize,
                    py::arg("input_array"), py::arg("min_value"), py::arg("max_value"));

    py::class_<AVXConvolution>(m, "AVXConvolution")
        .def_static("convolve", &AVXConvolution::convolve);

    py::class_<AVXImageProcessor>(m, "AVXImageProcessor")
        .def_static("compute_average_color", &AVXImageProcessor::compute_average_color);
}
