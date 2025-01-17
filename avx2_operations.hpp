#pragma once
#include <immintrin.h>
#include <vector>

class AVX2Operations {
public:
    static std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> transpose(const std::vector<float>& matrix);
    static std::vector<float> scalar_multiply(const std::vector<float>& a, float scalar);
    static std::vector<float> add_matrices(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> multiply_matrices(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> cross_product(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> vector_matrix_multiply(const std::vector<float>& vec, const std::vector<float>& mat);
    static std::vector<float> add_4(const std::vector<float>& a, const std::vector<float>& b);
};
