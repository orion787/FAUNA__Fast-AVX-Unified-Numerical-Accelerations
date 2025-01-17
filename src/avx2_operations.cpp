#include "avx2_operations.hpp"
#include <immintrin.h>
#include <stdexcept>

std::vector<float> AVX2Operations::add(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size.");
    }

    size_t size = a.size();
    std::vector<float> result(size);

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 avx_a = _mm256_loadu_ps(&a[i]);
        __m256 avx_b = _mm256_loadu_ps(&b[i]);
        __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
        _mm256_storeu_ps(&result[i], avx_result);
    }

    // Обрабатываем оставшиеся элементы, если они есть
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }

    return result;
}

std::vector<float> AVX2Operations::multiply(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same size.");
    }

    size_t size = a.size();
    std::vector<float> result(size);

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 avx_a = _mm256_loadu_ps(&a[i]);
        __m256 avx_b = _mm256_loadu_ps(&b[i]);
        __m256 avx_result = _mm256_mul_ps(avx_a, avx_b);
        _mm256_storeu_ps(&result[i], avx_result);
    }

    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }

    return result;
}

std::vector<float> AVX2Operations::transpose(const std::vector<float>& matrix) {
    if (matrix.size() != 16) {
        throw std::invalid_argument("Matrix must be 4x4.");
    }

    std::vector<float> result(16);
    __m256 row1 = _mm256_loadu_ps(&matrix[0]);
    __m256 row2 = _mm256_loadu_ps(&matrix[4]);
    __m256 row3 = _mm256_loadu_ps(&matrix[8]);
    __m256 row4 = _mm256_loadu_ps(&matrix[12]);

    __m256 tmp1 = _mm256_unpacklo_ps(row1, row2);
    __m256 tmp2 = _mm256_unpackhi_ps(row1, row2);
    __m256 tmp3 = _mm256_unpacklo_ps(row3, row4);
    __m256 tmp4 = _mm256_unpackhi_ps(row3, row4);

    _mm256_storeu_ps(&result[0], _mm256_permute2f128_ps(tmp1, tmp3, 0x20));
    _mm256_storeu_ps(&result[4], _mm256_permute2f128_ps(tmp1, tmp3, 0x31));
    _mm256_storeu_ps(&result[8], _mm256_permute2f128_ps(tmp2, tmp4, 0x20));
    _mm256_storeu_ps(&result[12], _mm256_permute2f128_ps(tmp2, tmp4, 0x31));

    return result;
}

std::vector<float> AVX2Operations::scalar_multiply(const std::vector<float>& a, float scalar) {
    size_t size = a.size();
    std::vector<float> result(size);

    size_t i = 0;
    __m256 avx_scalar = _mm256_set1_ps(scalar);

    for (; i + 8 <= size; i += 8) {
        __m256 avx_a = _mm256_loadu_ps(&a[i]);
        __m256 avx_result = _mm256_mul_ps(avx_a, avx_scalar);
        _mm256_storeu_ps(&result[i], avx_result);
    }

    for (; i < size; ++i) {
        result[i] = a[i] * scalar;
    }

    return result;
}

std::vector<float> AVX2Operations::add_matrices(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != 16 || b.size() != 16) {
        throw std::invalid_argument("Both matrices must be 4x4.");
    }

    std::vector<float> result(16);
    for (size_t i = 0; i < 16; i += 8) {
        __m256 avx_a = _mm256_loadu_ps(&a[i]);
        __m256 avx_b = _mm256_loadu_ps(&b[i]);
        __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
        _mm256_storeu_ps(&result[i], avx_result);
    }
    return result;
}

std::vector<float> AVX2Operations::multiply_matrices(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != 16 || b.size() != 16) {
        throw std::invalid_argument("Both matrices must be 4x4.");
    }

    std::vector<float> result(16);
    for (size_t i = 0; i < 16; i += 4) {
        __m128 row = _mm_loadu_ps(&a[i]);
        for (size_t j = 0; j < 16; j += 4) {
            __m128 col = _mm_loadu_ps(&b[j]);
            __m128 product = _mm_dp_ps(row, col, 0xF1);
            _mm_storeu_ps(&result[i + j], product);
        }
    }
    return result;
}

std::vector<float> AVX2Operations::cross_product(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("Cross product requires vectors of size 3.");
    }

    std::vector<float> result(3);
    __m128 avx_a = _mm_loadu_ps(&a[0]);
    __m128 avx_b = _mm_loadu_ps(&b[0]);

    __m128 temp1 = _mm_shuffle_ps(avx_a, avx_a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 temp2 = _mm_shuffle_ps(avx_b, avx_b, _MM_SHUFFLE(3, 0, 2, 1));

    __m128 cross = _mm_sub_ps(
        _mm_mul_ps(temp1, avx_b),
        _mm_mul_ps(avx_a, temp2)
    );

    _mm_storeu_ps(&result[0], cross);
    return result;
}

std::vector<float> AVX2Operations::vector_matrix_multiply(const std::vector<float>& vec, const std::vector<float>& mat) {
    if (vec.size() != 4 || mat.size() != 16) {
        throw std::invalid_argument("Vector must be size 4 and matrix must be 4x4.");
    }

    std::vector<float> result(4);
    __m128 avx_vec = _mm_loadu_ps(&vec[0]);
    for (size_t i = 0; i < 4; ++i) {
        __m128 avx_mat_col = _mm_loadu_ps(&mat[i * 4]);
        __m128 product = _mm_dp_ps(avx_vec, avx_mat_col, 0xF1);
        result[i] = _mm_cvtss_f32(product);
    }

    return result;
}

std::vector<float> AVX2Operations::add_4(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != 4 || b.size() != 4) {
        throw std::invalid_argument("Both vectors must have size 4.");
    }

    std::vector<float> result(4);
    __m128 avx_a = _mm_loadu_ps(&a[0]);
    __m128 avx_b = _mm_loadu_ps(&b[0]);
    __m128 avx_result = _mm_add_ps(avx_a, avx_b);
    _mm_storeu_ps(&result[0], avx_result);

    return result;
}
