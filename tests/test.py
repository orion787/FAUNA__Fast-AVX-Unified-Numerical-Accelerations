import pytest
import numpy as np
from fauna import AVX2Operations, AVXFilterNormalize, AVXConvolution, AVXImageProcessor

# Тест для AVX2Operations
@pytest.mark.parametrize("a, b, expected", [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
     [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
])
def test_add(a, b, expected):
    assert AVX2Operations.add(a, b) == pytest.approx(expected)

@pytest.mark.parametrize("a, b, expected", [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
     [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
     [8.0, 14.0, 18.0, 20.0, 20.0, 18.0, 14.0, 8.0])
])
def test_multiply(a, b, expected):
    assert AVX2Operations.multiply(a, b) == pytest.approx(expected)

@pytest.mark.parametrize("matrix_a, matrix_b, expected", [
    ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
     [16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
     [17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0])
])
def test_add_matrices(matrix_a, matrix_b, expected):
    assert AVX2Operations.add_matrices(matrix_a, matrix_b) == pytest.approx(expected)

@pytest.mark.parametrize("vector_a, vector_b, expected", [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
])
def test_cross_product(vector_a, vector_b, expected):
    assert AVX2Operations.cross_product(vector_a, vector_b) == pytest.approx(expected)

@pytest.mark.parametrize("vector, matrix, expected", [
    ([1.0, 2.0, 3.0, 4.0],
     [1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0],
     [1.0, 2.0, 3.0, 4.0])
])
def test_vector_matrix_multiply(vector, matrix, expected):
    assert AVX2Operations.vector_matrix_multiply(vector, matrix) == pytest.approx(expected)

@pytest.mark.parametrize("a_4, b_4, expected", [
    ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [5.0, 5.0, 5.0, 5.0])
])
def test_add_4(a_4, b_4, expected):
    assert AVX2Operations.add_4(a_4, b_4) == pytest.approx(expected)

# Тест для AVXFilterNormalize
def test_filter_and_normalize():
    data = np.random.uniform(-10, 10, size=1000000).astype(np.float32)
    result = AVXFilterNormalize.filter_and_normalize(data, min_value=0.0, max_value=5.0)
    assert result.min() >= 0.0
    assert result.max() <= 5.0

# Тест для AVXConvolution
def test_convolution():
    signal = np.random.uniform(-10, 10, size=100000).astype(np.float32)
    kernel = np.array([0.2, 0.5, 0.2], dtype=np.float32)
    result = AVXConvolution.convolve(signal, kernel)
    assert len(result) == len(signal) - len(kernel) + 1

# Тест для AVXImageProcessor
def test_compute_average_color():
    pixels = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    avg_color = AVXImageProcessor.compute_average_color(pixels)
    assert avg_color == pytest.approx((127.5, 127.5, 127.5))
