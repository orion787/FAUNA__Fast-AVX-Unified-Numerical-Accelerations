#include <vector>
#include <immintrin.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class AVXConvolution {
public:
    static std::vector<float> convolve(const std::vector<float>& signal, const std::vector<float>& kernel) {
        int signal_size = signal.size();
        int kernel_size = kernel.size();
        int output_size = signal_size - kernel_size + 1;

        if (kernel_size > signal_size || kernel_size == 0) {
            throw std::invalid_argument("Invalid kernel size for convolution");
        }

        std::vector<float> result(output_size, 0.0f);

        for (int i = 0; i < output_size; ++i) {
            __m256 acc = _mm256_setzero_ps(); // Инициализация сумматора

            int j = 0;
            for (; j <= kernel_size - 8; j += 8) {
                __m256 signal_chunk = _mm256_loadu_ps(&signal[i + j]);
                __m256 kernel_chunk = _mm256_loadu_ps(&kernel[j]);
                acc = _mm256_fmadd_ps(signal_chunk, kernel_chunk, acc); // Умножение и суммирование
            }

            // Вычисление суммы элементов в векторе acc
            float acc_array[8];
            _mm256_storeu_ps(acc_array, acc);
            float sum = 0.0f;
            for (float v : acc_array) sum += v;

            // Обработка оставшихся элементов
            for (; j < kernel_size; ++j) {
                sum += signal[i + j] * kernel[j];
            }

            result[i] = sum;
        }

        return result;
    }
};
