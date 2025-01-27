#include <vector>
#include <tuple>
#include <immintrin.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class AVXImageProcessor {
public:
    static std::tuple<float, float, float> compute_average_color(const std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>& pixels) {
        if (pixels.empty()) {
            throw std::invalid_argument("Pixel array is empty");
        }

        __m256 sum_r = _mm256_setzero_ps();
        __m256 sum_g = _mm256_setzero_ps();
        __m256 sum_b = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 8 <= pixels.size(); i += 8) {
            alignas(32) float r[8], g[8], b[8];
            for (int j = 0; j < 8; ++j) {
                r[j] = std::get<0>(pixels[i + j]);
                g[j] = std::get<1>(pixels[i + j]);
                b[j] = std::get<2>(pixels[i + j]);
            }

            sum_r = _mm256_add_ps(sum_r, _mm256_load_ps(r));
            sum_g = _mm256_add_ps(sum_g, _mm256_load_ps(g));
            sum_b = _mm256_add_ps(sum_b, _mm256_load_ps(b));
        }

        // Суммируем оставшиеся пиксели
        float total_r = 0, total_g = 0, total_b = 0;
        for (; i < pixels.size(); ++i) {
            total_r += std::get<0>(pixels[i]);
            total_g += std::get<1>(pixels[i]);
            total_b += std::get<2>(pixels[i]);
        }

        // Суммируем все значения внутри AVX-регистров
        alignas(32) float sum_r_arr[8], sum_g_arr[8], sum_b_arr[8];
        _mm256_store_ps(sum_r_arr, sum_r);
        _mm256_store_ps(sum_g_arr, sum_g);
        _mm256_store_ps(sum_b_arr, sum_b);

        for (int j = 0; j < 8; ++j) {
            total_r += sum_r_arr[j];
            total_g += sum_g_arr[j];
            total_b += sum_b_arr[j];
        }

        float num_pixels = static_cast<float>(pixels.size());
        return {total_r / num_pixels, total_g / num_pixels, total_b / num_pixels};
    }
};
