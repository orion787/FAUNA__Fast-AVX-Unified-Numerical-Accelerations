#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h> // Для AVX2
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

class AVXFilterNormalize {
public:
    static py::array_t<float> filter_and_normalize(
        const py::array_t<float>& input_array,
        float min_value,
        float max_value
    ) {
        // Проверяем входной массив
        auto buf = input_array.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("Входной массив должен быть одномерным.");
        }

        const float* data = static_cast<float*>(buf.ptr);
        size_t size = buf.size;

        // Создаём буфер для результата
        std::vector<float> filtered_data;
        filtered_data.reserve(size); // Резервируем место для скорости

        // Оптимизированный AVX2-код
        __m256 min_vec = _mm256_set1_ps(min_value);
        __m256 max_vec = _mm256_set1_ps(max_value);

        for (size_t i = 0; i + 8 <= size; i += 8) {
            __m256 values = _mm256_loadu_ps(data + i);
            __m256 mask_min = _mm256_cmp_ps(values, min_vec, _CMP_GE_OQ);
            __m256 mask_max = _mm256_cmp_ps(values, max_vec, _CMP_LE_OQ);
            __m256 mask = _mm256_and_ps(mask_min, mask_max);

            int bitmask = _mm256_movemask_ps(mask);
            for (int j = 0; j < 8; ++j) {
                if (bitmask & (1 << j)) {
                    filtered_data.push_back(data[i + j]);
                }
            }
        }

        // Обработка оставшихся элементов
        for (size_t i = (size / 8) * 8; i < size; ++i) {
            if (data[i] >= min_value && data[i] <= max_value) {
                filtered_data.push_back(data[i]);
            }
        }

        // Нормализация данных
        if (filtered_data.empty()) {
            throw std::runtime_error("Нет элементов, попадающих в заданный диапазон.");
        }

        float min_filtered = *std::min_element(filtered_data.begin(), filtered_data.end());
        float max_filtered = *std::max_element(filtered_data.begin(), filtered_data.end());

        float range = max_filtered - min_filtered;
        if (range == 0.0f) {
            throw std::runtime_error("Все элементы одинаковы, нормализация невозможна.");
        }

        for (float& value : filtered_data) {
            value = (value - min_filtered) / range;
        }

        // Возвращаем результат в виде numpy массива
        py::array_t<float> result(filtered_data.size());
        std::copy(filtered_data.begin(), filtered_data.end(), result.mutable_data());

        return result;
    }
};
