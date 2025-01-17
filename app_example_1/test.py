import fauna

# Пример для add
a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

result_add = fauna.AVX2Operations.add(a, b)
print(f"Result of add: {result_add}")

# Пример для multiply
result_multiply = fauna.AVX2Operations.multiply(a, b)
print(f"Result of multiply: {result_multiply}")

# Пример для add_matrices
matrix_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
matrix_b = [16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

result_add_matrices = fauna.AVX2Operations.add_matrices(matrix_a, matrix_b)
print(f"Result of add_matrices: {result_add_matrices}")

# Пример для cross_product
vector_a = [1.0, 0.0, 0.0]
vector_b = [0.0, 1.0, 0.0]

result_cross_product = fauna.AVX2Operations.cross_product(vector_a, vector_b)
print(f"Result of cross_product: {result_cross_product}")

# Пример для vector_matrix_multiply
vector = [1.0, 2.0, 3.0, 4.0]
matrix = [1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0]

result_vector_matrix_multiply = fauna.AVX2Operations.vector_matrix_multiply(vector, matrix)
print(f"Result of vector_matrix_multiply: {result_vector_matrix_multiply}")

# Пример для add_4
a_4 = [1.0, 2.0, 3.0, 4.0]
b_4 = [4.0, 3.0, 2.0, 1.0]

result_add_4 = fauna.AVX2Operations.add_4(a_4, b_4)
print(f"Result of add_4: {result_add_4}")
