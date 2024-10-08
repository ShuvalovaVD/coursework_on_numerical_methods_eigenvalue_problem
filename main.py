'''
ДОДЕЛАТЬ:
1) нахождение собственных векторов - решаем слау, можно методом Гаусса
4) QR-разложение по методам: Преобразование Хаусхолдера, Поворот Гивенса
'''

def multiply_two_matrices(matrix_a, matrix_b):
    """
    Вспомогательная функция. Умножает две матрицы matrix_a и matrix_b друг на друга и возвращает итоговую матрицу
    matrix_c. Предполагается, что умножение матриц определено, то есть кол-во столбцов matrix_a = кол-ву строк matrix_b.
    """
    # размеры матриц matrix_a и matrix_b: кол-во строк и столбцов соответственно
    m_a, n_a, m_b, n_b = len(matrix_a), len(matrix_a[0]), len(matrix_b), len(matrix_b[0])  # предполагается: n_a = m_b
    m_c, n_c = m_a, n_b  # размеры итоговой матрицы matrix_c: кол-во строк и столбцов соответственно
    matrix_c = []  # создадим матрицу matrix_c, заполненную нулями
    for i in range(m_c): matrix_c.append([0] * n_c)
    # при умножении двух матриц строки левой матрицы покоординатно умножаются на столбцы правой матрицы
    for i in range(m_c):
        for j in range(n_c):
            # элемент matrix_c[i][j] равен покоординатному умножению i-ой строки matrix_a на j-ый столбцец matrix_b
            matrix_c[i][j] = sum([matrix_a[i][k] * matrix_b[k][j] for k in range(n_a)])
    return matrix_c


def qr_decomposition_gram_schmidt_process(matrix_a):
    '''
    Функция осуществляет QR-разложение вещественной матрицы matrix_a на ортогональную матрицу matrix_q
    и верхнетреугольную матрицу matrix_r: matrix_a = matrix_q * matrix_r.
    Для реализации QR-разложения используется процесс Грама-Шмидта.
    Далее в комментариях будет описан ход этого процесса.
    '''

    n = len(matrix_a)  # количество строк и столбцов соответственно, предполагается, что матрица квадратная
    vector_columns = [[matrix_a[i][j] for i in range(n)] for j in range(n)]  # вектор-столбцы a_1, a_2, ..., a_n

    # для вектор-столбцов a_1, a_2, ..., a_n нужно получить систему ортогональных векторов b_1, b_2, ..., b_n
    # система ортогональных векторов - это система векторов, где все векторы попарно ортогональны, т.е. перпендикулярны
    orthogonal_vectors = [vector_columns[0]]  # b_1 = a_1
    # далее b_j (2 <= j <= n) рассчитывается по следующим формулам:
    # b_2 = a_2 - proj_b_1_a_2; b_3 = a_3 - proj_b_1_a_3 - proj_b_2_a_3; ...
    # b_n = a_n - proj_b_1_a_n - proj_b_2_a_n - ... - proj_b_n-1_a_n
    # proj_b_a - проекция вектора a на вектор b, proj_b_a = scal_a_b / scal_b_b * b
    # scal_a_b - скалярное произведение векторов a(x1, y1, z1) и b(x2, y2, z2), scal_a_b = x1 * x2 + y1 * y2 + z1 * z2
    for j in range(1, n):  # в списках индексация с нуля => старт j с 1, а не с 2
        a_j = vector_columns[j]
        projections_b_j = []  # посчитаем для b_j проекции proj_b_1_a_j, proj_b_2_a_j, ..., proj_b_j-1_a_j
        for i in range(j):
            b_i = orthogonal_vectors[i]
            scal_a_j_b_i = sum([a_j[k] * b_i[k] for k in range(n)])
            scal_b_i_b_i = sum([b_i[k] * b_i[k] for k in range(n)])
            scal_mult = scal_a_j_b_i / scal_b_i_b_i
            proj_b_i_a_j = [scal_mult * elem for elem in b_i]
            projections_b_j.append(proj_b_i_a_j)
        b_j = [a_j[i] - sum([projections_b_j[k][i] for k in range(j)]) for i in range(n)]
        orthogonal_vectors.append(b_j)

    # из ортогональных векторов b_1, b_2, ..., b_n нужно получить нормированные векторы e_1, e_2, ..., e_n
    # нормированный (единичный) вектор - это вектор единичной длины, он получается делением вектора на его норму
    # e_j = b_j / ||b_j||, где ||b_j|| - норма вектора, ||b_j|| = sqrt(sum(elem_b_j**2)) для 1 <= j <= n
    normed_vectors = []
    for j in range(n):
        b_j = orthogonal_vectors[j]
        b_j_norm = sum([elem ** 2 for elem in b_j]) ** 0.5
        e_j = [elem / b_j_norm for elem in b_j]
        normed_vectors.append(e_j)

    # теперь найдем ортогональную матрицу matrix_q и верхнетреугольную матрицу matrix_r
    # ортогональная матрица - это квадратная матрица с вещественными элементами, результат умножения которой на
    # транспонированную матрицу равен единичной матрице:
    # matrix_q * matrix_q_transposed = matrix_q_transposed * matrix_q = E
    # верхнетреугольная матрица - это матрица, у которой все элементы, стоящие ниже главной диагонали, равны 0

    # столбцы ортогональной матрицы matrix_q формируются из векторов e_j для 1 <= j <= n
    matrix_q = [[normed_vectors[j][i] for j in range(n)] for i in range(n)]  # её размеры: m x n

    # найдем верхнетреугольную матрицу matrix_r из выражения matrix_a = matrix_q * matrix_r
    # умножим обе части этого уравнения на matrix_q ** (-1) слева
    # тгд: matrix_q ** (-1) * matrix_a = matrix_q ** (-1) * matrix_q * matrix_r
    # учитывая, что: matrix_q ** (-1) * matrix_q = E, где E - единичная матрица => E * matrix_r = matrix_r
    # получаем: matrix_r = matrix_q ** (-1) * matrix_a
    # чтобы не искать обратную матрицу matrix_q ** (-1), можно воспользоваться св-вом ортогональной матрицы matrix_q:
    # matrix_q ** (-1) = matrix_q_transposed, где matrix_q_transposed - транспонированная матрица
    # тгд верхнетреугольную матрицу matrix_r получаем по ф-ле: matrix_r = matrix_q_transposed * matrix_a
    matrix_q_transposed = [[matrix_q[i][j] for i in range(n)] for j in range(n)]
    matrix_r = multiply_two_matrices(matrix_q_transposed, matrix_a)  # её размеры: n x n

    return matrix_q, matrix_r


def qr_decomposition_givens_turn(matrix_a):
    n = len(matrix_a)
    matrix_r = matrix_a.copy()
    g = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for j in range(n):
        for i in range(j + 1, n):
            a_j, a_i = matrix_r[j][j], matrix_r[i][j]
            c = a_j / ((a_j ** 2 + a_i ** 2) ** 0.5)
            s = (-a_i) / ((a_j ** 2 + a_i ** 2) ** 0.5)
            g[j][j], g[i][i] = c, c
            g[i][j], g[j][i] = s, -s
            matrix_r = multiply_two_matrices(g, matrix_r)
            g[j][j], g[i][i] = 1, 1
            g[i][j], g[j][i] = 0, 0

    # найдем ортогональную матрицу matrix_q из выражения matrix_a = matrix_q * matrix_r
    # умножим обе части этого уравнения на matrix_r ** (-1) справа
    # тгд: matrix_a * matrix_r ** (-1) = matrix_q * matrix_r * matrix_r ** (-1)
    # учитывая, что: matrix_r * matrix_r ** (-1) = E, где E - единичная матрица => matrix_q * E = matrix_q
    # получаем: matrix_q = matrix_a * matrix_r ** (-1), тгд требуется найти обратную матрицу matrix_r ** (-1)
    return matrix_r


def qr_algorithm(matrix_a, eps):
    """Функция осуществляет QR-алгоритм"""

    n = len(matrix_a)
    iters = 1000  # ? подобрать кол-во итераций, оценив сложность алгоритма
    for iter in range(iters):
        matrix_q, matrix_r = qr_decomposition_gram_schmidt_process(matrix_a)
        matrix_r_extra = qr_decomposition_givens_turn(matrix_a)
        print("iter №", iter + 1)
        print("R - gram:")
        for elem in matrix_r: print(elem)
        print("R - givens:")
        for elem in matrix_r_extra: print(elem)
        print()
        matrix_a = multiply_two_matrices(matrix_r, matrix_q)
        if all(abs(matrix_a[i][j]) < eps for j in range(n) for i in range(j + 1, n)):
            print("Всего было итераций:", iter + 1)
            break
    print("A:")
    for elem in matrix_a:
        print(*elem)


a = [[5, 2, -3], [4, 5, -4], [6, 4, -4]]  # l = 1, l = 2, l = 3
eps = 1 / 10 ** 15
qr_algorithm(a, eps)
