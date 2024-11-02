import math
import numpy as np  # нужно для вычисления обратной матрицы
import prettytable
'''
ДОДЕЛАТЬ:
2) написать комментарии для всего кода, где они не написаны
3) написать всю обёртку
'''


def sign(x):
    """
    Вспомогательная функция: математическая функция sign(x).
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def multiply_two_matrices(matrix_a, matrix_b):
    """
    Вспомогательная функция: умножает матрицы matrix_a и matrix_b друг на друга и возвращает итоговую матрицу matrix_c.
    Предполагается, что умножение матриц определено: кол-во столбцов matrix_a = кол-ву строк matrix_b.
    """
    # размеры матриц matrix_a и matrix_b: кол-во строк и столбцов соответственно
    m_a, n_a, m_b, n_b = len(matrix_a), len(matrix_a[0]), len(matrix_b), len(matrix_b[0])  # предполагается n_a = m_b
    m_c, n_c = m_a, n_b  # размеры итоговой матрицы matrix_c: кол-во строк и столбцов соответственно
    matrix_c = []  # создадим матрицу matrix_c, заполненную нулями
    for i in range(m_c): matrix_c.append([0] * n_c)
    # при умножении двух матриц строки левой матрицы покоординатно умножаются на столбцы правой матрицы
    for i in range(m_c):
        for j in range(n_c):
            # элемент matrix_c[i][j] равен покоординатному умножению i-ой строки matrix_a на j-ый столбцец matrix_b
            matrix_c[i][j] = sum([matrix_a[i][k] * matrix_b[k][j] for k in range(n_a)])
    return matrix_c


def check_matrix_symmetry(matrix_a):
    """
    Вспомогательная функция: проверка матрицы на симметричность.
    """
    flag = True  # изначально считаем матрицу симметричной
    for i in range(n):
        for j in range(n):
            if matrix_a[i][j] != matrix_a[j][i]:  # если встретили несимметричную пару элементов, меняем флаг
                flag = False
                break
    return flag


def qr_decomposition_gram_schmidt_process(matrix_a):
    '''
    Функция осуществляет QR-разложение вещественной матрицы matrix_a на ортогональную матрицу matrix_q
    и верхнетреугольную матрицу matrix_r: matrix_a = matrix_q * matrix_r. Для реализации QR-разложения используется
    процесс Грама-Шмидта. Далее в комментариях будет описан ход этого процесса.
    '''
    n = len(matrix_a)  # размер матрицы, предполагается, что она квадратная
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
    matrix_q = [[normed_vectors[j][i] for j in range(n)] for i in range(n)]
    # найдем верхнетреугольную матрицу matrix_r из выражения matrix_a = matrix_q * matrix_r
    # умножим обе части этого уравнения на matrix_q ** (-1) слева
    # тгд: matrix_q ** (-1) * matrix_a = matrix_q ** (-1) * matrix_q * matrix_r
    # учитывая, что: matrix_q ** (-1) * matrix_q = E, где E - единичная матрица => E * matrix_r = matrix_r
    # получаем: matrix_r = matrix_q ** (-1) * matrix_a
    # чтобы не искать обратную матрицу matrix_q ** (-1), можно воспользоваться св-вом ортогональной матрицы matrix_q:
    # matrix_q ** (-1) = matrix_q_transposed, где matrix_q_transposed - транспонированная матрица
    # тгд верхнетреугольную матрицу matrix_r получаем по ф-ле: matrix_r = matrix_q_transposed * matrix_a
    matrix_q_transposed = [[matrix_q[i][j] for i in range(n)] for j in range(n)]
    matrix_r = multiply_two_matrices(matrix_q_transposed, matrix_a)
    return matrix_q, matrix_r


def qr_decomposition_givens_turn(matrix_a):  # !!!
    n = len(matrix_a)
    matrix_r = matrix_a.copy()
    g = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for j in range(n):
        for i in range(j + 1, n):
            a_j, a_i = matrix_r[j][j], matrix_r[i][j]
            c = a_j / ((a_j ** 2 + a_i ** 2) ** 0.5)
            s = (-a_i) / ((a_j ** 2 + a_i ** 2) ** 0.5)
            g[j][j] = g[i][i] = c
            g[i][j], g[j][i] = s, -s
            matrix_r = multiply_two_matrices(g, matrix_r)
            g[j][j] = g[i][i] = 1
            g[i][j] = g[j][i] = 0
    # найдем ортогональную матрицу matrix_q из выражения matrix_a = matrix_q * matrix_r
    # умножим обе части этого уравнения на matrix_r ** (-1) справа
    # тгд: matrix_a * matrix_r ** (-1) = matrix_q * matrix_r * matrix_r ** (-1)
    # учитывая, что: matrix_r * matrix_r ** (-1) = E, где E - единичная матрица => matrix_q * E = matrix_q
    # получаем: matrix_q = matrix_a * matrix_r ** (-1), тгд требуется найти обратную матрицу matrix_r ** (-1)
    matrix_r_inv = np.linalg.inv(np.array(matrix_r)).tolist()
    matrix_q = multiply_two_matrices(matrix_a, matrix_r_inv)
    return matrix_q, matrix_r


def qr_algorithm(matrix_a, eps, qr_decomposition):  # !!!
    """Функция осуществляет QR-алгоритм"""
    n = len(matrix_a)
    iters = 1000  # ? подобрать кол-во итераций, оценив сложность алгоритма
    for iter in range(iters):
        matrix_q, matrix_r = qr_decomposition(matrix_a)
        matrix_a = multiply_two_matrices(matrix_r, matrix_q)
        if max([abs(matrix_a[i][j]) for j in range(n) for i in range(j + 1, n)]) < eps:
            break
    eigenvalues = [matrix_a[i][i] for i in range(n)]
    return eigenvalues


def jakobi_rotation(matrix_a, eps):  # !!!
    n = len(matrix_a)
    while max([abs(matrix_a[i][j]) for i in range(n) for j in range(n) if i != j]) >= eps:
        for j in range(n - 1):
            for k in range(j + 1, n):
                if abs(matrix_a[j][k]) < eps:
                    continue
                # составим матрицу вращения matrix_j
                if matrix_a[j][j] == matrix_a[k][k]:
                    theta = math.pi / 4
                    c = math.cos(theta)
                    s = math.sin(theta)
                else:
                    tau = (matrix_a[j][j] - matrix_a[k][k]) / (2 * matrix_a[j][k])
                    t = sign(tau) / (abs(tau) + (1 + tau ** 2) ** 0.5)
                    c = 1 / ((1 + t ** 2) ** 0.5)
                    s = t * c
                matrix_j = [[1 if ii == jj else 0 for jj in range(n)] for ii in range(n)]
                matrix_j[j][j] = matrix_j[k][k] = c
                matrix_j[k][j], matrix_j[j][k] = s, -s
                matrix_j_transpored = [[matrix_j[ii][jj] for ii in range(n)] for jj in range(n)]
                matrix_a = multiply_two_matrices(matrix_j_transpored, matrix_a)
                matrix_a = multiply_two_matrices(matrix_a, matrix_j)
    eigenvalues = [matrix_a[i][i] for i in range(n)]
    return eigenvalues


def gauss_method(matrix_a, b):
    """
    Метод Гаусса для любых СЛАУ: несовместных, определённых, неопределённых. Для неопределённых СЛАУ некоторые
    произвольные неизвестные задаёт = 1 для красоты собственного вектора.
    """
    n = len(matrix_a)
    for i in range(n):  # объединяем матрицу matrix_a и вектор свободных членов b в расширенную матрицу
        matrix_a[i].append(b[i])
    # прямой ход: приводим расширенную матрицу matrix_a к верхнетреугольному виду
    for ii in range(n):  # будем обнулять неизвестную x_ii, для этого найдём строку, в к-рой x_ii != 0
        ind_row_x_ii = ii  # индекс строки, в которой x_ii != 0, изначально предполагаем, что такая x_ii в iiой строке
        for i in range(ii + 1, n):
            if abs(matrix_a[i][ii]) > abs(matrix_a[ind_row_x_ii][ii]):  # будем искать строку, в к-рой x_ii max по abs
                ind_row_x_ii = i
        # меняем строки с индексами ind_row_x_ii и ii, чтобы такая x_ii стояла в строке с индексом ii
        matrix_a[ii], matrix_a[ind_row_x_ii] = matrix_a[ind_row_x_ii], matrix_a[ii]
        elem_to_zero = matrix_a[ii][ii]  # обозначим такую x_ii за elem_to_zero
        if elem_to_zero == 0:  # если обнулять оказалось нечего, то пропускаем эту x_ii
            continue
        for i in range(ii + 1, n):  # для обнуления эл-та elem_to_zero в последующих строках нужно к этим строкам
            mult = -matrix_a[i][ii] / elem_to_zero  # прибавлять строку, в к-рой есть elem_to_zero, умноженную на mult
            for j in range(n + 1):
                matrix_a[i][j] += matrix_a[ii][j] * mult
    # определяем совместность или несовместность СЛАУ
    flag_matrix_a = "совместная"
    for i in range(n):
        if all(matrix_a[i][j] == 0 for j in range(n)) and matrix_a[i][n] != 0:
            flag_matrix_a = "несовместная"
            break
    if flag_matrix_a == "несовместная":
        print("СЛАУ не имеет решений")  # но такая ситуация невозможна для задачи, так как для
        # собственных значений всегда есть собственный вектор
        input()  # чтобы консоль не закрылась
        exit()  # завершение всей программы
    # если система совместна - её определённость или неопределённость
    flag_matrix_a = "определённая"
    for i in range(n):
        if all(matrix_a[i][j] == 0 for j in range(n)) and matrix_a[i][n] == 0:
            flag_matrix_a = "неопределённая"
            break
    vector_column_x = [None] * n
    if flag_matrix_a == "определённая":  # => СЛАУ имеет 1 реш. - начинаем обратный ход: вычисляем значения неизвестных
        for i in range(n - 1, -1, -1):
            mult = matrix_a[i][i]
            s = 0
            for j in range(i + 1, n):
                s += matrix_a[i][j] * vector_column_x[j]
            vector_column_x[i] = (matrix_a[i][n] - s) / mult
    else:  # flag_matrix_a == "неопределённая" => СЛАУ имеет бесконечно много решений - обратный ход реализован сложнее
        # тогда в матрице matrix_a некоторые неизвестные могут определяться однозначно, а некоторые - быть любыми
        for ii in range(n):  # нужно n итераций, чтобы определить n неизвестных
            ind_i = None
            for i in range(n):  # ищем строку, в которой можно определить одно неизвестное x
                cnt_x_can_define = sum([1 for j in range(n) if (matrix_a[i][j] != 0 and vector_column_x[j] == None)])
                if cnt_x_can_define == 1:
                    ind_i = i
                    break
            if cnt_x_can_define == 1:  # если нашли такую строку, определяем это неизвестное x однозначно
                s, ind_j = 0, None
                for j in range(n):
                    if matrix_a[ind_i][j] != 0:
                        if vector_column_x[j] == None:
                            ind_j = j
                            mult = matrix_a[ind_i][j]
                        else:
                            s += matrix_a[ind_i][j] * vector_column_x[j]
                vector_column_x[ind_j] = (matrix_a[ind_i][n] - s) / mult
            else:  # иначе придадим одному неизвестному x произвольное значение, пусть это будет 1 для красоты
                for j in range(n):
                    if vector_column_x[j] == None:
                        vector_column_x[j] = 1  # в след итерации цикла for ii возможно другие неизвестные снова
                        break  # будут определяться однозначно
    return vector_column_x


def main():
    # задаём в коде программы матрицу matrix_a, для которой будет решаться задача
    matrix_a = [
        [2, 1, 1],
        [1, 2, 1],
        [1, 1, 2]
    ]
    # вывод матрицы
    matrix_a_prettytable = prettytable.PrettyTable()
    matrix_a_prettytable.set_style(prettytable.PLAIN_COLUMNS)  # задаём стиль таблицы без границ полей
    matrix_a_prettytable.header = False  # убираем верхнюю заголовочную строку
    for row in matrix_a: matrix_a_prettytable.add_row(row)
    print("Исходная матрица:")
    print(matrix_a_prettytable)
    # проверка матрицы на возможность нахождения собственных значений
    if len(matrix_a) != len(matrix_a[0]):
        print("Невозможно решить задачу для данной матрицы:",
              "собственные числа существуют только у квадратных матриц.",
              "Измените матрицу в коде программы, если хотите решить задачу.", sep="\n")
        return
    # ввод точности от пользователя
    print("\nВведите точность eps от 10^-10 до 10^-1 в формате [0.0...0[ненулевое число]], например: eps = 0.001")
    # далее информация про min точность и время работы программы
    while True:
        try:
            eps = float(input("eps = "))
        except ValueError:
            print("Вы ввели не число - введите ещё раз")
        else:
            if not (10**(-10) <= eps <= 10**(-1)):
                print("Вы ввели число вне диапазона [10^-10; 10^-1] - введите ещё раз")
            elif str(eps).count('0') != (len(str(eps)) - 2):
                print("Вы ввели число не в формате [0.0...0[ненулевое число]] - введите ещё раз")
            else:
                break
    eps_signs = len(str(eps)) - 2  # столько знаков после точки надо будет оставлять при выводе ответов
    # выбор численного метода пользователем
    print("\nКакой численный метод применить для нахождения собственных чисел?",
    "1 - QR-алгоритм, 2 - Метод вращений Якоби", sep="\n")
    while True:
        try:
            choice_number_1 = int(input("Введите число 1 или 2: "))
        except ValueError:
            print("Вы ввели не число - введите ещё раз")
        else:
            if choice_number_1 != 1 and choice_number_1 != 2:
                print("Вы ввели число, отличное от 1 или 2 - введите ещё раз")
            else:
                break
    # проверка на невозможность применения метода вращений Якоби
    if choice_number_1 == 2 and not check_matrix_symmetry(matrix_a):
        print("Метод вращений Якоби применяется только для симметричных матриц.")
        answer = input("Применить QR-алгоритм? да/нет")
        if answer.lower() == "да":
            choice_number_1 = 1
        else:
            if answer.lower() == "нет":
                print("Программа завершена.",
                      "Измените матрицу в коде программы, если хотите применить метод вращений Якоби",
                      sep="\n")
            else:
                print("Ваш ответ был интерпретирован как отрицательный. Программа завершена."
                      "Измените матрицу в коде программы, если хотите применить метод вращений Якоби",
                      sep="\n")
            return
    # нахождение собственных значений
    if choice_number_1 == 1:
        print("\nКакой подход для QR-разложения применить?",
              "1 - Процесс Грама-Шмидта, 2 - Поворот Гивенса", sep="\n")
        while True:
            try:
                choice_number_2 = int(input("Введите число 1 или 2: "))
            except ValueError:
                print("Вы ввели не число - введите ещё раз")
            else:
                if choice_number_2 != 1 and choice_number_2 != 2:
                    print("Вы ввели число, отличное от 1 или 2 - введите ещё раз")
                else:
                    break
        if choice_number_2 == 1:
            eigenvalues = qr_algorithm(matrix_a, eps, qr_decomposition_gram_schmidt_process)
        else:
            eigenvalues = qr_algorithm(matrix_a, eps, qr_decomposition_givens_turn)
    else:
        eigenvalues = jakobi_rotation(matrix_a, eps)
    print()
    # вывод собственных значений
    for i in range(len(eigenvalues)):
        print(f"L{i + 1} = {eigenvalues[i]:.{eps_signs}f}")
    print()
    # нахождение собственных векторов
    print("Собственные векторы для собственных чисел:")
    for i in range(len(eigenvalues)):  # находим собственный вектор для каждого собственного значения
        e_val = eigenvalues[i]
        # ЗДЕСЬ ЧТО-ТО НЕ ТАК С КОПИЕЙ
        matrix_a_slau = matrix_a.copy()  # составляем матрицу для СЛАУ (A - L * E) * x = 0
        for ii in range(len(matrix_a_slau)):
            matrix_a_slau[ii][ii] -= round(e_val)
        matrix_b_slau = [0] * len(matrix_a_slau)
        eigenvectors = gauss_method(matrix_a_slau, matrix_b_slau)
        print(f"Для собственного значения L{i + 1} = {e_val:.{eps_signs}f} собственный вектор:")
        for e_vector in eigenvectors:
            print(f"{e_vector:.{eps_signs}f}")
        print()


main()
input()  # чтобы консоль не закрылась
