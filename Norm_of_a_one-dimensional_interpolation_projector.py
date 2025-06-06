import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Генерация равномерных узлов
def uniform_nodes(n):
    return np.linspace(-1, 1, n + 1)

# Генерация узлов Чебышева
def chebyshev_nodes(n):
    k = np.arange(0, n + 1)
    return np.cos((2 * k + 1) * np.pi / (2 * (n + 1)))

# Вычисление базисного многочлена Лагранжа l_j(x)
def lagrange_basis(x, nodes, j):
    xj = nodes[j]
    lj = np.ones_like(x)
    for m in range(len(nodes)):
        if m != j:
            lj *= (x - nodes[m]) / (xj - nodes[m])
    return lj

# Норма интерполяционного проектора ||P|| = max_{x in [-1,1]} sum_j |l_j(x)|
def projector_norm(nodes, x_eval):
    n = len(nodes) - 1
    L = np.zeros_like(x_eval)
    for j in range(n + 1):
        L += np.abs(lagrange_basis(x_eval, nodes, j))
    return np.max(L)

# Главная функция вычислений
def compute_norms(max_n):
    x_eval = np.linspace(-1, 1, 1000)
    results = []

    for n in range(1, max_n + 1):
        un_nodes = uniform_nodes(n)
        ch_nodes = chebyshev_nodes(n)
        norm_uniform = projector_norm(un_nodes, x_eval)
        norm_chebyshev = projector_norm(ch_nodes, x_eval)
        results.append((n, norm_uniform, norm_chebyshev))

    df = pd.DataFrame(results, columns=["n", "Uniform Nodes", "Chebyshev Nodes"])
    return df

# Интерактивный ввод и запуск
if __name__ == "__main__":
    try:
        user_n = int(input("Введите максимальное значение степени интерполяционного многочлена n (например, 20): "))
        if user_n <= 0:
            raise ValueError("Число должно быть положительным.")
    except ValueError as e:
        print(f"Ошибка: {e}")
        exit(1)

    df = compute_norms(user_n)
    print("\nРезультаты вычислений:\n")
    print(df)

    # Сохранение результатов
    df.to_csv("projector_norms.csv", index=False)

    # Построение графика
    plt.plot(df["n"], df["Uniform Nodes"], label="Равномерные узлы", marker="o")
    plt.plot(df["n"], df["Chebyshev Nodes"], label="Узлы Чебышёва", marker="x")
    plt.xlabel("n (степень интерполяционного многочлена)")
    plt.ylabel("Норма проектора")
    plt.title("Сравнение норм интерполяционных проекторов")
    plt.legend()
    plt.grid(True)
    plt.show()
