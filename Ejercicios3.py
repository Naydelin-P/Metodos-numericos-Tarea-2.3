#   Codigo que implementa el esquema numerico 
#   de interpolacion para determinar la raiz de
#   una ecuacion
# 
#           Autor:
#   Naydelin Palomo Martinez
#   naydelin.palomo.270602@gmiail.com
import numpy as np
import matplotlib.pyplot as plt

# Función original
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección con cálculo de errores
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")

    errores_abs = []
    errores_rel = []
    errores_cuad = []

    c_old = a  # Para calcular errores en la primera iteración
    for _ in range(max_iter):
        c = (a + b) / 2
        error_abs = abs(c - c_old)
        error_rel = error_abs / abs(c) if c != 0 else 0
        error_cuad = error_abs ** 2

        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)

        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, errores_abs, errores_rel, errores_cuad

        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

        c_old = c

    return (a + b) / 2, errores_abs, errores_rel, errores_cuad

# Selección de puntos adecuados dentro del intervalo  
x0 = -2.3
x1 = 1.7
x2 = 3.2
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(x0, x2, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar la raíz del polinomio interpolante usando bisección
root, errores_abs, errores_rel, errores_cuad = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), x0, x2)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="$f(x) = e^{-x}$ - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.4f}")

# Imprimir los errores
print("Errores en cada iteración del método de bisección:")
print("Iteración\tError Absoluto\tError Relativo\tError Cuadrático")
for i, (ea, er, ec) in enumerate(zip(errores_abs, errores_rel, errores_cuad)):
    print(f"{i+1}\t{ea:.6e}\t{er:.6e}\t{ec:.6e}")
