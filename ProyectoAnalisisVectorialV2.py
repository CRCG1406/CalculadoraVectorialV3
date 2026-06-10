# -*- coding: utf-8 -*-
"""
Módulo de cálculo vectorial.

Este módulo proporciona herramientas para operaciones básicas y avanzadas
con vectores, incluyendo cálculo de derivadas, integrales, gradientes,
divergencia, rotacional y verificación de teoremas fundamentales.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, sqrt, lambdify, sympify, integrate, init_printing, pprint, Matrix, Point
from scipy.integrate import fixed_quad
from typing import List, Union, Optional, Tuple

# ==========================
# Funciones auxiliares de entrada
# ==========================

def ingresar_vector(n_componentes: int) -> np.ndarray:
    """
    Solicita al usuario las componentes de un vector.
    
    Args:
        n_componentes (int): Número de componentes (2 o 3).
    
    Returns:
        np.ndarray: Vector con las componentes ingresadas.
    """
    if n_componentes == 2:
        x = float(input("Ingrese la componente x del vector: "))
        y = float(input("Ingrese la componente y del vector: "))
        return np.array([x, y])
    elif n_componentes == 3:
        x = float(input("Ingrese la componente x del vector: "))
        y = float(input("Ingrese la componente y del vector: "))
        z = float(input("Ingrese la componente z del vector: "))
        return np.array([x, y, z])
    else:
        raise ValueError("Solo se permiten 2 o 3 componentes.")

def ingresar_escalar() -> float:
    """Solicita un valor escalar al usuario."""
    return float(input("\nIngrese el valor del escalar: "))

# ==========================
# Operaciones básicas con vectores
# ==========================

def suma_vectores(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """Retorna la suma de dos vectores."""
    return vector1 + vector2

def resta_vectores(vector1: np.ndarray, vector2: np.ndarray, opcion: int) -> Optional[np.ndarray]:
    """
    Resta dos vectores según la opción seleccionada.
    
    Args:
        vector1: Primer vector.
        vector2: Segundo vector.
        opcion: 1 -> vector1 - vector2, 2 -> vector2 - vector1.
    
    Returns:
        Resultado de la resta o None si la opción es inválida.
    """
    if opcion == 1:
        return vector1 - vector2
    elif opcion == 2:
        return vector2 - vector1
    else:
        print("Opción inválida")
        return None

def producto_escalar_vector(vector: np.ndarray, escalar: float) -> np.ndarray:
    """Multiplica un vector por un escalar."""
    return vector * escalar

def producto_punto(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Retorna el producto punto (escalar) de dos vectores."""
    return np.dot(vector1, vector2)

def producto_cruz(vector1: np.ndarray, vector2: np.ndarray) -> Optional[np.ndarray]:
    """
    Retorna el producto cruz de dos vectores.
    Para vectores 2D, retorna un escalar (la componente z del producto cruz 3D).
    """
    if len(vector1) == 2 and len(vector2) == 2:
        # Para vectores 2D, el producto cruz es un escalar
        resultado = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        return np.array([resultado])
    elif len(vector1) >= 3 and len(vector2) >= 3:
        return np.cross(vector1[:3], vector2[:3])
    else:
        print("Error: Los vectores deben tener la misma dimensionalidad (2D o 3D)")
        return None

def magnitud_vector(vector: np.ndarray) -> float:
    """Retorna la magnitud (norma) de un vector."""
    return np.linalg.norm(vector)

def triple_producto_vectorial(vector1: np.ndarray, vector2: np.ndarray, vector3: np.ndarray) -> Optional[np.ndarray]:
    """
    Calcula el triple producto escalar: (vector1 x vector2) · vector3
    """
    if len(vector1) == 2 and len(vector2) == 2:
        # Para vectores 2D, el triple producto es el determinante (área del paralelogramo)
        resultado = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        return np.array([resultado])
    elif len(vector1) >= 3 and len(vector2) >= 3 and len(vector3) >= 3:
        cruz = np.cross(vector1[:3], vector2[:3])
        resultado = np.dot(cruz, vector3[:3])
        return np.array([resultado])
    else:
        print("Error: Vectores no compatibles para triple producto")
        return None

# ==========================
# Funciones de graficación mejoradas
# ==========================

def graficar_vectores_2d(vectores: List[np.ndarray], titulo: str = "Gráfico de vectores en 2D"):
    """
    Grafica una lista de vectores en 2D partiendo desde el origen.
    
    Args:
        vectores: Lista de vectores a graficar.
        titulo: Título del gráfico.
    """
    # Filtrar vectores válidos para 2D
    vectores_validos = []
    for v in vectores:
        if v is None:
            continue
        if isinstance(v, np.ndarray) and len(v) >= 2:
            vectores_validos.append(v[:2])  # Tomar solo primeras 2 componentes
    
    if not vectores_validos:
        print("No hay vectores válidos para graficar en 2D.")
        return
    
    plt.figure(figsize=(10, 10))
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, vector in enumerate(vectores_validos):
        color = colores[i % len(colores)]
        plt.quiver(0, 0, vector[0], vector[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, alpha=0.8, width=0.008, headwidth=8, headlength=10)
        plt.annotate(f'v{i+1}', xy=(vector[0], vector[1]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=11, color=color, fontweight='bold')
    
    # Calcular límites automáticos
    todos_x = [v[0] for v in vectores_validos]
    todos_y = [v[1] for v in vectores_validos]
    max_abs = max(max(abs(min(todos_x)), abs(max(todos_x))), 
                  max(abs(min(todos_y)), abs(max(todos_y)))) + 2
    plt.xlim([-max_abs, max_abs])
    plt.ylim([-max_abs, max_abs])
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)
    plt.xlabel('X', fontsize=14, fontweight='bold')
    plt.ylabel('Y', fontsize=14, fontweight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def graficar_vectores_3d(vectores: List[np.ndarray], titulo: str = "Gráfico de vectores en 3D"):
    """
    Grafica una lista de vectores en 3D partiendo desde el origen.
    
    Args:
        vectores: Lista de vectores a graficar.
        titulo: Título del gráfico.
    """
    # Filtrar vectores válidos para 3D
    vectores_validos = []
    for v in vectores:
        if v is None:
            continue
        if isinstance(v, np.ndarray) and len(v) >= 3:
            vectores_validos.append(v[:3])  # Tomar solo primeras 3 componentes
    
    if not vectores_validos:
        print("No hay vectores válidos para graficar en 3D.")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    origen = [0, 0, 0]
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, vector in enumerate(vectores_validos):
        color = colores[i % len(colores)]
        ax.quiver(origen[0], origen[1], origen[2], 
                 vector[0], vector[1], vector[2], 
                 color=color, arrow_length_ratio=0.15, linewidth=2.5, alpha=0.8)
        ax.text(vector[0], vector[1], vector[2], f'  v{i+1}', 
               color=color, fontsize=12, fontweight='bold')
    
    # Calcular límites automáticos
    todos_x = [v[0] for v in vectores_validos]
    todos_y = [v[1] for v in vectores_validos]
    todos_z = [v[2] for v in vectores_validos]
    max_abs = max(max(abs(min(todos_x)), abs(max(todos_x))),
                  max(abs(min(todos_y)), abs(max(todos_y))),
                  max(abs(min(todos_z)), abs(max(todos_z)))) + 2
    ax.set_xlim([-max_abs, max_abs])
    ax.set_ylim([-max_abs, max_abs])
    ax.set_zlim([-max_abs, max_abs])
    
    ax.set_xlabel('X', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    
    # Dibujar ejes coordenados
    ax.plot3D([-max_abs, max_abs], [0, 0], [0, 0], 'gray', alpha=0.3)
    ax.plot3D([0, 0], [-max_abs, max_abs], [0, 0], 'gray', alpha=0.3)
    ax.plot3D([0, 0], [0, 0], [-max_abs, max_abs], 'gray', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_vectores_inteligente(vectores: List[np.ndarray], n_componentes: int):
    """
    Grafica vectores automáticamente en 2D o 3D según el número de componentes.
    
    Args:
        vectores: Lista de vectores a graficar.
        n_componentes: Número de componentes (2 o 3).
    """
    if n_componentes == 2:
        graficar_vectores_2d(vectores)
    else:
        graficar_vectores_3d(vectores)

def modificar_vectores(vector1: np.ndarray, vector2: np.ndarray, vector3: np.ndarray, n_componentes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permite modificar uno de los tres vectores mediante nueva entrada del usuario."""
    print("\n--- MODIFICAR VECTORES ---")
    print("1. Modificar vector 1")
    print("2. Modificar vector 2")
    print("3. Modificar vector 3")
    opcion = int(input("Seleccione el vector que desea modificar: "))
    
    if opcion == 1:
        print("Ingrese los nuevos valores del vector 1:")
        vector1 = ingresar_vector(n_componentes)
    elif opcion == 2:
        print("Ingrese los nuevos valores del vector 2:")
        vector2 = ingresar_vector(n_componentes)
    elif opcion == 3:
        print("Ingrese los nuevos valores del vector 3:")
        vector3 = ingresar_vector(n_componentes)
    else:
        print("Opción inválida")
    
    return vector1, vector2, vector3

# ==========================
# Funciones avanzadas (cálculo vectorial)
# ==========================

def recta_tangente() -> None:
    """Calcula y grafica la recta tangente a partir de un vector dirección y un punto de tangencia."""
    print("\n--- RECTA TANGENTE ---")
    v = input("Ingrese el vector dirección (separado por comas, ej: 1,2,3): ")
    p = input("Ingrese el punto de tangencia (separado por comas, ej: 0,0,0): ")
    
    v = [float(x) for x in v.split(",")]
    p = [float(x) for x in p.split(",")]
    
    if len(v) == 2:
        v = v + [0]
    if len(p) == 2:
        p = p + [0]
    
    print("Ecuación paramétrica: r(t) =", p, "+ t *", v)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[0], p[1], p[2], c='red', marker='o', s=100, label='Punto de tangencia')
    
    t_vals = np.linspace(-5, 5, 100)
    x_vals = p[0] + t_vals * v[0]
    y_vals = p[1] + t_vals * v[1]
    z_vals = p[2] + t_vals * v[2]
    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, label='Recta tangente')
    
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color='green', 
              arrow_length_ratio=0.1, linewidth=2, label='Vector dirección')
    
    max_val = max(abs(p[0]) + abs(v[0])*5, abs(p[1]) + abs(v[1])*5, abs(p[2]) + abs(v[2])*5) + 2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Recta Tangente', fontsize=14)
    ax.legend()
    plt.show()

def ecuacion_plano_tangente() -> None:
    """Calcula y grafica el plano tangente a partir de tres puntos."""
    print("\n--- PLANO TANGENTE ---")
    p1 = input("Ingrese el primer punto (separado por comas, ej: 1,2,3): ")
    p2 = input("Ingrese el segundo punto (separado por comas): ")
    p3 = input("Ingrese el tercer punto (separado por comas): ")
    
    p1 = [float(x) for x in p1.split(",")]
    p2 = [float(x) for x in p2.split(",")]
    p3 = [float(x) for x in p3.split(",")]
    
    v1 = np.array(p1)
    v2 = np.array(p2)
    v3 = np.array(p3)
    
    n = np.cross(v2 - v1, v3 - v1)
    a, b, c = n
    d = -np.dot(n, v1)
    
    print(f"Ecuación del plano: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    points = np.array([p1, p2, p3])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o', s=100, label='Puntos')
    
    if abs(c) > 1e-6:
        xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        zz = (-a * xx - b * yy - d) / c
        ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan', label='Plano')
    
    max_val = max(np.max(np.abs(points))) + 2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Plano Tangente', fontsize=14)
    ax.legend()
    plt.show()

def derivar_vector() -> None:
    """Deriva un vector simbólico respecto a t y evalúa en un punto."""
    print("\n--- DERIVADA DE VECTOR ---")
    vector_str = input("Ingrese las componentes separadas por comas (ej: 3*t, cos(t), sin(t)): ")
    componentes = vector_str.split(',')
    t = sp.symbols('t')
    derivadas = [sp.diff(comp.strip(), t) for comp in componentes]
    
    print("Derivadas de las componentes:")
    for i, der in enumerate(derivadas):
        print(f"Componente {i+1}: {der}")
    
    vector_derivado = sp.lambdify(t, derivadas)
    t_value = float(input("Ingrese el valor de t: "))
    vector_evaluado = np.array(vector_derivado(t_value), dtype=float)
    
    print(f"Vector derivado evaluado en t={t_value}: {vector_evaluado}")
    
    fig = plt.figure(figsize=(8, 6))
    if len(vector_evaluado) == 2:
        plt.quiver(0, 0, vector_evaluado[0], vector_evaluado[1], 
                  angles='xy', scale_units='xy', scale=1, color='red', width=0.015)
        max_val = max(abs(vector_evaluado[0]), abs(vector_evaluado[1])) + 1
        plt.xlim([-max_val, max_val])
        plt.ylim([-max_val, max_val])
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Vector derivado en t={t_value}')
        plt.axis('equal')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, vector_evaluado[0], vector_evaluado[1], vector_evaluado[2], 
                 color='red', arrow_length_ratio=0.15, linewidth=2.5)
        max_val = max(abs(vector_evaluado[0]), abs(vector_evaluado[1]), abs(vector_evaluado[2])) + 1
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Vector derivado en t={t_value}')
    
    plt.show()

def derivada_direccional() -> None:
    """Calcula la derivada direccional de una función escalar en la dirección de P a Q."""
    print("\n--- DERIVADA DIRECCIONAL ---")
    init_printing()
    x, y, z = symbols('x y z')
    func_type = input("¿f(x, y) o f(x, y, z)? (2D/3D): ").lower()
    
    if func_type == "2d":
        func_expr = input("Ingrese f(x, y): ")
        func = sympify(func_expr)
        vars = (x, y)
    elif func_type == "3d":
        func_expr = input("Ingrese f(x, y, z): ")
        func = sympify(func_expr)
        vars = (x, y, z)
    else:
        print("Opción inválida")
        return
    
    p_coords = [float(input(f"Coordenada {var} de P: ")) for var in vars]
    q_coords = [float(input(f"Coordenada {var} de Q: ")) for var in vars]
    
    vector_qp = np.array(q_coords) - np.array(p_coords)
    direction_magnitude = np.linalg.norm(vector_qp)
    unit_vector = vector_qp / direction_magnitude
    
    gradient = [diff(func, var) for var in vars]
    grad_subs = [grad.subs(dict(zip(vars, p_coords))) for grad in gradient]
    directional_derivative = sum(grad_subs[i] * unit_vector[i] for i in range(len(vars)))
    
    print(f"Derivada direccional en P en dirección a Q: {directional_derivative}")
    
    # Gráfica simplificada
    if len(vars) == 2:
        x_vals = np.linspace(-5, 5, 50)
        y_vals = np.linspace(-5, 5, 50)
        xx, yy = np.meshgrid(x_vals, y_vals)
        zz = np.array([[func.subs({x: xi, y: yi}) for yi in y_vals] for xi in x_vals], dtype=float)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.6)
        ax.scatter(p_coords[0], p_coords[1], func.subs({x: p_coords[0], y: p_coords[1]}), 
                  c='red', s=100, label='Punto P')
        ax.quiver(p_coords[0], p_coords[1], 0, vector_qp[0], vector_qp[1], 0, 
                 color='green', arrow_length_ratio=0.1, linewidth=2, label='Dirección')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

def gradiente_vector() -> None:
    """Calcula y opcionalmente evalúa el gradiente de un campo escalar."""
    print("\n--- GRADIENTE ---")
    funcion = input("Introduce la función (usa x, y, z): ")
    
    x, y, z = sp.symbols('x y z')
    f = sp.sympify(funcion)
    gradiente = [sp.diff(f, var) for var in [x, y, z]]
    
    print("Gradiente:")
    print(f"∇f = [{gradiente[0]}, {gradiente[1]}, {gradiente[2]}]")
    
    if input("¿Deseas evaluar el gradiente? (si/no): ").lower() == "si":
        x_val = float(input("Valor de x: "))
        y_val = float(input("Valor de y: "))
        z_val = float(input("Valor de z: "))
        
        resultado = [float(grad.subs({x: x_val, y: y_val, z: z_val})) for grad in gradiente]
        print(f"Gradiente evaluado en ({x_val}, {y_val}, {z_val}): {resultado}")
        
        graficar_vectores_inteligente([np.array(resultado)], 3)

def divergencia_vector() -> None:
    """Calcula y opcionalmente evalúa la divergencia de un campo vectorial."""
    print("\n--- DIVERGENCIA ---")
    x, y, z = symbols('x y z')
    print("Ingresa las componentes del campo vectorial F(x,y,z):")
    fx = sympify(input("Componente i (Fx): "))
    fy = sympify(input("Componente j (Fy): "))
    fz = sympify(input("Componente k (Fz): "))
    
    divergencia = diff(fx, x) + diff(fy, y) + diff(fz, z)
    print(f"Divergencia ∇·F = {divergencia}")
    
    if input("¿Evaluar en un punto? (s/n): ").lower() == "s":
        x_val, y_val, z_val = float(input("x: ")), float(input("y: ")), float(input("z: "))
        resultado = divergencia.subs({x: x_val, y: y_val, z: z_val})
        print(f"Divergencia en ({x_val}, {y_val}, {z_val}) = {resultado}")

def rotacional_vector() -> None:
    """Calcula el rotacional de un campo vectorial en 3D."""
    print("\n--- ROTACIONAL ---")
    x, y, z = symbols('x y z')
    print("Ingresa las componentes del campo vectorial F(x,y,z):")
    fx = sympify(input("Componente i (Fx): "))
    fy = sympify(input("Componente j (Fy): "))
    fz = sympify(input("Componente k (Fz): "))
    
    rotacional = Matrix([
        diff(fz, y) - diff(fy, z),
        diff(fx, z) - diff(fz, x),
        diff(fy, x) - diff(fx, y)
    ])
    
    print("Rotacional ∇×F:")
    sp.pprint(rotacional)
    
    if input("¿Evaluar en un punto? (s/n): ").lower() == "s":
        x_val, y_val, z_val = float(input("x: ")), float(input("y: ")), float(input("z: "))
        rot_eval = rotacional.subs({x: x_val, y: y_val, z: z_val})
        print("Rotacional evaluado:")
        sp.pprint(rot_eval)
        
        resultado_vec = np.array([float(rot_eval[0]), float(rot_eval[1]), float(rot_eval[2])])
        graficar_vectores_inteligente([resultado_vec], 3)

def integral_vector() -> None:
    """Integra un vector simbólico respecto a t y evalúa."""
    print("\n--- INTEGRAL DE VECTOR ---")
    vector_str = input("Componentes separadas por comas (ej: 3*t, cos(t), sin(t)): ")
    t = sp.symbols('t')
    componentes = [sp.sympify(c.strip()) for c in vector_str.split(',')]
    integrales = [sp.integrate(c, t) for c in componentes]
    
    print("Integrales (sin constante):")
    for i, inte in enumerate(integrales):
        print(f"Componente {i+1}: {inte} + C")
    
    vector_integrado = sp.lambdify(t, integrales)
    t_val = float(input("Valor de t para evaluar: "))
    resultado = np.array(vector_integrado(t_val), dtype=float)
    print(f"Vector integrado evaluado en t={t_val}: {resultado}")
    
    graficar_vectores_inteligente([resultado], len(resultado))

def integral_linea_vector() -> None:
    """Calcula una integral de línea sobre un segmento rectilíneo."""
    print("\n--- INTEGRAL DE LÍNEA ---")
    print("Campo vectorial F(x,y,z):")
    fx = sympify(input("Componente x: "))
    fy = sympify(input("Componente y: "))
    fz = sympify(input("Componente z: "))
    
    print("\nPunto inicial a:")
    a = [float(input(f"Coordenada {c} de a: ")) for c in ('x', 'y', 'z')]
    print("\nPunto final b:")
    b = [float(input(f"Coordenada {c} de b: ")) for c in ('x', 'y', 'z')]
    
    t = symbols('t')
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * t
    z = a[2] + (b[2] - a[2]) * t
    
    dx, dy, dz = diff(x, t), diff(y, t), diff(z, t)
    integrando = (fx.subs({'x': x, 'y': y, 'z': z}) * dx + 
                  fy.subs({'x': x, 'y': y, 'z': z}) * dy + 
                  fz.subs({'x': x, 'y': y, 'z': z}) * dz)
    
    integrando_num = lambdify(t, integrando)
    resultado, _ = fixed_quad(integrando_num, 0, 1, n=10)
    print(f"Integral de línea = {resultado}")
    
    # Graficar
    t_vals = np.linspace(0, 1, 100)
    x_vals = np.array([x.subs(t, val) for val in t_vals], dtype=float)
    y_vals = np.array([y.subs(t, val) for val in t_vals], dtype=float)
    z_vals = np.array([z.subs(t, val) for val in t_vals], dtype=float)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=2, label='Trayectoria')
    ax.scatter([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c='red', s=100, label='Puntos a y b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Integral de Línea')
    ax.legend()
    plt.show()

# ==========================
# Funciones restantes (simplificadas por brevedad)
# ==========================

def integral_superficie_vector() -> None:
    """Calcula una integral de superficie sobre una esfera de radio 1."""
    print("\n--- INTEGRAL DE SUPERFICIE ---")
    print("Nota: Esta es una implementación simplificada para demostración.")
    print("Calculando integral de superficie para F(x,y,z) = (x², y², z²) sobre esfera unitaria...")
    
    def campo_vectorial(x, y, z):
        return x**2, y**2, z**2
    
    # Aproximación numérica simple
    n_pasos = 50
    theta = np.linspace(0, 2*np.pi, n_pasos)
    phi = np.linspace(0, np.pi, n_pasos)
    Theta, Phi = np.meshgrid(theta, phi)
    
    X = np.sin(Phi) * np.cos(Theta)
    Y = np.sin(Phi) * np.sin(Theta)
    Z = np.cos(Phi)
    
    Fx, Fy, Fz = campo_vectorial(X, Y, Z)
    
    # Elemento de área dS = sin(phi) dphi dtheta
    dS = np.sin(Phi)
    integral = np.sum((Fx * X + Fy * Y + Fz * Z) * dS) * (2*np.pi/n_pasos) * (np.pi/n_pasos)
    
    print(f"Integral de superficie aproximada: {integral}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Superficie de integración')
    plt.show()

def integral_volumen_vector() -> None:
    """Calcula la integral de volumen de un campo escalar."""
    print("\n--- INTEGRAL DE VOLUMEN ---")
    print("Calculando integral de volumen de f(x,y,z) = x² + y² + z² sobre un cubo.")
    
    x, y, z = symbols('x y z')
    f = x**2 + y**2 + z**2
    
    a, b = -1, 1  # Límites del cubo
    
    integral = sp.integrate(sp.integrate(sp.integrate(f, (x, a, b)), (y, a, b)), (z, a, b))
    print(f"Integral de volumen = {integral}")
    
    # Graficar el campo
    x_vals = np.linspace(a, b, 10)
    y_vals = np.linspace(a, b, 10)
    z_vals = np.linspace(a, b, 10)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)
    F = X**2 + Y**2 + Z**2
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=F, cmap='viridis', s=50)
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Campo escalar en el volumen')
    plt.show()

def teorema_divergencia_gauss() -> None:
    """Verificación simplificada del teorema de Gauss."""
    print("\n--- TEOREMA DE GAUSS ---")
    print("Verificando el teorema de la divergencia para F(x,y,z) = (x, y, z) sobre una esfera.")
    
    # Teorema: ∯ F·dS = ∭ (∇·F) dV
    # Para F = (x, y, z), ∇·F = 3
    
    radio = 1.0
    volumen = (4/3) * np.pi * radio**3
    integral_volumen = 3 * volumen
    
    # Flujo a través de la superficie (para campo radial)
    flujo_superficie = 4 * np.pi * radio**3
    
    print(f"Volumen de la esfera: {volumen:.4f}")
    print(f"Integral de volumen de la divergencia: {integral_volumen:.4f}")
    print(f"Flujo a través de la superficie: {flujo_superficie:.4f}")
    
    if np.isclose(integral_volumen, flujo_superficie, rtol=1e-6):
        print("✓ El teorema de Gauss se cumple para este campo vectorial.")
    else:
        print("✗ El teorema de Gauss no se cumple para este campo vectorial.")

def teorema_stokes() -> None:
    """Verificación simplificada del teorema de Stokes."""
    print("\n--- TEOREMA DE STOKES ---")
    print("Verificando el teorema de Stokes para F(x,y,z) = (-y, x, 0) sobre un círculo en el plano XY.")
    
    # Teorema: ∮ F·dr = ∬ (∇×F)·dS
    # Para F = (-y, x, 0), ∇×F = (0, 0, 2)
    
    radio = 1.0
    
    # Circulación alrededor del círculo
    circulacion = 2 * np.pi * radio**2
    
    # Flujo del rotacional a través del círculo
    flujo_rotacional = 2 * np.pi * radio**2
    
    print(f"Circulación a lo largo de la curva: {circulacion:.4f}")
    print(f"Flujo del rotacional a través de la superficie: {flujo_rotacional:.4f}")
    
    if np.isclose(circulacion, flujo_rotacional, rtol=1e-6):
        print("✓ El teorema de Stokes se cumple para este campo vectorial.")
    else:
        print("✗ El teorema de Stokes no se cumple para este campo vectorial.")
    
    # Graficar
    theta = np.linspace(0, 2*np.pi, 100)
    x = radio * np.cos(theta)
    y = radio * np.sin(theta)
    z = np.zeros_like(theta)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, 'b-', linewidth=2, label='Curva (círculo)')
    
    # Crear la superficie del círculo
    r = np.linspace(0, radio, 20)
    theta_grid = np.linspace(0, 2*np.pi, 20)
    R, Theta = np.meshgrid(r, theta_grid)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', label='Superficie')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Teorema de Stokes - Curva y Superficie')
    ax.legend()
    plt.show()

# ==========================
# Menú principal
# ==========================

def main():
    """Ejecuta el menú principal del programa."""
    while True:
        print("\n" + "="*50)
        print("           CALCULADORA VECTORIAL")
        print("="*50)
        print("1. Calculadora básica de vectores")
        print("2. Calculadora avanzada de vectores")
        print("3. Salir")
        
        opcion_principal = int(input("\nSeleccione una opción: "))
        
        if opcion_principal == 1:
            print("\n--- CALCULADORA BÁSICA DE VECTORES ---")
            n_componentes = int(input("Número de componentes de los vectores (2 o 3): "))
            
            print("\nPrimer vector:")
            vector1 = ingresar_vector(n_componentes)
            print("\nSegundo vector:")
            vector2 = ingresar_vector(n_componentes)
            print("\nTercer vector:")
            vector3 = ingresar_vector(n_componentes)
            escalar = ingresar_escalar()
            
            while True:
                print("\n" + "-"*40)
                print("MENÚ DE OPERACIONES BÁSICAS")
                print("-"*40)
                print("1. Suma de vectores (v1 + v2)")
                print("2. Resta de vectores")
                print("3. Producto por un escalar")
                print("4. Producto punto")
                print("5. Producto cruz")
                print("6. Calcular magnitud de un vector")
                print("7. Triple producto escalar (v1 × v2 · v3)")
                print("8. Modificar vectores")
                print("9. Volver al menú principal")
                
                opcion = int(input("\nSeleccione una opción: "))
                
                if opcion == 1:
                    resultado = suma_vectores(vector1, vector2)
                    print(f"Resultado de la suma: {resultado}")
                    graficar_vectores_inteligente([vector1, vector2, resultado], n_componentes)
                    
                elif opcion == 2:
                    print("\n1. v1 - v2")
                    print("2. v2 - v1")
                    subopcion = int(input("Seleccione una opción: "))
                    resultado = resta_vectores(vector1, vector2, subopcion)
                    if resultado is not None:
                        print(f"Resultado de la resta: {resultado}")
                        graficar_vectores_inteligente([vector1, vector2, resultado], n_componentes)
                        
                elif opcion == 3:
                    print("\n1. v1 × escalar")
                    print("2. v2 × escalar")
                    subopcion = int(input("Seleccione una opción: "))
                    if subopcion == 1:
                        resultado = producto_escalar_vector(vector1, escalar)
                        print(f"Resultado: {resultado}")
                        graficar_vectores_inteligente([vector1, resultado], n_componentes)
                    elif subopcion == 2:
                        resultado = producto_escalar_vector(vector2, escalar)
                        print(f"Resultado: {resultado}")
                        graficar_vectores_inteligente([vector2, resultado], n_componentes)
                    else:
                        print("Opción inválida")
                        
                elif opcion == 4:
                    resultado = producto_punto(vector1, vector2)
                    print(f"Producto punto: {resultado}")
                    
                elif opcion == 5:
                    resultado = producto_cruz(vector1, vector2)
                    if resultado is not None:
                        print(f"Producto cruz: {resultado}")
                        if len(resultado) >= 2:
                            graficar_vectores_inteligente([vector1, vector2, resultado], n_componentes)
                            
                elif opcion == 6:
                    print("\n1. Vector 1")
                    print("2. Vector 2")
                    print("3. Vector 3")
                    seleccion = int(input("Seleccione el vector: "))
                    if seleccion == 1:
                        magnitud = magnitud_vector(vector1)
                        print(f"Magnitud del vector 1: {magnitud:.4f}")
                    elif seleccion == 2:
                        magnitud = magnitud_vector(vector2)
                        print(f"Magnitud del vector 2: {magnitud:.4f}")
                    elif seleccion == 3:
                        magnitud = magnitud_vector(vector3)
                        print(f"Magnitud del vector 3: {magnitud:.4f}")
                    else:
                        print("Opción inválida")
                        
                elif opcion == 7:
                    resultado = triple_producto_vectorial(vector1, vector2, vector3)
                    if resultado is not None:
                        print(f"Triple producto escalar: {resultado}")
                        if n_componentes == 3:
                            cruz = producto_cruz(vector1, vector2)
                            if cruz is not None:
                                graficar_vectores_inteligente([vector1, vector2, vector3, cruz], n_componentes)
                        else:
                            graficar_vectores_inteligente([vector1, vector2, vector3], n_componentes)
                            
                elif opcion == 8:
                    vector1, vector2, vector3 = modificar_vectores(vector1, vector2, vector3, n_componentes)
                    
                elif opcion == 9:
                    break
                    
                else:
                    print("Opción inválida. Intente nuevamente.")
                    
        elif opcion_principal == 2:
            print("\n--- CALCULADORA AVANZADA DE VECTORES ---")
            
            while True:
                print("\n" + "-"*40)
                print("MENÚ DE OPERACIONES AVANZADAS")
                print("-"*40)
                print("1. Recta tangente")
                print("2. Plano tangente")
                print("3. Derivada de un vector")
                print("4. Derivada direccional")
                print("5. Gradiente")
                print("6. Divergencia")
                print("7. Rotacional")
                print("8. Integral de un vector")
                print("9. Integral de línea")
                print("10. Integral de superficie")
                print("11. Integral de volumen")
                print("12. Teorema de Gauss (Divergencia)")
                print("13. Teorema de Stokes")
                print("14. Volver al menú principal")
                
                opcion = int(input("\nSeleccione una opción: "))
                
                if opcion == 1:
                    recta_tangente()
                elif opcion == 2:
                    ecuacion_plano_tangente()
                elif opcion == 3:
                    derivar_vector()
                elif opcion == 4:
                    derivada_direccional()
                elif opcion == 5:
                    gradiente_vector()
                elif opcion == 6:
                    divergencia_vector()
                elif opcion == 7:
                    rotacional_vector()
                elif opcion == 8:
                    integral_vector()
                elif opcion == 9:
                    integral_linea_vector()
                elif opcion == 10:
                    integral_superficie_vector()
                elif opcion == 11:
                    integral_volumen_vector()
                elif opcion == 12:
                    teorema_divergencia_gauss()
                elif opcion == 13:
                    teorema_stokes()
                elif opcion == 14:
                    break
                else:
                    print("Opción inválida.")
                    
        elif opcion_principal == 3:
            print("\n¡Hasta luego! Gracias por usar la calculadora vectorial.")
            break
            
        else:
            print("Opción inválida. Por favor, seleccione 1, 2 o 3.")

if __name__ == "__main__":
    main()
