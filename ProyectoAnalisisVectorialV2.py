# Librerias a importar para el uen funcionamiento del programa
import matplotlib.pyplot as plt 
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, sqrt, lambdify, sympify, integrate
from sympy import *

def ingresar_vector(n_componentes):
    if n_componentes == 2:
        x = float(input("Ingrese la componente x del vector: "))
        y = float(input("Ingrese la componente y del vector: "))
        return np.array([x, y])
    elif n_componentes == 3:
        x = float(input("Ingrese la componente x del vector: "))
        y = float(input("Ingrese la componente y del vector: "))
        z = float(input("Ingrese la componente z del vector: "))
        return np.array([x, y, z])

def ingresar_escalar():
    escalar = float(input("\nIngrese el valor del escalar: "))
    return escalar

def suma_vectores(vector1, vector2):
    resultado = vector1 + vector2
    return resultado

def resta_vectores(vector1, vector2, opcion):
    if opcion == 1:
        resultado = vector1 - vector2
    elif opcion == 2:
        resultado = vector2 - vector1
    else:
        print("Opción inválida")
        return None
    return resultado

def producto_escalar(vector1, vector2, escalar, opcion):
    if opcion == 1:
        resultado = vector1 * escalar
    elif opcion == 2:
        resultado = vector2 * escalar 
    else:
        print("Opción inválida")
        return None
    return resultado

def producto_punto(vector1, vector2):
    resultado = np.dot(vector1, vector2)
    return resultado

def producto_cruz(vector1, vector2):
    resultado = np.cross(vector1, vector2)
    return resultado

def magnitud_vector(vector):
    magnitud = np.linalg.norm(vector)
    return magnitud

def triple_producto_vectorial(vector1, vector2, vector3):
    producto_cruz_1 = np.cross(vector1, vector2)
    resultado = np.dot(producto_cruz_1, vector3)
    return resultado

def graficar_vectores(vectores):
    origen = [0, 0, 0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vector in vectores:
        ax.quiver(*origen, vector[0], vector[1], vector[2], arrow_length_ratio=0.1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    plt.show()
    
def modificar_vectores(vector1, vector2, vector3, n_componentes):
    print("1. Modificar vector 1")
    print("2. Modificar vector 2")
    print("3. Modificar vector 3")
    opcion = int(input("Seleccione el vector que desea modificar: "))

    if opcion == 1:
        vector1 = ingresar_vector(n_componentes)
    elif opcion == 2:
        vector2 = ingresar_vector(n_componentes)
    elif opcion == 3:
        vector3 = ingresar_vector(n_componentes)
    else:
        print("Opción inválida")

    return vector1, vector2, vector3

def recta_tangente():
    # Ingreso de valores
    v = input("Ingrese el vector inicial (separado por comas): ")
    p = input("Ingrese el punto de tangencia (separado por comas): ")

    # Conversión de los valores a listas de números
    v = [float(x) for x in v.split(",")]
    p = [float(x) for x in p.split(",")]

    # Cálculo de la recta tangente
    r = np.array(v)
    t = np.array(p)

    # Impresión del vector resultante
    print("El vector resultante es:", r + t)

    # Graficar la recta tangente
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Punto de tangencia
    ax.scatter(p[0], p[1], p[2], c='r', marker='o')

    # Recta tangente
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color='b')

    # Configuración de la gráfica
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar la gráfica
    plt.show()

def ecuacion_plano_tangente():
    # Ingreso de valores
    p1 = input("Ingrese el primer punto (separado por comas): ")
    p2 = input("Ingrese el segundo punto (separado por comas): ")
    p3 = input("Ingrese el tercer punto (separado por comas): ")

    # Conversión de los valores a listas de números
    p1 = [float(x) for x in p1.split(",")]
    p2 = [float(x) for x in p2.split(",")]
    p3 = [float(x) for x in p3.split(",")]

    # Cálculo del vector normal al plano
    v1 = np.array(p1)
    v2 = np.array(p2)
    v3 = np.array(p3)

    n = np.cross(v2 - v1, v3 - v1)

    # Ecuación del plano: ax + by + cz + d = 0
    a, b, c = n
    d = -np.dot(n, v1)

    # Impresión de la ecuación del plano
    print("La ecuación del plano tangente es: {}x + {}y + {}z + {} = 0".format(a, b, c, d))

    # Graficar los puntos y el plano
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Puntos
    points = np.array([p1, p2, p3])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    # Plano
    xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
    zz = (-a * xx - b * yy - d) / c
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Configuración de la gráfica
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar la gráfica
    plt.show()

def derivar_vector():
    # Solicitar las componentes del vector al usuario
    vector_str = input("Ingrese las componentes del vector separadas por comas (por ejemplo: 3*t, cos(t), sin(t)): ")

    # Separar las componentes del vector
    componentes = vector_str.split(',')

    # Crear una lista para almacenar las derivadas de las componentes
    derivadas = []

    # Variables simbólicas
    t = sp.symbols('t')

    # Derivar cada componente y almacenar los resultados
    for componente in componentes:
        derivada = sp.diff(componente.strip(), t)
        derivadas.append(derivada)

    # Imprimir las derivadas de las componentes
    print("Las derivadas de las componentes del vector son:")
    for i, derivada in enumerate(derivadas):
        print(f"Componente {i+1}: {derivada}")

    # Crear una función lambda para evaluar las derivadas
    vector_derivado = sp.lambdify(t, derivadas)

    # Solicitar el valor de t al usuario
    t_value = float(input("Ingrese el valor de t: "))

    # Evaluar las derivadas en el valor de t dado
    vector_evaluado = np.array(vector_derivado(t_value))

    # Crear una figura y un conjunto de ejes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Configurar los límites del gráfico
    ax.set_xlim([0, vector_evaluado[0]])
    ax.set_ylim([0, vector_evaluado[1]])
    ax.set_zlim([0, vector_evaluado[2]])

    # Graficar el vector resultante
    ax.quiver(0, 0, 0, vector_evaluado[0], vector_evaluado[1], vector_evaluado[2])

    # Configurar las etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar la gráfica
    plt.show()

def derivada_direccional():
    init_printing()

    # Definición de las variables simbólicas
    x, y, z = symbols('x y z')

    # Solicitar al usuario la función
    func_type = input("¿La función es de la forma f(x, y) o f(x, y, z)? (2D/3D): ").lower()

    if func_type == "2d":
        # Solicitar al usuario la función de la forma f(x, y)
        func_expr = input("Ingrese la función f(x, y): ")
        func = sympify(func_expr)
        vars = (x, y)
    elif func_type == "3d":
        # Solicitar al usuario la función de la forma f(x, y, z)
        func_expr = input("Ingrese la función f(x, y, z): ")
        func = sympify(func_expr)
        vars = (x, y, z)
    else:
        print("Opción inválida. Por favor, seleccione 2D o 3D.")
        exit()

    # Solicitar al usuario los puntos P y Q
    p_coords = []
    q_coords = []

    for var in vars:
        p_coord = float(input(f"Ingrese la coordenada {var} del punto P: "))
        p_coords.append(p_coord)
        q_coord = float(input(f"Ingrese la coordenada {var} del punto Q: "))
        q_coords.append(q_coord)

    # Construir los puntos P y Q
    point_p = Point(*p_coords)
    point_q = Point(*q_coords)

    # Calcular el vector entre los puntos P y Q
    vector_qp = point_q - point_p

    # Calcular la magnitud del vector de dirección
    direction_magnitude = sqrt(sum(coord**2 for coord in vector_qp))

    # Normalizar el vector de dirección y obtener el vector unitario
    unit_vector = [coord / direction_magnitude for coord in vector_qp]

    # Calcular la derivada direccional
    gradient = [diff(func, var) for var in vars]
    gradient_vector = Matrix(gradient)
    directional_derivative = gradient_vector.dot(unit_vector)

    # Imprimir el resultado
    print("La derivada direccional de la función en la dirección de P a Q es:")
    pprint(directional_derivative)

    # Crear un rango de valores para x e y (y z en caso de ser 3D)
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)

    # Crear una malla de coordenadas para evaluar la función
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = np.array([[func.subs([(vars[0], x_val), (vars[1], y_val)]) for y_val in y_vals] for x_val in x_vals])

    # Crear una figura y un conjunto de ejes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la función
    if len(vars) == 2:
        ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.5)
    else:
        ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.5)

    # Graficar los puntos P y Q en 3D
    #ax.scatter(p_coords[0], p_coords[1], p_coords[2], c='r', marker='o')
    #ax.scatter(q_coords[0], q_coords[1], q_coords[2], c='r', marker='o')
    ax.scatter(p_coords[0], p_coords[1], 0, c='r', marker='o')
    ax.scatter(q_coords[0], q_coords[1], 0, c='r', marker='o')

    # Graficar el vector entre los puntos P y Q en 3D
    #ax.quiver(p_coords[0], p_coords[1], p_coords[2], vector_qp[0], vector_qp[1], vector_qp[2], color='g')
    ax.quiver(p_coords[0], p_coords[1], 0, vector_qp[0], vector_qp[1], 0, color='g')


    # Configurar las etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar la gráfica
    plt.show()

def gradiente_vector():
    funcion = input("Introduce la función: ")
    variables = input("Introduce las variables separadas por comas (ejemplo: x, y, z): ")

    # Definir las variables simbólicas
    x, y, z = sp.symbols(variables)

    # Crear la función simbólica
    f = eval(funcion)

    # Calcular el gradiente
    gradiente = [sp.diff(f, var) for var in [x, y, z]]

    print("El gradiente es:")
    print(f"i: {gradiente[0]}")
    print(f"j: {gradiente[1]}")
    print(f"k: {gradiente[2]}")

    opcion = input("¿Deseas evaluar el resultado del gradiente? (si/no): ")

    if opcion.lower() == "si":
        x_val = float(input("Introduce el valor de x: "))
        y_val = float(input("Introduce el valor de y: "))
        z_val = float(input("Introduce el valor de z: "))

        # Definir las variables simbólicas localmente
        x, y, z = sp.symbols('x y z')

        resultado = [sp.N(grad.subs([(x, x_val), (y, y_val), (z, z_val)])) for grad in gradiente]

        print("El resultado del gradiente evaluado es:")
        print(f"i: {resultado[0]}")
        print(f"j: {resultado[1]}")
        print(f"k: {resultado[2]}")

        # Graficar el resultado
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, resultado[0], resultado[1], resultado[2], length=1)
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

def divergencia_vector():
    x, y, z = symbols('x y z')

    print("Ingresa la función a derivar:")
    funciones = []
    funciones.append(input("Ingrese la componente i: "))
    funciones.append(input("Ingrese la componente j: "))
    funciones.append(input("Ingrese la componente k: "))

    funciones_completas = ", ".join(funciones)
    print("Funcion Completa: ", funciones_completas)

    divergencia = [diff(funciones[0], x), diff(funciones[1], y), diff(funciones[2], z)]

    print("Divergencia:")
    print("d/dx:", divergencia[0], "i")
    print("d/dy:", divergencia[1], "j")
    print("d/dz:", divergencia[2], "k")

    print("Divergencia completa:")
    divergencia_completa = str(divergencia[0]) + "i + " + str(divergencia[1]) + "j + " + str(divergencia[2]) + "k"
    print(divergencia_completa)

    evaluar = input("¿Desea evaluar la divergencia? (s/n): ")

    if evaluar.lower() == "s":
        x_val = float(input("Ingrese el valor de x: "))
        y_val = float(input("Ingrese el valor de y: "))
        z_val = float(input("Ingrese el valor de z: "))

        resultado = sum([divergencia[0].subs([(x, x_val), (y, y_val), (z, z_val)]),
                        divergencia[1].subs([(x, x_val), (y, y_val), (z, z_val)]),
                        divergencia[2].subs([(x, x_val), (y, y_val), (z, z_val)])])

        print("Resultado de la evaluación:", resultado)

def rotacional_vector():
    # Pedir al usuario las componentes del vector
    component_i = input("Ingrese la componente i: ")
    component_j = input("Ingrese la componente j: ")
    component_k = input("Ingrese la componente k: ")

    # Convertir las componentes a expresiones simbólicas
    x, y, z = sp.symbols('x y z')
    component_i_expr = sp.sympify(component_i)
    component_j_expr = sp.sympify(component_j)
    component_k_expr = sp.sympify(component_k)

    # Imprimir las componentes ingresadas
    print("Componente i:", component_i_expr)
    print("Componente j:", component_j_expr)
    print("Componente k:", component_k_expr)

    # Calcular las derivadas parciales
    dVy_dx = sp.diff(component_i_expr, y)
    dVz_dx = sp.diff(component_i_expr, z)

    dVx_dy = sp.diff(component_j_expr, x)
    dVz_dy = sp.diff(component_j_expr, z)

    dVx_dz = sp.diff(component_k_expr, x)
    dVy_dz = sp.diff(component_k_expr, y)

    # Imprimir las derivadas parciales
    print("Derivada parcial de la componente i respecto a y:", dVy_dx)
    print("Derivada parcial de la componente i respecto a z:", dVz_dx)

    print("Derivada parcial de la componente j respecto a x:", dVx_dy)
    print("Derivada parcial de la componente j respecto a z:", dVz_dy)

    print("Derivada parcial de la componente k respecto a x:", dVx_dz)
    print("Derivada parcial de la componente k respecto a y:", dVy_dz)

    # Calcular el rotacional
    rotacional = sp.Matrix([(dVy_dz - dVz_dy), (dVz_dx - dVx_dz), (dVx_dy - dVy_dx)])

    # Imprimir el rotacional
    print("Rotacional:")
    sp.pprint(rotacional)

    # Preguntar al usuario si desea evaluar el rotacional
    evaluar_rotacional = input("¿Desea evaluar el rotacional en un punto? (s/n): ")

    if evaluar_rotacional.lower() == 's':
        # Pedir al usuario los valores de x, y, z
        x_val = float(input("Ingrese el valor de x: "))
        y_val = float(input("Ingrese el valor de y: "))
        z_val = float(input("Ingrese el valor de z: "))

        # Evaluar el rotacional en el punto dado
        rotacional_evaluado = rotacional.subs([(x, x_val), (y, y_val), (z, z_val)])
        print("Rotacional evaluado en el punto ({}, {}, {}):".format(x_val, y_val, z_val))
        sp.pprint(rotacional_evaluado)

        # Graficar el resultado del rotacional
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(0, 0, 0, rotacional_evaluado[0], rotacional_evaluado[1], rotacional_evaluado[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        print("No se evaluará el rotacional en un punto.")

def integral_vector():
    # Solicitar las componentes del vector al usuario
    vector_str = input("Ingrese las componentes del vector separadas por comas (por ejemplo: 3*t, cos(t), sin(t)): ")

    # Separar las componentes del vector
    componentes = vector_str.split(',')

    # Crear una lista para almacenar las integrales de las componentes
    integrales = []

    # Variables simbólicas
    t = sp.symbols('t')

    # Integrar cada componente y almacenar los resultados
    for componente in componentes:
        integral = sp.integrate(componente.strip(), t)
        integrales.append(integral)

    # Imprimir las integrales de las componentes
    print("Las integrales de las componentes del vector son:")
    for i, integral in enumerate(integrales):
        print(f"Componente {i+1}: {integral}")

    # Crear una función lambda para evaluar las integrales
    vector_integrado = sp.lambdify(t, integrales)

    # Solicitar el valor de t al usuario
    t_value = float(input("Ingrese el valor de t: "))

    # Evaluar las integrales en el valor de t dado
    vector_evaluado = np.array(vector_integrado(t_value))

    # Crear una figura y un conjunto de ejes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Configurar los límites del gráfico
    ax.set_xlim([0, vector_evaluado[0]])
    ax.set_ylim([0, vector_evaluado[1]])
    ax.set_zlim([0, vector_evaluado[2]])

    # Graficar el vector resultante
    ax.quiver(0, 0, 0, vector_evaluado[0], vector_evaluado[1], vector_evaluado[2])

    # Configurar las etiquetas de los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Mostrar la gráfica
    plt.show()

def integral_linea_vector():
    # Pedir la función en términos de x, y, z
    f_x = sympify(input("Ingresa la componente x de la función: "))
    f_y = sympify(input("Ingresa la componente y de la función: "))
    f_z = sympify(input("Ingresa la componente z de la función: "))

    # Pedir los puntos a y b
    a_x = float(input("Ingresa la coordenada x del punto a: "))
    a_y = float(input("Ingresa la coordenada y del punto a: "))
    a_z = float(input("Ingresa la coordenada z del punto a: "))

    b_x = float(input("Ingresa la coordenada x del punto b: "))
    b_y = float(input("Ingresa la coordenada y del punto b: "))
    b_z = float(input("Ingresa la coordenada z del punto b: "))

    # Definir las variables simbólicas
    t = symbols('t')

    # Obtener la parametrización del vector entre los puntos a y b
    x = a_x + (b_x - a_x) * t
    y = a_y + (b_y - a_y) * t
    z = a_z + (b_z - a_z) * t

    # Calcular la derivada de la parametrización con respecto a t
    dx_dt = diff(x, t)
    dy_dt = diff(y, t)
    dz_dt = diff(z, t)

    # Convertir las funciones a funciones numéricas
    f_x_lambda = lambdify(t, f_x)
    f_y_lambda = lambdify(t, f_y)
    f_z_lambda = lambdify(t, f_z)
    dx_dt_lambda = lambdify(t, dx_dt)
    dy_dt_lambda = lambdify(t, dy_dt)
    dz_dt_lambda = lambdify(t, dz_dt)

    # Definir la función vectorial
    vector_func = lambda t: [f_x_lambda(t), f_y_lambda(t), f_z_lambda(t)]

    # Definir la función integrando
    integrand_func = lambda t: np.dot(vector_func(t), [dx_dt_lambda(t), dy_dt_lambda(t), dz_dt_lambda(t)])

    # Realizar la integración numérica usando cuadratura gaussiana
    from scipy.integrate import fixed_quad
    resultado, _ = fixed_quad(integrand_func, 0, 1, n=5)

    # Imprimir el resultado
    print("El resultado de la integral de línea del campo vectorial es:", resultado)

    # Graficar la función, los puntos a y b, y la parametrización r(t)
    t_vals = np.linspace(0, 1, 100)
    x_vals = np.array([x.subs(t, val) for val in t_vals], dtype=float)
    y_vals = np.array([y.subs(t, val) for val in t_vals], dtype=float)
    z_vals = np.array([z.subs(t, val) for val in t_vals], dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la función
    ax.plot(x_vals, y_vals, z_vals, label='Función')

    # Graficar los puntos a y b
    ax.plot([a_x, b_x], [a_y, b_y], [a_z, b_z], 'ro', label='Puntos')

    # Graficar la parametrización r(t)
    ax.plot(x_vals, y_vals, z_vals, label='Parametrización')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend()

    plt.show()

def integral_superficie_vector():
    # Definición del campo vectorial F(x, y, z)
    def campo_vectorial(x, y, z):
        F_x = x**2
        F_y = y**2
        F_z = z**2
        
        return F_x, F_y, F_z

    # Definición de la superficie S(x, y, z)
    def superficie(x, y, z):
        ecuacion = x**2 + y**2 + z**2 - 1
        
        return ecuacion

    # Obtención de los rangos de integración del usuario
    inicio_x = float(input("Ingrese el valor inicial de x: "))
    fin_x = float(input("Ingrese el valor final de x: "))
    inicio_y = float(input("Ingrese el valor inicial de y: "))
    fin_y = float(input("Ingrese el valor final de y: "))
    inicio_z = float(input("Ingrese el valor inicial de z: "))
    fin_z = float(input("Ingrese el valor final de z: "))

    # Definición de los pasos de integración
    n_pasos = 100
    rango_x = np.linspace(inicio_x, fin_x, n_pasos)
    rango_y = np.linspace(inicio_y, fin_y, n_pasos)
    rango_z = np.linspace(inicio_z, fin_z, n_pasos)

    # Cálculo de la integral de superficie
    def integral_superficie(campo_vectorial, superficie, rango_x, rango_y, rango_z):
        x, y, z = np.meshgrid(rango_x, rango_y, rango_z)
        valores_superficie = superficie(x, y, z)
        dS = np.abs(np.gradient(valores_superficie))
        F_x, F_y, F_z = campo_vectorial(x, y, z)
        integral = np.sum(F_x * dS[0]) + np.sum(F_y * dS[1]) + np.sum(F_z * dS[2])
        
        return integral

    # Cálculo de la integral de superficie
    resultado = integral_superficie(campo_vectorial, superficie, rango_x, rango_y, rango_z)
    print("El resultado de la integral de superficie es:", resultado)

    # Generar mallas en 3D
    x, y, z = np.meshgrid(rango_x, rango_y, rango_z)

    # Evaluar la superficie y el campo vectorial
    valores_superficie = superficie(x, y, z)
    F_x, F_y, F_z = campo_vectorial(x, y, z)

    # Graficar la superficie en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x[:,:,0], y[:,:,0], z[:,:,0], facecolors=plt.cm.viridis(valores_superficie[:,:,0]), alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Superficie')
    plt.show()

    # Graficar el campo vectorial en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, F_x, F_y, F_z, length=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Campo Vectorial')
    plt.show()

def integral_volumen_vector():
	# Solicitar al usuario ingresar las funciones del campo vectorial
	fx = sp.sympify(input("Ingrese la función Fx(x, y, z): "))
	fy = sp.sympify(input("Ingrese la función Fy(x, y, z): "))
	fz = sp.sympify(input("Ingrese la función Fz(x, y, z): "))

	# Definir las variables simbólicas
	x, y, z = sp.symbols('x y z')

	# Solicitar al usuario ingresar los límites de integración
	a = float(input("Ingrese el límite inferior para x: "))
	b = float(input("Ingrese el límite superior para x: "))
	c = float(input("Ingrese el límite inferior para y: "))
	d = float(input("Ingrese el límite superior para y: "))
	e = float(input("Ingrese el límite inferior para z: "))
	f = float(input("Ingrese el límite superior para z: "))

	# Calcular la integral de volumen utilizando la función integrate de sympy
	integral = sp.integrate(sp.integrate(sp.integrate(fz, (x, a, b)), (y, c, d)), (z, e, f))

	# Convertir las funciones del campo vectorial en funciones numéricas
	fxf = sp.lambdify((x, y, z), fx, 'numpy')
	fyf = sp.lambdify((x, y, z), fy, 'numpy')
	fzf = sp.lambdify((x, y, z), fz, 'numpy')

	# Generar puntos en el espacio tridimensional
	x_vals = np.linspace(a, b, 10)
	y_vals = np.linspace(c, d, 10)
	z_vals = np.linspace(e, f, 10)
	X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

	# Evaluar el campo vectorial en los puntos
	U = fxf(X, Y, Z)
	V = fyf(X, Y, Z)
	W = fzf(X, Y, Z)

	# Graficar el campo vectorial
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.quiver(X, Y, Z, U, V, W)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Gráfico vectorial')
	plt.show()

	# Generar puntos en el plano xy para el gráfico de contorno
	xy_vals = np.linspace(a, b, 50)
	X_contour, Y_contour = np.meshgrid(xy_vals, xy_vals)

	# Evaluar el campo vectorial en los puntos del plano xy
	U_contour = fxf(X_contour, Y_contour, np.zeros_like(X_contour))
	V_contour = fyf(X_contour, Y_contour, np.zeros_like(X_contour))
	W_contour = fzf(X_contour, Y_contour, np.zeros_like(X_contour))

	# Calcular la magnitud del campo vectorial en los puntos del plano xy
	magnitude_contour = np.sqrt(U_contour**2 + V_contour**2 + W_contour**2)

	# Graficar el gráfico de contorno
	plt.figure()
	plt.contourf(X_contour, Y_contour, magnitude_contour)
	plt.colorbar()
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Gráfico de contorno')
	plt.show()

	# Generar puntos en el plano xz para el gráfico de superficie
	xz_vals = np.linspace(a, b, 50)
	X_surface, Z_surface = np.meshgrid(xz_vals, xz_vals)

	# Evaluar el campo vectorial en los puntos del plano xz
	U_surface = fxf(X_surface, np.zeros_like(X_surface), Z_surface)
	V_surface = fyf(X_surface, np.zeros_like(X_surface), Z_surface)
	W_surface = fzf(X_surface, np.zeros_like(X_surface), Z_surface)

	# Calcular la magnitud del campo vectorial en los puntos del plano xz
	magnitude_surface = np.sqrt(U_surface**2 + V_surface**2 + W_surface**2)

	# Graficar el gráfico de superficie
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X_surface, np.zeros_like(X_surface), Z_surface, facecolors=plt.cm.viridis(magnitude_surface/magnitude_surface.max()))
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('Gráfico de superficie')
	plt.show()

	# Mostrar el resultado de la integral
	print("La integral de volumen del campo vectorial F(x, y, z) =", integral)

def teorema_divergencia_gauss():
    # Función para calcular la divergencia de un campo vectorial en un punto específico
    def calcular_divergencia(F):
        # Implementa la fórmula para calcular la divergencia
        div = np.gradient(F[0])[0] + np.gradient(F[1])[1] + np.gradient(F[2])[2]
        return div

    # Función para calcular el flujo del campo a través de una superficie cerrada
    def calcular_flujo(F, dA):
        # Calcula el producto escalar entre el campo vectorial y los elementos diferenciales de área
        producto_escalar = np.sum(F * dA, axis=1)
        # Calcula la integral del producto escalar sumando todos los elementos
        flujo = np.sum(producto_escalar)
        return flujo

    # Solicitar al usuario ingresar las variables necesarias
    num_puntos = int(input("Ingrese el número de puntos en la superficie: "))

    # Crear arreglos para almacenar las componentes del campo vectorial y los elementos diferenciales de área
    F = np.zeros((num_puntos, 3))
    dA = np.zeros((num_puntos, 3))

    # Solicitar al usuario ingresar las componentes del campo vectorial y los elementos diferenciales de área
    for i in range(num_puntos):
        print(f"\nPunto {i+1}:")
        F[i] = np.array([float(input("Ingrese la componente x del campo vectorial: ")),
                        float(input("Ingrese la componente y del campo vectorial: ")),
                        float(input("Ingrese la componente z del campo vectorial: "))])
        dA[i] = np.array([float(input("Ingrese el elemento diferencial de área en la dirección x: ")),
                         float(input("Ingrese el elemento diferencial de área en la dirección y: ")),
                         float(input("Ingrese el elemento diferencial de área en la dirección z: "))])

    # Calcular la divergencia del campo vectorial en el punto dado
    divergencia = calcular_divergencia(F)

    # Calcular el flujo del campo a través de la superficie cerrada
    flujo = calcular_flujo(F, dA)

    # Imprimir los resultados
    print("\n--- Resultados ---")
    print("Divergencia del campo: ", divergencia)
    print("Flujo del campo a través de la superficie: ", flujo)

    # Gráficos
    fig = plt.figure()

    # Gráfico 3D del campo vectorial
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.quiver(0, 0, 0, F[:, 0], F[:, 1], F[:, 2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Campo Vectorial')

    # Gráfico de barras de los elementos diferenciales de área
    ax2 = fig.add_subplot(132)
    ax2.bar(range(num_puntos), np.linalg.norm(dA, axis=1))
    ax2.set_xlabel('Puntos')
    ax2.set_ylabel('Magnitud')
    ax2.set_title('Elementos Diferenciales de Área')

    # Gráfico de dispersión de la magnitud del campo vectorial
    magnitudes = np.linalg.norm(F, axis=1)
    ax3 = fig.add_subplot(133)
    ax3.scatter(range(num_puntos), magnitudes)
    ax3.set_xlabel('Puntos')
    ax3.set_ylabel('Magnitud')
    ax3.set_title('Magnitud del Campo Vectorial')

    plt.tight_layout()
    plt.show()

def teorema_stokes():
    # Solicitar al usuario ingresar los puntos de la curva cerrada en el orden x, y, z
    print("Ingrese los puntos de la curva cerrada:")
    puntos = []
    while True:
        punto = input("Ingrese un punto (o 'q' para finalizar): ")
        if punto.lower() == 'q':
            break
        puntos.append([float(coord) for coord in punto.split(',')])

    # Solicitar al usuario ingresar las coordenadas del vector de campo F en el orden x, y, z
    f_x = float(input("Ingrese la componente x del vector de campo F: "))
    f_y = float(input("Ingrese la componente y del vector de campo F: "))
    f_z = float(input("Ingrese la componente z del vector de campo F: "))

    # Solicitar al usuario ingresar el área y la normal de la superficie
    area_s = float(input("Ingrese el área de la superficie: "))
    normal_s = [float(coord) for coord in input("Ingrese las componentes de la normal de la superficie (en el orden x, y, z): ").split(',')]

    # Crear los arreglos para graficar la curva cerrada y la superficie
    puntos_curva = np.array(puntos)
    puntos_curva = np.append(puntos_curva, [puntos[0]], axis=0)  # Agregar el primer punto al final para cerrar la curva
    x_curva = puntos_curva[:, 0]
    y_curva = puntos_curva[:, 1]
    z_curva = puntos_curva[:, 2]

    # Graficar la curva cerrada
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_curva, y_curva, z_curva)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Curva cerrada')

    # Graficar el campo vectorial F
    x_campo, y_campo, z_campo = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    u_campo = f_x * np.ones_like(x_campo)
    v_campo = f_y * np.ones_like(y_campo)
    w_campo = f_z * np.ones_like(z_campo)
    ax.quiver(x_campo, y_campo, z_campo, u_campo, v_campo, w_campo, length=0.1)

    plt.show()

    # Calcular la circulación a lo largo de la curva cerrada
    circulacion = 0
    for i in range(len(puntos)):
        x, y, z = puntos[i]
        if i < len(puntos) - 1:
            x_sig, y_sig, z_sig = puntos[i + 1]
        else:
            x_sig, y_sig, z_sig = puntos[0]
        circulacion += f_x * (y_sig - y) - f_y * (x_sig - x)

    # Calcular el flujo del rotacional a través de la superficie
    flujo_rotacional = np.dot([f_y, -f_x, 0], normal_s)

    # Imprimir los resultados
    print("La circulación a lo largo de la curva cerrada es:", circulacion)
    print("El flujo del rotacional a través de la superficie es:", flujo_rotacional)

    # Graficar la superficie
    if flujo_rotacional == 0:
        print("No se puede graficar la superficie ya que el flujo del rotacional es cero.")
    else:
        x_superficie = np.linspace(-1, 1, 100)
        y_superficie = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x_superficie, y_superficie)
        Z = np.zeros_like(X)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Superficie')

        # Graficar el campo vectorial F en la superficie
        u_superficie = f_x * np.ones_like(X)
        v_superficie = f_y * np.ones_like(Y)
        w_superficie = f_z * np.ones_like(Z)
        ax.quiver(X, Y, Z, u_superficie, v_superficie, w_superficie, length=0.1)

        # Graficar los puntos de la curva
        ax.scatter(x_curva, y_curva, z_curva, color='red')


        plt.show()

    # Comprobar si se cumple el teorema de Stokes
    if np.isclose(circulacion, flujo_rotacional):
        print("El teorema de Stokes se cumple.")
    else:
        print("El teorema de Stokes no se cumple.")

while True:
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Calculadora básica de vectores")
    print("2. Calculadora avanzada de vectores")
    print("3. Salir")
    opcion_principal = int(input("Seleccione una opción: "))

    if opcion_principal == 1:
        # Ingresar valores de los vectores y el escalar
        print("\n--- CALCULADORA BÁSICA DE VECTORES ---")
        
        n_componentes = int(input("Ingrese el número de componentes de los vectores (2 o 3): "))

        print("Ingrese los valores del primer vector:")
        vector1 = ingresar_vector(n_componentes)

        print("\nIngrese los valores del segundo vector:")
        vector2 = ingresar_vector(n_componentes)

        print("\nIngrese los valores del tercer vector:")
        vector3 = ingresar_vector(n_componentes)
        
        escalar = ingresar_escalar()

        # Menú de operaciones
        while True:
            print("\n--- MENÚ ---")
            print("1. Suma de vectores")
            print("2. Resta de vectores")
            print("3. Producto por un escalar")
            print("4. Producto punto")
            print("5. Producto cruz")
            print("6. Calcular magnitud de un vector")
            print("7. Triple producto escalar (AXB·C)")
            print("8. Modificar vectores")
            print("9. Volver al menú principal")
            opcion = int(input("Seleccione una opción: "))

            if opcion == 1:
                resultado = suma_vectores(vector1, vector2)
                print("Resultado de la suma:", resultado)
                graficar_vectores([vector1, vector2, resultado])
            elif opcion == 2:
                print("\n--- MENÚ DE RESTA ---")
                print("1. Resta vector1 - vector2")
                print("2. Resta vector2 - vector1")
                subopcion = int(input("Seleccione una opción: "))
                resultado = resta_vectores(vector1, vector2, subopcion)
                if resultado is not None:
                    print("Resultad3o de la resta:", resultado)
                    graficar_vectores([vector1, vector2, resultado])
            elif opcion == 3:
                print("\n--- MENÚ DE PRODUCTO POR ESCALAR ---")
                print("1. Producto vector1 * escalar")
                print("2. Producto vector2 * escalar")
                subopcion = int(input("Seleccione una opción: "))
                resultado = producto_escalar(vector1, escalar, subopcion)
                if subopcion == 1:
                    print("Resultado del producto por un escalar:", resultado)
                    graficar_vectores([vector1, resultado])
                elif subopcion == 2:
                    print("Resultado del producto por un escalar:", resultado)
                    graficar_vectores([vector2, resultado])
            elif opcion == 4:
                resultado = producto_punto(vector1, vector2)
                print("Resultado del producto punto:", resultado)
            elif opcion == 5:
                resultado = producto_cruz(vector1, vector2)
                print("Resultado del producto cruz:", resultado)
                graficar_vectores([vector1, vector2, resultado])
            elif opcion == 6:
                print("1. Vector 1")
                print("2. Vector 2")
                seleccion = int(input("Seleccione el vector del cual desea calcular la magnitud: "))
                if seleccion == 1:
                    magnitud = magnitud_vector(vector1)
                    print("Magnitud del vector 1:", magnitud)
                elif seleccion == 2:
                    magnitud = magnitud_vector(vector2)
                    print("Magnitud del vector 2:", magnitud)
                else:
                    print("Opción inválida")
            elif opcion == 7:
                resultado = triple_producto_vectorial(vector1, vector2, vector3)
                print("Resultado del triple producto vectorial:", resultado)
                graficar_vectores([vector1, vector2, vector3, np.cross(vector1,vector2)])
            elif opcion == 8:
                vector1, vector2, vector3 = modificar_vectores(vector1, vector2, vector3, n_componentes)
            elif opcion == 9:
                break
            else:
                print("Opción inválida. Intente nuevamente.")

    elif opcion_principal == 2:
        print("\n--- CALCULADORA AVANZADA DE VECTORES ---")

        while True:
            print("\n--- MENÚ ---")
            print("1. Recta tangente")
            print("2. Plano tangente")
            print("3. Derivada de un vector")
            print("4. Derivada direccional de un vector")
            print("5. Gradiente de un vector")
            print("6. Divergencia de un vector")
            print("7. Rotacional de un vector")
            print("8. Integral de un vector")
            print("9. Integral de línea de un vector")
            print("10. Integral de superficie de un vector")
            print("11. Integral de volumen de un vector")
            print("12. Teorema de la divergencia de Gauss")
            print("13. Teorema de Stokes")
            print("14. Teorema de Green en el plano")
            print("15. Volver al menú principal")
            opcion = int(input("Seleccione una opción: "))

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
                teorema_green()
            elif opcion == 15:
                break
            else:
                print("Opción inválida.")

    elif opcion_principal == 3:
        print("¡Hasta luego!")
        break
    else:
        print("Opción inválida.")
