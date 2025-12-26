# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np


# ####################################################################
def eliminacion_gaussiana(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser n-by-(n+1)."
    n = A.shape[0]

    # Contadores
    sumas = 0
    restas = 0
    multiplicaciones = 0
    divisiones = 0
    intercambios = 0

    # Eliminación hacia adelante
    for i in range(n - 1):

        # búsqueda de pivote
        p = i
        while p < n and A[p, i] == 0:
            p += 1

        if p == n:
            raise ValueError("No existe solución única.")

        if p != i:
            A[[i, p]] = A[[p, i]]
            intercambios += 1

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            divisiones += 1

            for k in range(i, n + 1):
                A[j, k] = A[j, k] - m * A[i, k]
                multiplicaciones += 1
                restas += 1

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    # Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    divisiones += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma = suma + A[i, j] * solucion[j]
            multiplicaciones += 1
            sumas += 1

        solucion[i] = (A[i, n] - suma) / A[i, i]
        restas += 1
        divisiones += 1

    # Salida EXACTA como en la imagen
    print("Solución:", solucion.astype(int) if np.allclose(solucion, solucion.astype(int)) else solucion)
    print("Sumas realizadas:", sumas)
    print("Restas realizadas:", restas)
    print("Multiplicaciones realizadas:", multiplicaciones)
    print("Divisiones realizadas:", divisiones)
    print("Intercambios de filas:", intercambios)

    return solucion


# ####################################################################
def descomposicion_LU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Realiza la descomposición LU de una matriz cuadrada A.
    [IMPORTANTE] No se realiza pivoteo.

    ## Parameters

    ``A``: matriz cuadrada de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior. Se obtiene de la matriz ``A`` después de aplicar la eliminación gaussiana.
    """

    A = np.array(
        A, dtype=float
    )  # convertir en float, porque si no, puede convertir como entero

    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.zeros((n, n), dtype=float)

    for i in range(0, n):  # loop por columna

        # --- deterimnar pivote
        if A[i, i] == 0:
            raise ValueError("No existe solución única.")

        # --- Eliminación: loop por fila
        L[i, i] = 1
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]

            L[j, i] = m

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        raise ValueError("No existe solución única.")

    return L, A


# ####################################################################
def resolver_LU(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante la descomposición LU.

    ## Parameters

    ``L``: matriz triangular inferior.

    ``U``: matriz triangular superior.

    ``b``: vector de términos independientes.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """

    n = L.shape[0]

    # --- Sustitución hacia adelante
    logging.info("Sustitución hacia adelante")

    y = np.zeros((n, 1), dtype=float)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        suma = 0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]

    logging.info(f"y = \n{y}")

    # --- Sustitución hacia atrás
    logging.info("Sustitución hacia atrás")
    sol = np.zeros((n, 1), dtype=float)

    sol[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        logging.info(f"i = {i}")
        suma = 0
        for j in range(i + 1, n):
            suma += U[i, j] * sol[j]
        logging.info(f"suma = {suma}")
        logging.info(f"U[i, i] = {U[i, i]}")
        logging.info(f"y[i] = {y[i]}")
        sol[i] = (y[i] - suma) / U[i, i]

    logging.debug(f"x = \n{sol}")
    return sol


# ####################################################################
def matriz_aumentada(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Construye la matriz aumentada de un sistema de ecuaciones lineales.

    ## Parameters

    ``A``: matriz de coeficientes.

    ``b``: vector de términos independientes.

    ## Return

    ``Ab``: matriz aumentada.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)
    assert A.shape[0] == b.shape[0], "Las dimensiones de A y b no coinciden."
    return np.hstack((A, b.reshape(-1, 1)))


# ####################################################################
def separar_m_aumentada(Ab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa la matriz aumentada en la matriz de coeficientes y el vector de términos independientes.

    ## Parameters
    ``Ab``: matriz aumentada.

    ## Return
    ``A``: matriz de coeficientes.
    ``b``: vector de términos independientes.
    """
    logging.debug(f"Ab = \n{Ab}")
    if not isinstance(Ab, np.ndarray):
        logging.debug("Convirtiendo Ab a numpy array")
        Ab = np.array(Ab, dtype=float)
    return Ab[:, :-1], Ab[:, -1].reshape(-1, 1)

import numpy as np

def eliminacion_gaussiana_conteo(A):
    A = np.array(A)
    n = A.shape[0]

    sumas_restas = 0

    # Eliminación hacia adelante
    for i in range(n - 1):

        # pivoteo simple
        if A[i, i] == 0:
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    break

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            for k in range(i, n + 1):
                A[j, k] = A[j, k] - m * A[i, k]
                sumas_restas += 1   # ← resta

    # Sustitución hacia atrás
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * x[j]
            sumas_restas += 1   # ← suma
        x[i] = (A[i, n] - suma) / A[i, i]
        sumas_restas += 1       # ← resta final

    return x, sumas_restas

def gauss_jordan_conteo(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    sumas_restas = 0

    for i in range(n):

        # Normalizar fila pivote
        pivote = A[i, i]
        A[i, :] = A[i, :] / pivote

        # Eliminación arriba y abajo
        for j in range(n):
            if j != i:
                m = A[j, i]
                for k in range(n + 1):
                    A[j, k] = A[j, k] - m * A[i, k]
                    sumas_restas += 1   # ← resta

    solucion = A[:, -1]
    return solucion, sumas_restas

# ####################################################################
def gauss_jordan(A: np.ndarray | list[list[float | int]]) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de Gauss-Jordan.

    ## Parameters
    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1).

    ## Return
    ``solucion``: vector con la solución del sistema de ecuaciones lineales.
    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(A, dtype=float)

    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]

    for i in range(n):

        # --- Verificar pivote
        if A[i, i] == 0:
            # buscar fila para intercambiar
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    logging.debug(f"Intercambiando filas {i} y {k}")
                    break
            else:
                raise ValueError("No existe solución única.")

        # --- Normalizar fila pivote
        pivote = A[i, i]
        A[i, :] = A[i, :] / pivote

        # --- Eliminación arriba y abajo del pivote
        for j in range(n):
            if j != i:
                m = A[j, i]
                A[j, :] = A[j, :] - m * A[i, :]

        logging.info(f"\n{A}")

    solucion = A[:, -1]
    return solucion

# ####################################################################
def gauss_jordan_conteo(A):
    import numpy as np
    import logging

    A = np.array(A, dtype=float)
    n = A.shape[0]

    sumas = 0
    restas = 0
    multiplicaciones = 0
    divisiones = 0
    intercambios = 0

    for i in range(n):

        # pivoteo simple
        if A[i, i] == 0:
            for k in range(i + 1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    intercambios += 1
                    break
            else:
                raise ValueError("No existe solución única")

        # normalizar fila pivote
        pivote = A[i, i]
        for k in range(i, n + 1):
            A[i, k] = A[i, k] / pivote
            divisiones += 1

        # eliminar arriba y abajo
        for j in range(n):
            if j != i:
                m = A[j, i]
                for k in range(i, n + 1):
                    A[j, k] = A[j, k] - m * A[i, k]
                    multiplicaciones += 1
                    restas += 1

        logging.info(f"\n{A}")

    solucion = A[:, -1]

    print("Solución:", solucion)
    print("Sumas realizadas:", sumas)
    print("Restas realizadas:", restas)
    print("Multiplicaciones realizadas:", multiplicaciones)
    print("Divisiones realizadas:", divisiones)
    print("Intercambios de filas:", intercambios)

    return solucion
