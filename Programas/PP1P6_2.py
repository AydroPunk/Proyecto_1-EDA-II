# ┌─────────── BubbleSort ───────────┐
def bubbleSort(A):
    for i in range(len(A) - 1):
        for j in range(len(A) - i - 1):
            if A[j] > A[j + 1]:
                temp = A[j]
                A[j] = A[j + 1]
                A[j + 1] = temp


b = bubbleSort


# ┌─────────── CountingSort ─────────────┐
# Counting sort (Código Original)
def CreaLista(k):  # Esta función crea una lista de apoyo
    L = []
    for i in range(k + 1):
        L.append(0)
    return L


# Algoritmo de ordenamiento
def CountingSort(A, k):  # A es la lista y k es el valor máximo de la lista
    C = CreaLista(k)
    B = CreaLista(len(A) - 1)
    for j in range(1, len(A)):
        C[A[j]] = C[A[j]] + 1
    for i in range(1, k + 1):
        C[i] = C[i] + C[i - 1]
    for j in range(len(A) - 1, 0, -1):
        B[C[A[j]]] = A[j]
        C[A[j]] = C[A[j]] - 1
    return B  # Retorna el la lista de apoyo B la cual es la que está ordenada


c = CountingSort


# ┌───────────── HeapSort ───────────────────┐
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[largest] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)


# The main function to sort an array of given size
def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


h = heapSort


# ┌───────────── MergeSort ───────────────┐
def crearSubArreglo(A, indIzq, indDer):
    return A[indIzq:indDer + 1]


def Merge(A, p, q, r):
    Izq = crearSubArreglo(A, p, q)
    Der = crearSubArreglo(A, q + 1, r)
    i = 0
    j = 0
    for k in range(p, r + 1):
        if (j >= len(Der)) or (i < len(Izq) and Izq[i] and Izq[i] < Der[j]):
            A[k] = Izq[i]
            # print("lista Izquierda", Izq)
            i = i + 1
        else:
            A[k] = Der[j]
            # print("lista derecha", Der)
            j = j + 1


def MergeSort(A, p, r):
    if r - p > 0:
        q = int((p + r) / 2)
        MergeSort(A, p, q)
        MergeSort(A, q + 1, r)
        Merge(A, p, q, r)


m = MergeSort


# ┌───────── QuickSort ──────────┐
def intercambia(A, x, y):
    tmp = A[x]
    A[x] = A[y]
    A[y] = tmp


def particionar(A, p, r):
    x = A[r]
    i = p - 1
    for j in range(p, r):
        if (A[j] <= x):
            i = i + 1
            intercambia(A, i, j)
            # print(A[p:r + 1])
    intercambia(A, i + 1, r)
    # print(A[p:r + 1])
    return i + 1


def Quicksort(A, p, r):
    if (p < r):
        q = particionar(A, p, r)
        # print(A[p:r + 1])
        # print("El pivote sera:", A[r])
        Quicksort(A, p, q - 1)
        Quicksort(A, q + 1, r)


q = Quicksort


# ┌─────────── CountingSort ─────────────┐
def countingSort(arr, exp1):
    n = len(arr)

    # The output array elements that will have sorted arr
    output = [0] * (n)

    # initialize count array as 0
    count = [0] * (10)

    # Store count of occurrences in count[]
    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1

    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]


# ┌─────────── RadixSort ─────────────┐
def radixSort(arr=0):
    # Encuentra el número máximo para conocer el número de dígitos
    max1 = max(arr)

    # Hacer una ordenación de conteo para cada dígito. Tenga en cuenta que en lugar de
    # de pasar el número de dígitos, se pasa exp. exp es 10^i
    # donde i es el número de dígitos actual
    exp = 1
    while max1 / exp >= 1:
        countingSort(arr, exp)
        exp *= 10


r = radixSort

# importamos la función para generar enteros aleatorios y el tiempo
from random import randint
from time import time

# lista con todas las funciones para ordenar
Sorts = [b, c, h, m, q, r]

# tenemos 3 casos de entradas a las funciones
# 1) pasar la lista a ordenar
# 2) pasar la lista y el máximo de la lista
# 3) pasar la lista, 0 y len() de la lista menos 1
# como no hay parámetros opcionales si no se introducen los datos correctamente
# marca error, por lo que usar try es buena opción para los 3 casos

import matplotlib.pyplot as plt

points = 25
Datos = [i * 100 for i in range(1, points + 1)]
for sort in Sorts:
    plt.figure(figsize=[10, 10])
    plt.title(f"{sort.__name__}")
    T = []

    for n in Datos:
        L = [randint(1, 10 ** 7) for i in range(n)]
        maximum = max(L)
        # creamos una copia ya que algunos algoritmos son 'in place' y no queremos modificar el original
        X = L.copy()
        t = time()
        try:
            sort(X)
        except:
            try:
                sort(X, maximum)
            except:
                sort(X, 0, len(X) - 1)
        t = time() - t
        T.append(t)
        print(f"La función {sort.__name__} corrió en {t} segundos con {n} datos.")
    plt.plot(Datos, T)
    plt.scatter(Datos, T)
    # ┌───────────── Se muestra la grafica ────────────┐
    plt.show()
# Creamos el mismo arreglo para los diferentes algoritmos
# para volver la comparación más precisa
A = []
B = [x for x in A]
C = [x for x in A]

print('\n')
# ┌───────── Se muestra los arreglos ordenados ──────────┐
bubbleSort(L)
print(f"El arreglo de bubbleSort ordenado es: {L}")

CountingSort(L, max(L))
print(f"El arreglo de CountingSort CountingSort es: {L}")

heapSort(L)
print(f"El arreglo de heapSort ordenado es: {L}")

MergeSort(L, 0, len(B) - 1)
print(f"El arreglo de MergeSort ordenado es: {L}")

Quicksort(L, 0, len(C) - 1)
print(f"El arreglo de quickSort ordenado es: {L}")

radixSort(L)
print(f"El arreglo de radixSort ordenado es: {L}")
# plt.show()
