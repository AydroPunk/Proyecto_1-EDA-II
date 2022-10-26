"""6.-) Programa: Dada una matriz de n enteros positivos donde cada entero
puede tener dígitos hasta 10^6 (1000000) Se debe investigar la forma de generar esos
valores de manera aleatoria, y la matriz debe ser de 1000 valores, se
deberá hacer este ejercicio utilizando los 6 algoritmos revisados en clase
imprima los elementos de la matriz en orden ascendente. Al final se debe
presentar un cuadro comparativo de los resultados obtenidos.

Entrada: arr[] = {54, 724523015759812365462, 870112101220845, 8723}
Salida: 54 ,8723 ,870112101220845 ,724523015759812365462
Explicación:
Todos los elementos de la matriz se ordenan de forma no descendente (es
decir, ascendente)

bubbleSort
mergeSort
quickSort
heapSort
countingSort
radixSort"""
# ┌─────── Importando bibliotecas ───────┐
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ┌─────── Cargando módulos ───────┐
import random
from time import time


# ┌─────── BubbleSort ─────────┐
def bubbleSort(A):
    for i in range(1, len(A)):
        # print('PASADA', i)
        for j in range(len(A) - 1):
            if A[j] > A[j + 1]:
                temp = A[j]
                A[j] = A[j + 1]
                A[j + 1] = temp
            # print(A)


# A = [5, 4, 3, 2, 1]
# bubbleSort(A)
# print("\n")
# print(A)

def bubbleSort(A):
    for i in range(len(A) - 1):
        # print('PASADA', i)
        for j in range(len(A) - i - 1):
            if A[j] > A[j + 1]:
                temp = A[j]
                A[j] = A[j + 1]
                A[j + 1] = temp
            # print(A)


# A = [5, 4, 3, 2, 1]
# bubbleSort(A)
# print("\n")
# print(A)


# ┌─────── MergeSort ─────────┐
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


# A = [10, 7, 3, 1]
# MergeSort(A, 0, 3)
# print(A)


# ┌─────── QuickSort ─────────┐
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


# A = [10, 80, 30, 90, 40, 50, 70]
# Quicksort(A, 0, 6)
# print("Arreglo ordenado")
# print(A)


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


# arr = [1, 2, 3, 4, 5, 6, 7]
# heapSort(arr)
# n = len(arr)
# print("Sorted array is")
# for i in range(n):
#     print("%d" % arr[i])


# ┌─────────── CountingSort ─────────────┐
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


# A = [0, 9, 21, 4, 40, 10, 35]  # lista propuesta por la práctica
# print("Lista Ordenada", CountingSort(A, 40))  # Se manda a llamar la función de ordenamiento
# print("Lista Original", A)  # Se Imprime la lista original


# ┌─────── RadixSort ─────────┐
def countingSort(arr, exp1):
    n = len(arr)
    output = [0] * (n)
    count = [0] * (10)

    for i in range(0, n):
        index = arr[i] // exp1
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp1
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    i = 0
    for i in range(0, len(arr)):
        arr[i] = output[i]


def radixSort(arr):
    max1 = max(arr)
    exp = 1
    while max1 / exp >= 1:
        countingSort(arr, exp)
        exp *= 10


# # Driver code
# arr = [170, 45, 75, 90, 802, 24, 2, 66]
#
# # Function Call
# radixSort(arr)
#
# for i in range(len(arr)):
#     print(arr[i], end=" ")

# Tamaños de la lista de números aleatorios a generar
datos = [ii * 100 for ii in range(1, 1000)] # matriz de 1000 valores de 100 en 100
# print (datos)


time_bubbleSort = []  # Lista para guardar el tiempo de ejecución de Bubble Sort
time_MergeSort = []  # Lista para guardar el tiempo de ejecución de Merge Sort
time_Quicksort = []  # Lista para guardar el tiempo de ejecución de Quick Sort
time_heapSort = []  # Lista para guardar el tiempo de ejecución de Heap Sort
time_CountingSort = []  # Lista para guardar el tiempo de ejecución de Counting Sort
time_radixSort = []  # Lista para guardar el tiempo de ejecución de Radix Sort

for i in datos:
    lista_bubbleSort = random.sample(range(0, 10000000), i)
    # ┌─ Se hace una copia de la lista para que se ejecute el algoritmo con los mismo números ─┐
    lista_MergeSort = lista_bubbleSort.copy()
    lista_Quicksort = lista_bubbleSort.copy()
    lista_heapSort = lista_bubbleSort.copy()
    lista_CountingSort = lista_bubbleSort.copy()
    lista_radixSort = lista_bubbleSort.copy()

    t0 = time()  # Se guarda el tiempo inicial
    bubbleSort(lista_bubbleSort)
    time_bubbleSort.append(round(time() - t0, 6))  # Se le resta al tiempo actual, el tiempo inicial

    t0 = time()  # Se guarda el tiempo inicial
    (MergeSort(A=0, p=0, r=0), lista_MergeSort)
    time_MergeSort.append(round(time() - t0, 6))  # Se le resta al tiempo actual, el tiempo inicial

    t0 = time()  # Se guarda el tiempo inicial
    (Quicksort(A=0, p=0, r=0), time_Quicksort)
    time_Quicksort.append(round(time() - t0, 6))

    t0 = time()  # Se guarda el tiempo inicial
    heapSort(lista_heapSort)
    time_heapSort.append(round(time() - t0, 6))  # Se le resta al tiempo actual, el tiempo inicial

    t0 = time()  # Se guarda el tiempo inicial
    (CountingSort(A=0, k=0), lista_CountingSort)
    time_CountingSort.append(round(time() - t0, 6))  # Se le resta al tiempo actual, el tiempo inicial

    t0 = time()  # Se guarda el tiempo inicial
    radixSort(lista_radixSort)
    time_radixSort.append(round(time() - t0, 6))  # Se le resta al tiempo actual, el tiempo inicial


# NOTA: La función time() regresa el tiemo en segundos (https://docs.python.org/3/library/time.html#time.time).

# ┌──────────── Se imprimen los tiempos parciales de ejecución ────────────────────┐
print('\nTiempos parciales de ejecución de Bubble Sort: {} [s] \n'.format(time_bubbleSort))
print('Tiempos parciales de ejecución de Merge Sort: {} [s] \n'.format(time_MergeSort))
print('Tiempos parciales de ejecución de Quick Sort: {} [s] \n'.format(time_Quicksort))
print('Tiempos parciales de ejecución de Heap Sort: {} [s] \n'.format(time_heapSort))
print('Tiempos parciales de ejecución de Counting Sort: {} [s] \n'.format(time_CountingSort))
print('Tiempos parciales de ejecución de radix Sort: {} [s] \n'.format(time_radixSort))

# Se imprimen los tiempos totales de ejecución
# Para calcular el tiempo total se aplica la función sum() a las listas de tiempo
print("Tiempo total de ejecucion en Bubble Sort: {} [s]".format(sum(time_bubbleSort)))
print("Tiempo total de ejecucion en Merge Sort: {} [s]".format(sum(time_MergeSort)))
print("Tiempo total de ejecucion en Quick Sort: {} [s]".format(sum(time_Quicksort)))
print("Tiempo total de ejecucion en Heap Sort: {} [s]".format(sum(time_heapSort)))
print("Tiempo total de ejecucion en Counting Sort: {} [s]".format(sum(time_CountingSort)))
print("Tiempo total de ejecucion en radix Sort: {} [s]".format(sum(time_radixSort)))

# ┌──────────────────────── Se crea la grafica ────────────────────────┐
fig, ax = plt.subplots()
ax.plot(datos, time_bubbleSort, label='Bubble Sort', marker='o', color='r')
ax.plot(datos, time_MergeSort, label='Merge Sort', marker='o', color='g')
ax.plot(datos, time_Quicksort, label='Quick Sort', marker='o', color='y')
ax.plot(datos, time_heapSort, label='Heap Sort', marker='o', color='b')
ax.plot(datos, time_CountingSort, label='Counting Sort', marker='o', color='c')
ax.plot(datos, time_radixSort, label='Radix Sort', marker='o', color='k')
ax.set_xlabel('Datos')
ax.set_ylabel('Tiempo')
ax.grid(True)
ax.legend(loc=2)

# ┌──────────────────────── Se muestra la grafica ────────────────────────┐
plt.title('Tiempo de ejecucion [s]')
plt.show()
