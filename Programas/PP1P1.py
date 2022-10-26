# Librerias necesarias
from random import randint
from time import time
from tkinter import FALSE
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')

# Demostración práctica de la eficiencia de los métodos de ordenamiento
# Agregamos las funciones respectivas para cada método

# BubbleSort mejorado (Algoritmo visto en clase)
def bubbleSort(A):
	cambio = False
	for i in range(len(A)-1):
		for j in range(len(A)-i-1):
			if A[j]>A[j+1]:
				temp = A[j]
				A[j] = A[j+1]
				A[j+1] = temp
				cambio = True
		if cambio == False:
				break	

# MergeSort (Algoritmo visto en clase)
def CrearSubArreglo (A, indIzq, indDer):
    return A[indIzq:indDer + 1]

def Merge (A,p,q,r):
    Izq = CrearSubArreglo (A,p,q)
    Der = CrearSubArreglo(A,q + 1,r)
    i = 0
    j = 0
    for k in range (p, r + 1):
        if (j >= len(Der)) or (i < len(Izq) and Izq[i] < Der[j]):
            A[k] = Izq[i]
            i = i + 1
        else:
            A[k] = Der[j]
            j = j + 1
            
def MergeSort (A,p,r):
    if r - p > 0:
        q = int((p + r)/2)
        MergeSort(A,p,q)
        MergeSort(A,q + 1, r)
        Merge(A,p,q,r)

# QuickSort (Algoritmo visto en clase)
def intercambia(A,x,y):
    tmp=A[x]
    A[x]=A[y]
    A[y]=tmp
def particionar(A,p,r):
    x=A[r]
    i=p-1
    for j in range(p,r):
        if (A[j]<=x):
            i=i+1
            intercambia(A,i,j)
    intercambia(A,i+1,r)
    return i+1
def Quicksort(A,p,r):
    if(p<r):
        q=particionar(A,p,r)
        Quicksort(A,p,q-1)
        Quicksort(A,q+1,r)

# HeapSort (Algoritmo visto en clase)
def heapify(arr, n, i):
	largest = i # Initialize largest as root
	l = 2 * i + 1	 # left = 2*i + 1
	r = 2 * i + 2	 # right = 2*i + 2

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
		arr[i], arr[largest] = arr[largest], arr[i] # swap

		# Heapify the root.
		heapify(arr, n, largest)

# The main function to sort an array of given size
def heapSort(arr):
	n = len(arr)

	# Build a maxheap.
	for i in range(n//2 - 1, -1, -1):
		heapify(arr, n, i)

	# One by one extract elements
	for i in range(n-1, 0, -1):
		arr[i], arr[0] = arr[0], arr[i] # swap
		heapify(arr, i, 0)

# CountingSort (Algoritmo optimizado para considerar negativos)
def CreaLista(k): #Esta función crea una lista de apoyo
   L=[]
   for i in range(k + 1):
       L.append(0)
   return L

#Algoritmo de ordenamiento

def CountingSort(A,k): # A es la lista y k es el valor máximo de la lista
   minimo = min(A) # Se obtiene el valor mínimo de la lista
   C = CreaLista(k) # Se crea la lista de conteo
   B = CreaLista(len(A) - 1) # Se crea la lista solución
   for j in range(0,len(A)): # Se cuentan los elementos 
       if A[j] < 0:
           C[abs(minimo) - abs(A[j])] = C[abs(minimo) - abs(A[j])] + 1 # Va de 0 a minimo - 1, guardando lugar para los negativos
       else:
           C[A[j] + abs(minimo)] = C[A[j] + abs(minimo)] + 1 # Va minimo hasta el tamaño, dando lugar a los positivos
   for i in range (1,k + 1):
       C[i] = C[i] + C[i - 1] # Se construye la matriz de conteo
   for i in range (0,k + 1):
       C[i] -= 1 # Se disminuye en 1 todos los elementos de la matriz de conteo (esto permite contabilizar al 0)
   for j in range (len(A)-1,-1,-1):
       if A[j] < 0:
           B[C[abs(minimo) - abs(A[j])]] = A[j] # Mapeo de negativos
           C[abs(minimo) - abs(A[j])] = C[abs(minimo) - abs(A[j])] - 1
       else:
           B[C[A[j] + abs(minimo)]] = A[j] # Mapeo de positivos
           C[A[j] + abs(minimo)] = C[A[j] + abs(minimo)] - 1
   return B #Retorna el la lista de apoyo B la cual es la que está ordenada

# RadixSort (Algoritmo modificado para tomar en cuenta negativos)
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

def radix(arr):
    pos = []
    neg = []
    for i in range(len(arr)):
        if arr[i] < 0:
            neg.append(abs(arr[i]))
        else:
            pos.append(arr[i])
    
    if len(pos) != 0:
        radixSort(pos)
    
    if len(neg) != 0:
        radixSort(neg)

    conta = 0
    for i in range(len(neg) - 1, -1, -1):
        arr[conta] = -neg[i]
        conta += 1

    for i in range(len(pos)):
        arr[conta] = pos[i]
        conta += 1

    
# 1) Adecuarlos para que todos funcionen con valores positivos y negativos
# 2) Crear vectores aleatorios de números enteros entre el rango [-1000,1000]. Se crearan vectores de: 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000, 100000, 150000
# 200000 y 250000. NOTA: Crear el mismo vector para acomodarlo por los diferentes métodos y tener una mejor aproximación de la eficiencia de cada uno
# Observación: Únicamente se mostrará con 500 elementos que el código genera negativos y positivos para evitar la problemas con el equipo
# Cuerpo principal

Vectores = [500,1000,2000,5000,10000,20000,40000,80000,100000,150000,200000,250000]

A = []
tiempos = {} # Lista para almacenar los diferentes tiempos
tiempos["BubbleSort"] = []
tiempos["MergeSort"] = []
tiempos["QuickSort"] = []
tiempos["HeapSort"] = []
tiempos["CountingSort"] = []
tiempos["RadixSort"] = []

for i in Vectores:
	print("\n\tCICLO CON {} ELEMENTOS".format(i))
	A.clear()
	for j in range(i): # Creamos una lista base
		A.append(randint(-1000,1000))

	# Creamos el mismo arreglo para los diferentes algoritmos para volver la comparación más precisa
	B = [x for x in A]
	C = [x for x in A]
	D = [x for x in A]
	E = [x for x in A]
	F = [x for x in A]

	# Ordenamos con los diferentes métodos para visualizar
	to = time()
	bubbleSort(A)
	tf = time()
	tiempos["BubbleSort"].append([i,tf - to])

	to = time()
	MergeSort(B,0,len(B) - 1)
	tf = time()
	tiempos["MergeSort"].append([i,tf - to])

	to = time()
	Quicksort(C,0,len(C) - 1)
	tf = time()
	tiempos["QuickSort"].append([i,tf - to])

	to = time()
	heapSort(D)
	tf = time()
	tiempos["HeapSort"].append([i,tf - to])

	to = time()
	k = max(E) + abs(min(E))
	E = CountingSort(E,k)
	tf = time()
	tiempos["CountingSort"].append([i,tf - to])

	to = time()
	radix(F)
	tf = time()
	tiempos["RadixSort"].append([i,tf - to])

	if i == 500:
		print(A, end = "\n\n") 
		print(B, end = "\n\n") 
		print(C, end = "\n\n") 
		print(D, end = "\n\n") 
		print(E, end = "\n\n") 
		print(F, end = "\n\n") 
	B.clear()
	C.clear()
	D.clear()
	E.clear()
	F.clear()

A = None
B = None
C = None
D = None
E = None
F = None

# Mostramos los resultados obtenidos
y = []
for i, j in tiempos.items():
	print(f"\n\tMétodo: {i}")
	yi = []
	for k in j:
		print(f"{str(k[0]):>30} {str(k[1]):>30}")
		yi.append(k[1])
	y.append(yi)

# Graficación
metodos = ["BubbleSort","MergeSort","QuickSort","HeapSort","CountingSort","RadixSort"] 
fig, ax = plt.subplots(3, 2, sharex = "col", sharey = False, squeeze = True)

fig2, ax2 = plt.subplots(1)
for i in range(6):
	ax2.plot(Vectores,y[i], label = metodos[i])
ax2.set_title("MÉTODOS DE ORDENAMIENTO")
ax2.set(title = "MÉTODOS DE ORDENAMIENTO", xlabel = "Número de elementos", ylabel = "Tiempo en [s]")
ax2.legend(fancybox = True, shadow = True, borderpad = 1)

ax[0,0].plot(Vectores,y[0])
ax[0,0].set_title("BubbleSort")
ax[0,1].plot(Vectores,y[1])	
ax[0,1].set_title("MergeSort")
ax[1,0].plot(Vectores,y[2])
ax[1,0].set_title("QuickSort")
ax[1,1].plot(Vectores,y[3])
ax[1,1].set_title("HeapSort")
ax[2,0].plot(Vectores,y[4])
ax[2,0].set_title("CountingSort")
ax[2,1].plot(Vectores,y[5])
ax[2,1].set_title("RadixSort")

for i in range(3):
	for j in range(2):
		ax[i,j].set(ylabel = "Tiempo en [s]")

ax[2,0].set_xlabel("Num de elementos")
ax[2,1].set_xlabel("Num de elementos")

plt.show()

exit()