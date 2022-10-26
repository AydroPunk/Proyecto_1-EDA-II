# Cuente todos los pares distintos con una diferencia igual a k
# Dada una matriz de enteros y un entero positivo k, cuente todos los pares distintos
# con diferencias iguales a k.
# Ejemplos:
# Entrada: arr = [1,5,3,4,2], k = 3
# Salida: 2
# Hay dos pares con diferencia 3, los pares son [1,4] y [5,2]

def pares (arr,k):
    P = []
    conta = 0
    for (i,e1) in enumerate(arr):
        print(f"\nValor de i: {i}, Valor de e1: {e1}")
        for e2 in arr[i + 1:]:
            print(f"Valor de e2: {e2}")
            if abs(e1 - e2) == k:
                P.append((e1,e2))
                conta += 1
    return P,conta

# Cuerpo principal
n = int(input("Ingrese el número de elementos del arreglo: "))
arr = []
for i in range(n):
    arr.append(int(input("Ingrese número entero: ")))

k = int(input("Ingrese el valor de las diferencias (k): "))
P = [] # Lista de pares

P, c = pares(arr, k)
print ("\n\tSe encontraron {} pares, los cuales son: \n\t{}".format(c,P))