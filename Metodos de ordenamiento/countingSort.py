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

# A = [0, 9, 21, 4, 40, 10, 35]  # lista propuesta por la práctica
# print("Lista Ordenada", CountingSort(A, 40))  # Se manda a llamar la función de ordenamiento
# print("Lista Original", A)  # Se Imprime la lista original
