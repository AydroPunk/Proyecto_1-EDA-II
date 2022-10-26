# Ordenar por frecuencia
# Imprima los elementos en una matriz imprima los elementos de una matriz en la frecuencia decreciente si 2 números
# tienen la misma frecuencia, luego imprima el que vino primero.
# Ejemplos:
# Entrada: arr [] = {2, 5, 2, 8, 5, 6, 8, 8}
# Salida: arr [] = {8, 8, 8, 2, 2, 5, 5, 6}

def orden(A): # Muestro el orden de aparición de los elementos
    B = []
    for i in range(len(A)):
        if A[i] not in B:
            B.append(A[i])
    return B

def ordenamientoF(A): # A es la lista y k es el valor máximo de la lista
    O = orden(A)
    F = [] # Matriz que almacena la frecuencia de aparición  de cada elemento
    for i in range(len(O)):
        conta = 0
        for j in range(len(A)):
            if O[i] == A[j]:
                conta += 1
        F.append(conta)
    A = []
    for i in range(len(F)):
        mayor = max(F)
        # En caso de repetirse, se almacena en orden los elementos que tienen a 2
        for j in range(len(F)):
            if F[j] == mayor:
                A.extend([O[j]] * mayor)
                F[j] = 0
    return A
        
# Cuerpo principal
arr = [2,5,2,8,5,6,8,8,-9,0,0,0,1,-9,10]
print("\n\tLISTA SIN ORDENAR")
print(arr)
arr = ordenamientoF(arr)
print("\n\tLISTA ORDENADA")
print(arr)