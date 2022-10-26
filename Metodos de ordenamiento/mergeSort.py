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
# print(max(len(A), 1))
# MergeSort(A, 0, 3)
# print(A)
