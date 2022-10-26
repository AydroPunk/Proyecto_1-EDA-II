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
