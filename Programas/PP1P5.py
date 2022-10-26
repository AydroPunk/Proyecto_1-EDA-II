# Dadas dos matrices A1 y A2 que serán proporcionadas por el usuario en tiempo de ejecución y además los elementos contenidos en A2
# no deben sobrepasar a los de A1; orddene A1 de tal manera q1ue el orden relativo entre los elementos sea el
# mismo que el de A2. Para los elementos que no están presentes en A2, agréguelos al final en orden
# ordenado. Ejemplo:
# Entrada: A1 = [2,1,2,5,7,1,9,3,6,8,8] y A2 = [2,1,8,3]
# Salida: A1 = [2,2,1,1,8,8,3,5,6,7,9]

# Función de ordenamiento

def ordenamientoRel(A,B):
    R = []
    for i in B:
        j = 0
        while (True):
            if j >= (len(A)):
                break
            else:
                if A[j] == i:
                    R.append(i)
                    A.remove(i)
                    j -= 1
            j += 1
    A.sort()
    R.extend(A)
    return R
    

# Cuerpo principal
# Se ingresarán los dos vectores
A = []
B = []
print("\n\tIngrese los elementos de la matriz A (F para terminar)")
while (True):
    valor = str(input("Elemento: "))
    if (valor == "F") or (valor == "f"):
        break
    else:
        A.append(int(valor))

print(f"\n\tIngrese los elementos de la matriz B (Máximo {len(A)}) (F para terminar)")
i = 0
while (i < len(A)):
    valor = str(input("Elemento: "))
    if (valor == "F") or (valor == "f"):
        break
    elif (int(valor) in B):
        print("Ingrese un valor diferente")
        i -= 1
    else:
        B.append(int(valor))

print(A)
print(B)
# A se debe ordenar con respecto al orden de B
A = ordenamientoRel(A,B)
print(A)