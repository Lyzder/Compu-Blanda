# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:21:14 2020

@author: Santiago
"""

import numpy as np

a = np.arange(9).reshape(3,3)
print('a =\n', a, '\n')
# Matriz b, creada a partir de la matriz a
b = a*2
print('b =\n', b)
c = np.arange(27).reshape(3, 3, 3)
print('c =\n', c)

#Matriz a y b se usaran para los ejemplos

#Apilamiento
    #Apilamiento horizontal
print('Apilamiento horizontal =\n', np.hstack((a,b)) )
    #Si al usar concatenate se define el axis=1, se hará horizontal
print('Apilamiento horizontal con concatenate = \n', np.concatenate((a,b), axis=1) )

    #Apilamiento vertical
print( 'Apilamiento vertical =\n', np.vstack((a,b)) )
    #Si al usar concatenate se define axis=0, se hará vertical
print( 'Apilamiento vertical con concatenate =\n', np.concatenate((a,b), axis=0) )

    #Apilamiento en profundidad
print( 'Apilamiento en profundidad =\n', np.dstack((a,b)) )

    #Apilamiento por columnas, similar al horizontal
print( 'Apilamiento por columnas =\n', np.column_stack((a,b)) )

    #Apilamiento por filas, similar al vertical
print( 'Apilamiento por filas =\n', np.row_stack((a,b)) )

#Divisón
    #División horizontal
print('Array con división horizontal =\n', np.hsplit(a, 3), '\n')
    #Si al usar split se define axis=1 será horizontal
print('Array con división horizontal, uso de split() =\n', np.split(a, 3, axis=1))

    #Divisón vertical
print('División Vertical = \n', np.vsplit(a, 3), '\n')
    #Si al usar split se define axis=0 será vertical
print('Array con división vertical, uso de split() =\n', np.split(a, 3, axis=0))

    #División por profundidad
print('División en profundidad =\n', np.dsplit(c,3), '\n')

#Propiedades de los arrays
    #ndim calcula el # de dimensiones
print('b ndim: ', b.ndim)
    #size calcula el # de elementos
print('b size: ', b.size)
    #itemsize obtiene el # de bytes por cada elemento en el array
print('b itemsize: ', b.itemsize)
    #nbytes obtiene el # total de bytes del array
    #Equivalente a itemsize*size
print('b nbytes: ', b.nbytes, '\n')
    #Atributo T obtiene la traspuesta
b.resize(6,4)
print(b, '\n')
print('Transpuesta:\n', b.T, '\n')
    #Los num complejos se representan con j
b = np.array([1.j + 1, 2.j + 3])
print('Complejo:', b, '\n')
    #real devuelve la parte real de un array
print('real: ', b.real, '\n')
    #imag devuelve la parte imaginaria de un array
print('imaginario: ', b.imag)
    #Si el array contiene números complejos, entonces el tipo de datos
    #se convierte automáticamente a complejo
print(b.dtype, '\n')

#Flatiter
    # El atributo flat devuelve un objeto numpy.flatiter.
    # Esta es la única forma de adquirir un flatiter:
    # no tenemos acceso a un constructor de flatiter.
    # El apartamento El iterador nos permite recorrer una matriz
    # como si fuera una matriz plana, como se muestra a continuación:
    # En el siguiente ejemplo se clarifica este concepto
b = np.arange(4).reshape(2,2)
print('b:\n', b, '\n')
f = b.flat
print(f, '\n')
    # Ciclo que itera a lo largo de f
for item in f: print (item)
    # Selección de un elemento
print('\n')
print('Elemento 2: ', b.flat[2])
    # Operaciones directas con flat
b.flat = 7
print(b, '\n')
b.flat[[1,3]] = 1
print(b, '\n')