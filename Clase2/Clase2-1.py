# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:44:11 2020

@author: Santiago
"""

import numpy as np

#Ejemplo de propiedades de arreglos
a = np.arange(6)
print ("Arreglo a:", a, '\n')
print ("Tipo de a:", a.dtype, '\n')
print ("Dimensión de a:", a.ndim, '\n')
print ("No elementos de a:", a.shape, '\n')

#Ejemplos de arreglos bidimensionales
m = np.array([np.arange(2), np.arange(2)])
print(m)

a = np.array([[1,2], [3,4]])
print('a =\n', a, '\n')
print('a[0,0] =', a[0,0], '\n')
print('a[0,1] =', a[0,1], '\n')
print('a[1,0] =', a[1,0], '\n')
print('a[1,1] =', a[1,1], '\n')

#Ejemplos de acceso a un rango de elementos
a = np.arange(9)
print('a =', a, '\n')
print('a[0:9] = ', a[0:9], '\n')
print('a[3,7] =', a[3:7], ',\n')
print('a[0:9:1] =', a[0:9:1], '\n')
print('a[:9:1] =', a[:9:1], '\n')
print('a[0:9:2] =', a[0:9:2], '\n')
print('a[0:9:3] =', a[0:9:3], '\n')

#Ejemplos de arreglos multidimensionales
b = np.arange(24).reshape(2,3,4)
print('b =\n', b, '\n')
print('b[1,2,3] =', b[1,2,3], '\n')
print('b[0,2,2] =', b[0,2,2], '\n')
print('b[0,1,1] =', b[0,1,1], '\n')
print('b[0,0,0] =', b[0,0,0], '\n')
print('b[1,0,0] =', b[1,0,0], '\n')
print('b[:,0,0] =', b[:,0,0], '\n')
print('b[0] =\n', b[0], '\n')

print('b[0,:,:] =\n', b[0,:,:], '\n')
print('b[0, ...] =\n', b[0, ...], '\n')
print('b[0,1] =', b[0,1], '\n')

z = b[0,1]
print('z =', z, '\n')
print('z[::2] =', z[::2], '\n')
print('b[0,1,::2] =', b[0,1,::2], '\n')
print(b, '\n')
print('b[:,:,1] =\n', b[:,:,1], '\n')
print('b[...,1] =\n', b[...,1], '\n')
print('b[:,1] =', b[:,1], '\n')
print('b[0,:,1] =', b[0,:,1], '\n')
print('b[0,:,-1] =', b[0,:,-1], '\n')
print('b[0, ::-1, -1] =', b[0, ::-1, -1], '\n')
print('b[0, ::2, -1] =', b[0, ::2, -1], '\n')

#Ivertir la matriz
print(b, '\n-----------------------\n')
print(b[::-1], '\n')

#Transformar a un arreglo unidimensional
print('Matriz b =\n', b, '\n--------------------------\n')
print('Arreglo b = \n', b.ravel(), '\n')
    # La instrucción: flatten() es similar a ravel()
    # La diferencia es que flatten genera un nuevo espacio de memoria
print('Vector b con flatten =\n', b.flatten(), '\n')

#Reestructurar la 
    #Resize altera la matriz argumento
b.resize([2,12])
print('b =\n', b, '\n')
    #Reshape crea una copia de la matriz original
b.shape = (6,4)
print('b(6x4) =\n', b, '\n')

#Matriz traspuesta
print('b =\n', b, '\n------------------------\n')
print('Transpuesta de b =\n', b.transpose(), '\n')

