import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

np.seterr(all='ignore')

data = pd.read_csv('ventas.csv',usecols=['Month', 'Sales'])
data = data.to_numpy()
y = np.array(data[:, 1], dtype=np.float64)
x = np.array([])
for i in range(1, len(y)+1):
    x = np.append(x, [i])
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

colors = ['g', 'k', 'y', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

print("Número de entradas incorrectas:", np.sum(np.isnan(y)))

def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''

    # Crea una nueva figura, o activa una existente.
    # num = identificador, figsize: anchura, altura
    plt.figure(num=None, figsize=(8, 6))
    
    # Borra el espacio de la figura
    plt.clf()
    
    # Un gráfico de dispersión de y frente a x con diferentes tamaños 
    # y colores de marcador (tamaño = 10)
    plt.scatter(x, y, s=10)
    
    # Títulos de la figura
    # Título superior
    plt.title("Ventas")
    
    # Título en la base
    plt.xlabel("Año")
    
    # Título lateral
    plt.ylabel("Ventas/Mes")
    
    # Obtiene o establece las ubicaciones de las marcas 
    # actuales y las etiquetas del eje x.
    
    # Los primeros corchetes ([]) se refieren a las marcas en x
    # Los siguientes corchetes ([]) se refieren a las etiquetas
    
    # En el primer corchete se tiene: 1*12 + 2*12 + ..., hasta
    # completar el total de puntos en el eje horizontal, según
    # el tamaño del vector x
    plt.xticks(
        [w * 12 for w in range(10)], 
        [w for w in range(10)])

    # Aquí se evalúa el tipo de modelo recibido
    # Si no se envía ninguno, no se dibuja ninguna curva de ajuste
    if models:
        
        # Si no se define ningún valor para mx (revisar el 
        # código más adelante), el valor de mx será
        # calculado con la función linspace

        # NOTA: linspace devuelve números espaciados uniformemente 
        # durante un intervalo especificado. En este caso, sobre
        # el conjunto de valores x establecido
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
        
        # La función zip () toma elementos iterables 
        # (puede ser cero o más), los agrega en una tupla y los devuelve
     # Aquí se realiza un ciclo donde se grafican los distintos modelos
        # pasados como argumento, cada uno con un estilo de línea y color
        # distinto para su distinción.
        
        for model, style, color in zip(models, linestyles, colors):
            # print "Modelo:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        # Se añaden etiquetas en la esquina superior izqueirda para facilitar 
        # la identificación de los modelos según el grado
        plt.legend(["d=%i" % m.order for m in models], loc="upper left")
    
    # Aquí se ajusta las dimensiones de la gráfica en función de los
    # límites de datos.
    
    plt.autoscale(tight=True)
    
    # Debido a la naturaleza del modelo, se define el límite inferior de Y
    # como 0, ya que es irrealista un tráfico web negativo.
    plt.ylim(ymin=0)
    if ymax:
    
        # Si se pasó un argumento ymax, se define el límite superior de la
        # gráfica a ese valor.
        plt.ylim(ymax=ymax)
        
    if xmin:
        
        # Si se pasó un argumento xmin, se define el límite izquierdo de la
        # gráfica a ese valor.
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)
    
plot_models(x, y, None, "1400_01_01.png")
# Creación del modelo de grado 1. Se pasa el argumento full=True
fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)
f1 = sp.poly1d(fp1)

# Creación del modelo de grado 2. Se pasa el argumento full=True
fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)
print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = sp.poly1d(fp2)

# Creación del modelo de grado 3.
f3 = sp.poly1d(np.polyfit(x, y, 3))
# Creación del modelo de grado 4.
f4 = sp.poly1d(np.polyfit(x, y, 4))
# Creación del modelo de grado 5.
f5 = sp.poly1d(np.polyfit(x, y, 5))

# Se grafican los modelos
# -----------------------------------------------------------------

# Se grafican los datos, esta vez con cada modelo siendo agregado de forma
# progresiva, siendo guardados como archivos independientes.

plot_models(x, y, [f1], "1400_01_02.png")
plot_models(x, y, [f1, f2], "1400_01_03.png")
plot_models(
    x, y, [f1, f2, f3, f4, f5], "1400_01_04.png")

# Ajusta y dibuja un modelo utilizando el conocimiento del punto
# de inflexión
# -----------------------------------------------------------------

# Se calcula el punto de inflexión de los datos, para luego ser usado
# para poder estudiar los errores antes y después de la inflexión,
# así como índice en el arreglo de datos y usar los datos que indexan
# para crear dos rectas que representan los dos comportamientos".

inflexion = 8
xa = x[:int(inflexion)]
ya = y[:int(inflexion)]
xb = x[int(inflexion):]
yb = y[int(inflexion):]

# Se grafican dos líneas rectas
# -----------------------------------------------------------------

# Se crean las dos rectas como objetos de la clase poly1d para mayor
# facilidad de uso.
fa = sp.poly1d(np.polyfit(xa, ya, 1))
fb = sp.poly1d(np.polyfit(xb, yb, 1))

# Se presenta el modelo basado en el punto de inflexión
# -----------------------------------------------------------------
plot_models(x, y, [fa, fb], "1400_01_05.png")

# Función de error
# -----------------------------------------------------------------

# Se calcula el error de un modelo usando la sumatoria de (f(x) - y)^2,
# donde f(x) es el valor del modelo, e y es el dato real.
def error(f, x, y):
    return np.sum((f(x) - y) ** 2)

# Se imprimen los errores
# -----------------------------------------------------------------

# Se calcula el error de cada modelo evaluado en el conjunto completo
# de datos usando la función error  definida previamente, y se imprimen
# junto al grado del modelo para facilitar la evaluación.

print("Errores para el conjunto completo de datos:")
for f in [f1, f2, f3, f4, f5]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

# Se calcula el error de cada modelo evaluado solamente después de la
# inflexión, usando como argumentos los valores de la segunda recta
# que se graficó previamente para describir el comportamiento luego de
# la inflexión.

print("Errores solamente después del punto de inflexión")
for f in [f1, f2, f3, f4, f5]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))
    
# Se calcula exclusivamente el error de las rectas de inflexión, calculando
# sus errores independientemente para luego sumarlos.

print("Error de inflexión=%f" % (error(fa, xa, ya) + error(fb, xb, yb)))

# Se extrapola de modo que se proyecten respuestas en el futuro
# -----------------------------------------------------------------

# Se grafican todos los modelos al mismo tiempo, pero esta vez, se aplica
# una interpolación de forma que los modelos se extiendan e intenten predecir
# el comportamiento a futuro de los datos. Esta gráfica también es almacenada
# como un archivo independiente.
plot_models(
    x, y, [f1, f2, f3, f4, f5],
    "1400_01_06.png",
    mx=np.linspace(0 * 12, 5 * 12, 100),
    ymax=700, xmin=0 * 12)

# La parte que sigue es relativa al entrenamiento del modelo
# y la predicción

print("Entrenamiento de datos únicamente despúes del punto de inflexión")
fb1 = fb
fb2 = sp.poly1d(np.polyfit(xb, yb, 2))
fb3 = sp.poly1d(np.polyfit(xb, yb, 3))
f4 = sp.poly1d(np.polyfit(xb, yb, 4))
f5 = sp.poly1d(np.polyfit(xb, yb,5))

print("Errores después del punto de inflexión")
for f in [fb1, fb2, fb3, f4, f5]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

# Gráficas después del punto de inflexión
# -----------------------------------------------------------------
plot_models(
    x, y, [fb1, fb2, fb3, f4, f5],
    "1400_01_07.png",
    mx=np.linspace(0 * 12, 5 * 12, 100),
    ymax=700, xmin=0 * 12)

# Separa el entrenamiento de los datos de prueba
# -----------------------------------------------------------------
#Se dividen los modelos de forma aleateoria antes de entrenarlos, usando
#la función permutation de Scipy
frac = 0.3
split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
#Las divisiones se califican como puntos de entrenamiento y puntos de prueba
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
#Cada modelo es entrenado usando los puntos de entrenamiento
fbt1 = sp.poly1d(np.polyfit(xb[train], yb[train], 1))
fbt2 = sp.poly1d(np.polyfit(xb[train], yb[train], 2))
fbt3 = sp.poly1d(np.polyfit(xb[train], yb[train], 3))
fbt4 = sp.poly1d(np.polyfit(xb[train], yb[train], 4))
fbt5 = sp.poly1d(np.polyfit(xb[train], yb[train], 5))

#Usando los puntos de prueba, se testea cada modelo entrenado, haciendo
#uso de la función error definida previamente
print("Prueba de error para después del punto de inflexión")
for f in [fbt1, fbt2, fbt3, fbt4, fbt5]:
    print("Error d=%i: %f" % (f.order, error(f, xb[test], yb[test])))

#Se grafican todos los modelos para poder hacer evaluaciones visuales
plot_models(
    x, y, [fbt1, fbt2, fbt3, fbt4, fbt5],
    "1400_01_08.png",
    mx=np.linspace(0 * 12, 5 * 12, 100),
    ymax=1000, xmin=0 * 12)

#Se estimula el tiempo que requiere llegar a un valor objetivo y=1000
#por cada modelo
from scipy.optimize import fsolve
for f in [fbt1, fbt2, fbt3, fbt4, fbt5]:
    print(f)
    #Se despeja la ecuación para resolver x, el tiempo requerido
    print(f - 1000)
    #Se define un x0=40 debido a que la cantidad total de datos es 36,
    #y se desea predecir luego del último dato
    alcanzado_max = fsolve(f - 1000, x0=40)/12
    print("\n1,000 ventas/mes esperados en el año %f" % 
          alcanzado_max[0])