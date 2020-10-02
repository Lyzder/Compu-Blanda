## Santiago Torres Vasquez
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("web_traffic.tsv", delimiter="\t")
print(data[:10], '\n')

print(data.shape)

x = data[:,0]
y = data[:,1]

print(x, '\n')
print(y, '\n')

print(x.ndim, '\n')
print(y.ndim, '\n')

print(x.shape, '\n')
print(y.shape, '\n')

print(np.sum(np.isnan(y)))

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

plt.scatter(x, y, s=10)
plt.title("Tráfico Web en el último mes")
plt.xlabel("Tiempo")
plt.ylabel("Accesos/Hora")
plt.xticks([w*7*24 for w in range(10)], ['semana %i' % w for w in range(10)])
plt.autoscale(tight=True)

plt.grid(True, linestyle='-', color='0.75')

plt.show()
