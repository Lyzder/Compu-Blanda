import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt

x = np.arange(0, 11, 1)

calidad = sk.trimf(x, [9, 9, 10])

plt.figure()
plt.plot(x, calidad, 'b', linewidth=1.5, label='Servicio')
plt.title('Calidad del Servicio en un Restaurante')
plt.ylabel('Membresía')
plt.xlabel('Nivel del Servicio')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1, fancybox=True, shadow=True)