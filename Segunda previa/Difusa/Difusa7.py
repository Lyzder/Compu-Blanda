import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt

x = np.arange(-11, 11, 1)

vd_sigmoide = sk.sigmf(x, 0, 1)

plt.figure()
plt.plot(x, vd_sigmoide, 'b', linewidth=1.5, label='Servicio')
plt.title('Calidad del Servicio en un Restaurante')
plt.ylabel('Membresía')
plt.xlabel('Nivel del Servicio')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1, fancybox=True, shadow=True)