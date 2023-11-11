from math import *
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

ruta_del_archivo = "CO-OPS_8452660_met(1-2021).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))

plt.plot(tiempos[:240],alturas[:240])
plt.show()

fecha = '2021/01/01'

# Utiliza groupby() para agrupar por la columna 'Date' y luego get_group() para obtener los registros de la categoría deseada
registros_categoria_deseada = data.groupby('Date').get_group(fecha)

tiempos = registros_categoria_deseada['Time (GMT)'].tolist()
alturas = registros_categoria_deseada['Verified (m)'].tolist()

maximos = []
minimos = []

for i in range(0,len(alturas),1):
    if i == 0:
        if alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    elif i == (len(alturas)-1):
        if alturas[i] > alturas[i-1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    else:
        if alturas[i-1] < alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        elif alturas[i-1] > alturas[i] < alturas[i+1]:
            minimos.append(tiempos[i])

def tiempo_a_int(strings):
    nuevo_array=[]
    for tiempos in strings:
        horas, minutos = tiempos.split(':')
        horas = int(horas)
        minutos = int(minutos)
        minutos = minutos + horas * 60
        nuevo_array.append(minutos)
    return nuevo_array

def tiempo_a_strings(ints):
    tiempo = round(ints)
    horas = tiempo // 60
    minutos = tiempo % 60
    horas = str(horas)
    minutos = str(minutos)
    tiempo = horas + ':' + minutos
    return tiempo

def calculador_diferencias(tiempos):
    array_diferencias = []
    for i in range(0,len(tiempos)-1,1):
        diferencia = tiempos[i+1] - tiempos[i]
        array_diferencias.append(diferencia)
    return array_diferencias

tiempos_en_minutos_maximos = tiempo_a_int(maximos)
tiempos_en_minutos_minimos = tiempo_a_int(minimos)
diferencia_de_maximos = calculador_diferencias(tiempos_en_minutos_maximos)
diferencia_de_minimos = calculador_diferencias(tiempos_en_minutos_minimos)
promedio_diferencia_maximos = np.mean(diferencia_de_maximos)
promedio_diferencia_minimos = np.mean(diferencia_de_minimos)

print(tiempo_a_strings(promedio_diferencia_maximos))
print(tiempo_a_strings(promedio_diferencia_minimos))

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))
transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
mitad = transformada_fourier[:3720]
x = np.arange(mitad.size)
plt.bar(x,mitad)
plt.show()

frecuencia_mayor = np.argmax(transformada_fourier)
periodo = len(alturas)/frecuencia_mayor
print(periodo)

def pseudo_producto_escalar (vector1,vector2):
    resultado = 0
    for i in range(0,len(vector1),1):
        resultado = resultado + vector1[i] * vector2[i]
    return resultado

def calculador_pseudo_productos(fis):
    resultados = []
    iteraciones = 0
    for i in range(0,len(fis),1):
            for j in range(i,len(fis),1):
                resultados.append(pseudo_producto_escalar(fis[i],fis[j]))
    return resultados

def armador_matriz_fis(pseudo_productos,cantidad_fis):
    columnas = []
    for j in range(0,cantidad_fis,1):
        fila = []
        posicion = j
        ingresos = 0
        fila.append(pseudo_productos[posicion])
        ingresos = ingresos + 1
        
        if j != 0:
            for w in range(0,j,1):
                posicion = posicion+cantidad_fis-w
                fila.append(pseudo_productos[posicion])
                ingresos = ingresos + 1

        if cantidad_fis-ingresos != 0:
            for i in range(1,cantidad_fis-ingresos+1,1):
                fila.append(pseudo_productos[posicion+i])
        
        columnas.append(fila)
    matriz = np.matrix(columnas)
    return matriz

def armador_matriz_f(pseudo_productos,cantidad_fis):
    columnas = []
    posicion = 0
    for j in range(0,cantidad_fis,1):
        fila = []
        w=j-1
        if j <= 1:
            w=0
        posicion = posicion+cantidad_fis-w
        if posicion < len(pseudo_productos):
            fila.append(pseudo_productos[posicion])
        columnas.append(fila)
    matriz = np.matrix(columnas)
    return matriz

def armador_matriz_c(cantidad_fis):
    columnas = []
    for i in range(0,cantidad_fis,1):
        fila = []
        fila.append(0)
        columnas.append(fila)
    matriz = np.matrix(columnas)
    return matriz

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno = frecuencia_angular * frecuencia_mayor

fis = []
fi1 = []
fi2 = []
fi3 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno*i))
    fi3.append(np.sin(interior_coseno*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,3)
matriz_f = armador_matriz_f(pseudo_productos,3)
matriz_c = armador_matriz_c(3)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

c = np.sqrt(cs[1]**2+cs[2]**2)
k = np.arctan(-cs[1]/cs[2])

f = []
x = []
for i in range(0,7440,1):
    x.append(i)
    f.append(cs[0] + c*np.cos(interior_coseno*i+k))

f = np.squeeze(f)

plt.figure()
plt.plot(x,f)
plt.plot(tiempos,alturas, color="red")
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(2021).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))

plt.plot(tiempos, alturas)
plt.show()

fecha = '2021/01/01'

# Utilizo groupby() para agrupar por la columna 'Date' y luego get_group() para obtener los registros de la fecha deseada
registros_categoria_deseada = data.groupby('Date').get_group(fecha)

tiempos = registros_categoria_deseada['Time (GMT)'].tolist()
alturas = registros_categoria_deseada['Verified (m)'].tolist()

maximos = []
minimos = []

for i in range(0,len(alturas),1):
    if i == 0:
        if alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    elif i == (len(alturas)-1):
        if alturas[i] > alturas[i-1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    else:
        if alturas[i-1] < alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        elif alturas[i-1] > alturas[i] < alturas[i+1]:
            minimos.append(tiempos[i])

tiempos_en_minutos_maximos = tiempo_a_int(maximos)
tiempos_en_minutos_minimos = tiempo_a_int(minimos)
diferencia_de_maximos = calculador_diferencias(tiempos_en_minutos_maximos)
diferencia_de_minimos = calculador_diferencias(tiempos_en_minutos_minimos)
promedio_diferencia_maximos = np.mean(diferencia_de_maximos)
promedio_diferencia_minimos = np.mean(diferencia_de_minimos)

print(tiempo_a_strings(promedio_diferencia_maximos))
print(tiempo_a_strings(promedio_diferencia_minimos))

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))
transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
print(len(alturas))
mitad = transformada_fourier[:4380]
x = np.arange(mitad.size)
plt.bar(x,mitad)
plt.show()

frecuencia_mayor = np.argmax(transformada_fourier)
periodos = len(alturas)/frecuencia_mayor
print(periodos)

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno = frecuencia_angular * frecuencia_mayor

fis = []
fi1 = []
fi2 = []
fi3 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno*i))
    fi3.append(np.sin(interior_coseno*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,3)
matriz_f = armador_matriz_f(pseudo_productos,3)
matriz_c = armador_matriz_c(3)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,4380,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno*i) + cs[2]*np.sin(interior_coseno*i))

f = np.squeeze(f)

plt.clf()
plt.plot(x[:336],f[:336])
plt.plot(tiempos[:336],alturas[:336], color="red")
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(1-2022).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))

plt.clf()
plt.plot(tiempos, alturas)
plt.show()

fecha = '2022/01/01'

# Utiliza groupby() para agrupar por la columna 'Date' y luego get_group() para obtener los registros de la categoría deseada
registros_categoria_deseada = data.groupby('Date').get_group(fecha)

tiempos = registros_categoria_deseada['Time (GMT)'].tolist()
alturas = registros_categoria_deseada['Verified (m)'].tolist()

maximos = []
minimos = []

for i in range(0,len(alturas),1):
    if i == 0:
        if alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    elif i == (len(alturas)-1):
        if alturas[i] > alturas[i-1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    else:
        if alturas[i-1] < alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        elif alturas[i-1] > alturas[i] < alturas[i+1]:
            minimos.append(tiempos[i])

tiempos_en_minutos_maximos = tiempo_a_int(maximos)
tiempos_en_minutos_minimos = tiempo_a_int(minimos)
diferencia_de_maximos = calculador_diferencias(tiempos_en_minutos_maximos)
diferencia_de_minimos = calculador_diferencias(tiempos_en_minutos_minimos)
promedio_diferencia_maximos = np.mean(diferencia_de_maximos)
promedio_diferencia_minimos = np.mean(diferencia_de_minimos)

print(tiempo_a_strings(promedio_diferencia_maximos))
print(tiempo_a_strings(promedio_diferencia_minimos))

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))
transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
mitad = transformada_fourier[:3720]
x = np.arange(mitad.size)
plt.clf()
plt.bar(x,mitad)
plt.show()

frecuencia_mayor = np.argmax(transformada_fourier)
periodos = len(alturas)/frecuencia_mayor
print(periodos)

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno = frecuencia_angular * frecuencia_mayor

fis = []
fi1 = []
fi2 = []
fi3 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno*i))
    fi3.append(np.sin(interior_coseno*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,3)
matriz_f = armador_matriz_f(pseudo_productos,3)
matriz_c = armador_matriz_c(3)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,7440,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno*i) + cs[2]*np.sin(interior_coseno*i))

f = np.squeeze(f)

plt.clf()
plt.plot(x,f)
plt.plot(tiempos,alturas, color="red")
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(2022).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))
plt.clf()
plt.plot(tiempos, alturas)
plt.show()

fecha = '2022/01/01'

# Utiliza groupby() para agrupar por la columna 'Date' y luego get_group() para obtener los registros de la categoría deseada
registros_categoria_deseada = data.groupby('Date').get_group(fecha)

tiempos = registros_categoria_deseada['Time (GMT)'].tolist()
alturas = registros_categoria_deseada['Verified (m)'].tolist()

maximos = []
minimos = []

for i in range(0,len(alturas),1):
    if i == 0:
        if alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    elif i == (len(alturas)-1):
        if alturas[i] > alturas[i-1]:
            maximos.append(tiempos[i])
        else:
            minimos.append(tiempos[i])
    else:
        if alturas[i-1] < alturas[i] > alturas[i+1]:
            maximos.append(tiempos[i])
        elif alturas[i-1] > alturas[i] < alturas[i+1]:
            minimos.append(tiempos[i])

tiempos_en_minutos_maximos = tiempo_a_int(maximos)
tiempos_en_minutos_minimos = tiempo_a_int(minimos)
diferencia_de_maximos = calculador_diferencias(tiempos_en_minutos_maximos)
diferencia_de_minimos = calculador_diferencias(tiempos_en_minutos_minimos)
promedio_diferencia_maximos = np.mean(diferencia_de_maximos)
promedio_diferencia_minimos = np.mean(diferencia_de_minimos)

print(tiempo_a_strings(promedio_diferencia_maximos))
print(tiempo_a_strings(promedio_diferencia_minimos))

alturas = data['Verified (m)']
tiempos = np.linspace(0,len(alturas),len(alturas))
transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
mitad = transformada_fourier[:4380]
x = np.arange(mitad.size)
plt.clf()
plt.bar(x,mitad)
plt.show()

frecuencia_mayor = np.argmax(transformada_fourier)
periodos = len(alturas)/frecuencia_mayor
print(periodos)

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno = frecuencia_angular * frecuencia_mayor

fis = []
fi1 = []
fi2 = []
fi3 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno*i))
    fi3.append(np.sin(interior_coseno*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,3)
matriz_f = armador_matriz_f(pseudo_productos,3)
matriz_c = armador_matriz_c(3)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno*i) + cs[2]*np.sin(interior_coseno*i))

f = np.squeeze(f)
plt.clf()
plt.plot(x[:336],f[:336])
plt.plot(tiempos[:336],alturas[:336], color="red")
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(2021).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = data['Time (GMT)']
tiempos1 = np.linspace(0,len(tiempos),len(tiempos))
plt.clf()
plt.plot(tiempos1, alturas)
plt.show()

transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
transformada_fourier = transformada_fourier[:4380]
x = np.arange(transformada_fourier.size)
plt.clf()
plt.bar(x,transformada_fourier)
plt.show()

frecuencias_mayores = []
maximo = np.argmax(transformada_fourier)
transformada_fourier[maximo] = 0
maximo2 = np.argmax(transformada_fourier)
frecuencias_mayores.append(maximo)
frecuencias_mayores.append(maximo2)
periodos = []
periodos.append(len(alturas)/frecuencias_mayores[0])
periodos.append(len(alturas)/frecuencias_mayores[1])
print(periodos)

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno1 = frecuencia_angular * frecuencias_mayores[0]
interior_coseno2 = frecuencia_angular * frecuencias_mayores[1]

fis = []
fi1 = []
fi2 = []
fi3 = []
fi4 = []
fi5 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno1*i))
    fi3.append(np.sin(interior_coseno1*i))
    fi4.append(np.cos(interior_coseno2*i))
    fi5.append(np.sin(interior_coseno2*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(fi4)
fis.append(fi5)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,5)
matriz_f = armador_matriz_f(pseudo_productos,5)
matriz_c = armador_matriz_c(5)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i))

f = np.squeeze(f)

semana_alturas = alturas[:168]
semana_tiempos = tiempos1[:168]
semana_x = x[:168]
semana_f = f[:168]

np_alturas = np.array(alturas)
np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

transformada_fourier[maximo2] = 0
maximo3 = np.argmax(transformada_fourier)

frecuencias_mayores.append(maximo3)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno3 = frecuencia_angular * frecuencias_mayores[2]

fi6 = []
fi7 = []
for i in range(0,len(tiempos),1):
    fi6.append(np.cos(interior_coseno3*i))
    fi7.append(np.sin(interior_coseno3*i))

fis = fis[:-1]
fis.append(fi6)
fis.append(fi7)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,7)
matriz_f = armador_matriz_f(pseudo_productos,7)
matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo4 = np.argmax(transformada_fourier)

frecuencias_mayores.append(maximo4)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno4 = frecuencia_angular * frecuencias_mayores[3]

fi8 = []
fi9 = []
for i in range(0,len(tiempos),1):
    fi8.append(np.cos(interior_coseno4*i))
    fi9.append(np.sin(interior_coseno4*i))

fis = fis[:-1]
fis.append(fi8)
fis.append(fi9)
fis.append(alturas)


pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,9)
matriz_f = armador_matriz_f(pseudo_productos,9)

matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo5 = np.argmax(transformada_fourier)

frecuencias_mayores.append(maximo5)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno5 = frecuencia_angular * frecuencias_mayores[4]

fi10 = []
fi11 = []
for i in range(0,len(tiempos),1):
    fi10.append(np.cos(interior_coseno5*i))
    fi11.append(np.sin(interior_coseno5*i))

fis = fis[:-1]
fis.append(fi10)
fis.append(fi11)
fis.append(alturas)


pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,11)
matriz_f = armador_matriz_f(pseudo_productos,11)

matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i) + cs[9]*np.cos(interior_coseno5*i) + cs[10]*np.sin(interior_coseno5*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo6 = np.argmax(transformada_fourier)

frecuencias_mayores.append(maximo6)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno6 = frecuencia_angular * frecuencias_mayores[5]

fi12 = []
fi13 = []
for i in range(0,len(tiempos),1):
    fi12.append(np.cos(interior_coseno6*i))
    fi13.append(np.sin(interior_coseno6*i))

fis = fis[:-1]
fis.append(fi12)
fis.append(fi13)
fis.append(alturas)


pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,13)
matriz_f = armador_matriz_f(pseudo_productos,13)

matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i) + cs[9]*np.cos(interior_coseno5*i) + cs[10]*np.sin(interior_coseno5*i) + cs[11]*np.cos(interior_coseno6*i) + cs[12]*np.sin(interior_coseno6*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo7 = np.argmax(transformada_fourier)
indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo8 = np.argmax(transformada_fourier)
indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo9 = np.argmax(transformada_fourier)
indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo10 = np.argmax(transformada_fourier)
indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo11 = np.argmax(transformada_fourier)
indice_maximo = np.argmax(transformada_fourier)
transformada_fourier[indice_maximo] = 0
maximo12 = np.argmax(transformada_fourier)

frecuencias_mayores.append(maximo7)
frecuencias_mayores.append(maximo8)
frecuencias_mayores.append(maximo9)
frecuencias_mayores.append(maximo10)
frecuencias_mayores.append(maximo11)
frecuencias_mayores.append(maximo12)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno7 = frecuencia_angular * frecuencias_mayores[6]
interior_coseno8 = frecuencia_angular * frecuencias_mayores[7]
interior_coseno9 = frecuencia_angular * frecuencias_mayores[8]
interior_coseno10 = frecuencia_angular * frecuencias_mayores[9]
interior_coseno11 = frecuencia_angular * frecuencias_mayores[10]
interior_coseno12 = frecuencia_angular * frecuencias_mayores[11]

fi14 = []
fi15 = []
fi16 = []
fi17 = []
fi18 = []
fi19 = []
fi20 = []
fi21 = []
fi22 = []
fi23 = []
fi24 = []
fi25 = []
for i in range(0,len(tiempos),1):
    fi14.append(np.cos(interior_coseno7*i))
    fi15.append(np.sin(interior_coseno7*i))
    fi16.append(np.cos(interior_coseno8*i))
    fi17.append(np.sin(interior_coseno8*i))
    fi18.append(np.cos(interior_coseno9*i))
    fi19.append(np.sin(interior_coseno9*i))
    fi20.append(np.cos(interior_coseno10*i))
    fi21.append(np.sin(interior_coseno10*i))
    fi22.append(np.cos(interior_coseno11*i))
    fi23.append(np.sin(interior_coseno11*i))
    fi24.append(np.cos(interior_coseno12*i))
    fi25.append(np.sin(interior_coseno12*i))

fis = fis[:-1]
fis.append(fi14)
fis.append(fi15)
fis.append(fi16)
fis.append(fi17)
fis.append(fi18)
fis.append(fi19)
fis.append(fi20)
fis.append(fi21)
fis.append(fi22)
fis.append(fi23)
fis.append(fi24)
fis.append(fi25)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,25)
matriz_f = armador_matriz_f(pseudo_productos,25)

matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i) + cs[9]*np.cos(interior_coseno5*i) + cs[10]*np.sin(interior_coseno5*i) + cs[11]*np.cos(interior_coseno6*i) + cs[12]*np.sin(interior_coseno6*i) + cs[13]*np.cos(interior_coseno7*i) + cs[14]*np.sin(interior_coseno7*i) + cs[15]*np.cos(interior_coseno8*i) + cs[16]*np.sin(interior_coseno8*i) + cs[17]*np.cos(interior_coseno9*i) + cs[18]*np.sin(interior_coseno9*i) + cs[19]*np.cos(interior_coseno10*i) + cs[20]*np.sin(interior_coseno10*i) + cs[21]*np.cos(interior_coseno11*i) + cs[22]*np.sin(interior_coseno11*i) + cs[23]*np.cos(interior_coseno12*i) + cs[24]*np.sin(interior_coseno12*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(2022).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = data['Time (GMT)']
tiempos1 = np.linspace(0,len(tiempos),len(tiempos))
plt.clf()
plt.plot(tiempos1, alturas)
plt.show()

transformada_fourier = abs(np.fft.fft(alturas))
transformada_fourier[0]=0
transformada_fourier = transformada_fourier[:4380]
x = np.arange(transformada_fourier.size)
plt.clf()
plt.bar(x,transformada_fourier)
plt.show()

frecuencias_mayores = []
maximo = np.argmax(transformada_fourier)
transformada_fourier[maximo] = 0
maximo2 = np.argmax(transformada_fourier)
frecuencias_mayores.append(maximo)
frecuencias_mayores.append(maximo2)
periodos = []
periodos.append(len(alturas)/frecuencias_mayores[0])
periodos.append(len(alturas)/frecuencias_mayores[1])
print(periodos)

frecuencia_angular = 2 * np.pi / len(alturas)
interior_coseno1 = frecuencia_angular * frecuencias_mayores[0]
interior_coseno2 = frecuencia_angular * frecuencias_mayores[1]

fis = []
fi1 = []
fi2 = []
fi3 = []
fi4 = []
fi5 = []
for i in range(0,len(tiempos),1):
    fi1.append(1)
    fi2.append(np.cos(interior_coseno1*i))
    fi3.append(np.sin(interior_coseno1*i))
    fi4.append(np.cos(interior_coseno2*i))
    fi5.append(np.sin(interior_coseno2*i))

fis.append(fi1)
fis.append(fi2)
fis.append(fi3)
fis.append(fi4)
fis.append(fi5)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,5)
matriz_f = armador_matriz_f(pseudo_productos,5)
matriz_c = armador_matriz_c(5)

inversa_matriz_fi = np.linalg.inv(matriz_fi)

matriz_c = inversa_matriz_fi * matriz_f

cs = np.array(matriz_c)

c1 = np.sqrt(cs[1]**2+cs[2]**2)
c2 = np.sqrt(cs[3]**2+cs[4]**2)

k1 = np.arctan(-cs[1]/cs[2])
k2 = np.arctan(-cs[3]/cs[4])


f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + c1*np.cos(interior_coseno1*i+k1) + c2*np.cos(interior_coseno2*i+k2))

f = np.squeeze(f)

semana_alturas = alturas[:168]
semana_tiempos = tiempos1[:168]
semana_x = x[:168]
semana_f = f[:168]

np_alturas = np.array(alturas)
np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

transformada_fourier[maximo2] = 0
maximo3 = np.argmax(transformada_fourier)
transformada_fourier[maximo3] = 0
maximo4 = np.argmax(transformada_fourier)
transformada_fourier[maximo4] = 0
maximo5 = np.argmax(transformada_fourier)
transformada_fourier[maximo5] = 0
maximo6 = np.argmax(transformada_fourier)
transformada_fourier[maximo6] = 0
maximo7 = np.argmax(transformada_fourier)
transformada_fourier[maximo7] = 0
maximo8 = np.argmax(transformada_fourier)
transformada_fourier[maximo8] = 0
maximo9 = np.argmax(transformada_fourier)
transformada_fourier[maximo9] = 0
maximo10 = np.argmax(transformada_fourier)
transformada_fourier[maximo10] = 0
maximo11 = np.argmax(transformada_fourier)
transformada_fourier[maximo11] = 0
maximo12 = np.argmax(transformada_fourier)
frecuencias_mayores.append(maximo3)
frecuencias_mayores.append(maximo4)
frecuencias_mayores.append(maximo5)
frecuencias_mayores.append(maximo6)
frecuencias_mayores.append(maximo7)
frecuencias_mayores.append(maximo8)
frecuencias_mayores.append(maximo9)
frecuencias_mayores.append(maximo10)
frecuencias_mayores.append(maximo11)
frecuencias_mayores.append(maximo12)

frecuencia_angular = 2 * np.pi / len(tiempos)
interior_coseno3 = frecuencia_angular * frecuencias_mayores[2]
interior_coseno4 = frecuencia_angular * frecuencias_mayores[3]
interior_coseno5 = frecuencia_angular * frecuencias_mayores[4]
interior_coseno6 = frecuencia_angular * frecuencias_mayores[5]
interior_coseno7 = frecuencia_angular * frecuencias_mayores[6]
interior_coseno8 = frecuencia_angular * frecuencias_mayores[7]
interior_coseno9 = frecuencia_angular * frecuencias_mayores[8]
interior_coseno10 = frecuencia_angular * frecuencias_mayores[9]
interior_coseno11 = frecuencia_angular * frecuencias_mayores[10]
interior_coseno12 = frecuencia_angular * frecuencias_mayores[11]

fi6 = []
fi7 = []
fi8 = []
fi9 = []
fi10 = []
fi11 = []
fi12 = []
fi13 = []
fi14 = []
fi15 = []
fi16 = []
fi17 = []
fi18 = []
fi19 = []
fi20 = []
fi21 = []
fi22 = []
fi23 = []
fi24 = []
fi25 = []
for i in range(0,len(tiempos),1):
    fi6.append(np.cos(interior_coseno3*i))
    fi7.append(np.sin(interior_coseno3*i))
    fi8.append(np.cos(interior_coseno4*i))
    fi9.append(np.sin(interior_coseno4*i))
    fi10.append(np.cos(interior_coseno5*i))
    fi11.append(np.sin(interior_coseno5*i))
    fi12.append(np.cos(interior_coseno6*i))
    fi13.append(np.sin(interior_coseno6*i))
    fi14.append(np.cos(interior_coseno7*i))
    fi15.append(np.sin(interior_coseno7*i))
    fi16.append(np.cos(interior_coseno8*i))
    fi17.append(np.sin(interior_coseno8*i))
    fi18.append(np.cos(interior_coseno9*i))
    fi19.append(np.sin(interior_coseno9*i))
    fi20.append(np.cos(interior_coseno10*i))
    fi21.append(np.sin(interior_coseno10*i))
    fi22.append(np.cos(interior_coseno11*i))
    fi23.append(np.sin(interior_coseno11*i))
    fi24.append(np.cos(interior_coseno12*i))
    fi25.append(np.sin(interior_coseno12*i))

fis = fis[:-1]
fis.append(fi6)
fis.append(fi7)
fis.append(fi8)
fis.append(fi9)
fis.append(fi10)
fis.append(fi11)
fis.append(fi12)
fis.append(fi13)
fis.append(fi14)
fis.append(fi15)
fis.append(fi16)
fis.append(fi17)
fis.append(fi18)
fis.append(fi19)
fis.append(fi20)
fis.append(fi21)
fis.append(fi22)
fis.append(fi23)
fis.append(fi24)
fis.append(fi25)
fis.append(alturas)

pseudo_productos = calculador_pseudo_productos(fis)

matriz_fi = armador_matriz_fis(pseudo_productos,25)
matriz_f = armador_matriz_f(pseudo_productos,25)

matriz_c = np.linalg.solve(matriz_fi,matriz_f)

cs = np.array(matriz_c)

f = []
x = []
for i in range(0,8760,1):
    x.append(i)
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i) + cs[9]*np.cos(interior_coseno5*i) + cs[10]*np.sin(interior_coseno5*i) + cs[11]*np.cos(interior_coseno6*i) + cs[12]*np.sin(interior_coseno6*i) + cs[13]*np.cos(interior_coseno7*i) + cs[14]*np.sin(interior_coseno7*i) + cs[15]*np.cos(interior_coseno8*i) + cs[16]*np.sin(interior_coseno8*i) + cs[17]*np.cos(interior_coseno9*i) + cs[18]*np.sin(interior_coseno9*i) + cs[19]*np.cos(interior_coseno10*i) + cs[20]*np.sin(interior_coseno10*i) + cs[21]*np.cos(interior_coseno11*i) + cs[22]*np.sin(interior_coseno11*i) + cs[23]*np.cos(interior_coseno12*i) + cs[24]*np.sin(interior_coseno12*i))

f = np.squeeze(f)

semana_x = x[:168]
semana_f = f[:168]

np_f = np.array(f)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)
plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(semana_x,semana_f)
plt.show()

ruta_del_archivo = "CO-OPS_8452660_met(2023).csv"
data = pd.read_csv(ruta_del_archivo)

alturas = data['Verified (m)']
tiempos = data['Time (GMT)']
tiempos1 = np.linspace(0,len(tiempos),len(tiempos))
print(len(tiempos1))
plt.clf()
plt.plot(tiempos1, alturas)
plt.show()

f = [] 
for i in np.arange(0,168,0.1):
    f.append(cs[0] + cs[1]*np.cos(interior_coseno1*i) + cs[2]*np.sin(interior_coseno1*i) + cs[3]*np.cos(interior_coseno2*i) + cs[4]*np.sin(interior_coseno2*i) + cs[5]*np.cos(interior_coseno3*i) + cs[6]*np.sin(interior_coseno3*i) + cs[7]*np.cos(interior_coseno4*i) + cs[8]*np.sin(interior_coseno4*i) + cs[9]*np.cos(interior_coseno5*i) + cs[10]*np.sin(interior_coseno5*i) + cs[11]*np.cos(interior_coseno6*i) + cs[12]*np.sin(interior_coseno6*i) + cs[13]*np.cos(interior_coseno7*i) + cs[14]*np.sin(interior_coseno7*i) + cs[15]*np.cos(interior_coseno8*i) + cs[16]*np.sin(interior_coseno8*i) + cs[17]*np.cos(interior_coseno9*i) + cs[18]*np.sin(interior_coseno9*i) + cs[19]*np.cos(interior_coseno10*i) + cs[20]*np.sin(interior_coseno10*i) + cs[21]*np.cos(interior_coseno11*i) + cs[22]*np.sin(interior_coseno11*i) + cs[23]*np.cos(interior_coseno12*i) + cs[24]*np.sin(interior_coseno12*i))

x = np.linspace(0,1680,1680)
f = np.squeeze(f)


semana_alturas = alturas
semana_tiempos = tiempos1

np_f = np.array(f)
np_alturas = np.array(alturas)
ECM = np.sqrt(((np_alturas - np_f)**2).mean())
print(ECM)

plt.clf()
plt.plot(semana_tiempos,semana_alturas, color="red")
plt.plot(x,f)
plt.show()