#!/usr/bin/env python
# coding: utf-8

# #Reporte2 nombre:Ariana Sato Flores código:20192304
# 
# El autor busca responder si se puede modelar la política monetaria no convencional, de forma que se extienda el modelo estándar de demanda y oferta agregada de economía cerrada a modo de innovación de las políticas monetarias aplicadas para contrarrestar las secuelas que trajo la crisis internacional del 2008-2009. A través de ello, también se querrá demostrar que los antiguos modelos y métodos sirven para manejar los problemas macroeconómicos contemporáneos.
# La adaptación de una mirada más teórica y tradicional también puede dar grandes aportes para enfrentar problemas económicos actuales. Traer a colación a Keynes (1936) y el keynesianismo de Hicks (1937) ayuda a la incorporación de innovaciones a las políticas monetarias americanas de manera más afable, sencillo y sobre todo persuasivo que por ser más tradicional no quita su protagonismo y eficacia al resolver problemas de una forma más concluyente que la macroeconomía moderna convencional. 
# 
# El autor complementa su aporte a la investigación con a aplicación de ecuaciones lineales para explicar el mercado de bienes y así simplificar el mercado monetario como el mercado de bonos a largo plazo para que el entendimiento del lector sobre el propósito de apelar a instrumentos no convencionales de política mentaría que desarrolla a través del artículo.
# El uso de gráfico es necesario para el entendimiento de la aplicación de las ecuaciones, como por ejemplo la curva IS y LM. Además, también se hace una explicación de los gráficos y sus pendientes positiva o negativa. Considero importante una buena interpretación de las expresiones matemáticas al contexto en el que se está explicando el fenómeno, si el desarrollo queda solamente en números, no se entendería el propósito del autor.  
# 
# El análisis de las acciones de la Fed al administrar la tasa de interés a corto plazo y la flexibilización cuantitativa por la compra de valores a largo plazo, permiten hablar de una demanda agregada de la economía y aquello es conjugado con una curva de oferta agregada convencional que genera un modelo macroeconómico que nos ayuda a determinar los valores de equilibrio de producción, tasa de interés de largo plazo y el nivel de precios. A partir de ello, se puede ahondar en los efectos de las políticas no convencionales y además efectos de la política fiscal y choques adversos de la oferta.
# El autor brinda definiciones analíticas y enlaza la demanda y la oferta agregada para analizar los efectos de las variables endógenas del modelo de política monetaria convencional. En adición, mejora el entendimiento con modelos dinámicos sencillos, hace una explicación de la ecuación paso por paso y la deriva de manera entendible al lector.
# 
# A través de la búsqueda de otros autores que complementen  la investigación de este artículo, Oyola de los Ríos (2020) también aplica soluciones a problemas de Oferta y Demanda en su aplicación dentro de la economía con ecuaciones ordinarias , dentro de su investigación también le da importancia a la herramienta matemática para obtener su solución analítica, gráfica y numérica. 
# 
# 
# 

# *CÓDIGO*
# 
# Ariana Sato Flores 20192304
# Claudia Andrea Zevallos Amaya 20191511

# Pregunta 1:
#     
# a) Encuentre las ecuaciones de Ingreso  y tasa de interés de equilibrio:
# 
# 
# Recordemos las ecuaciones IS y LM:
# Donde:
#     r=(Co+Io+Xo)/h-((1-(b-m)*(1-t))/h)*Y
#     
#    LM=
#    r=-(1/j)*(Ms/P)+(k/j)*Y
#     
# Debemos encontrar "Y" para reemplazarlo en las ecuaciones, por lo que iniciaremos a derivar de la siguiente manera:
#     
# A manera de simplificación y para despejar la "r" , hacemos la siguiente ecuación igualando:
# 
# *(Co+Io+Xo)/h-((1-(b-m)*(1-t))/h)*Y
# 
# Factorizamos Y:
# *Ecuación de equilibrio*
# 
# Ye=(j*(Co+Io+Go+Xo)/(h*k+j*(1-(b-m)
# 
# Factorizamos "Y" 
# 
# re=(k*(Co+Io+Go+Xo)/(h*k+j*(1-(b-m)
# 
# 
#     

# Gráfico del punto de equilibrio

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
import numpy as np
import random
import math
import sklearn
import scipy as sp
import networkx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from causalgraphicalmodels import CausalGraphicalModel


# In[12]:


#--------------------------------------------------
    # Curva IS

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo)/h - ( ( 1-(b-m)*(1-t) ) / h)*Y  
    return r_IS

r_is = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # Curva LM 

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación LM

def r_LM(k, j, Ms, P, Y):
    r_LM = - (1/j)*(Ms/P) + (k/j)*Y
    return r_LM

r_lm = r_LM( k, j, Ms, P, Y)

# Gráfico del modelo IS-LM

# Dimensiones del gráfico
y_max = np.max(r_lm)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
# Curva IS
ax.plot(Y, r_is, label = "IS", color = "#0000EE") #IS
# Curva LM
ax.plot(Y, r_lm, label="LM", color = "#66CDAA")  #LM

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Texto y figuras agregadas
# Graficar la linea horizontal - r
plt.axvline(x=51.5,  ymin= 0, ymax= 0.52, linestyle = ":", color = "green")
# Grafica la linea vertical - Y
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "green")

# Plotear los textos 
plt.text(49,100, '$E_0$', fontsize = 14, color = 'red')
plt.text(0,100, '$r_0$', fontsize = 12, color = 'red')
plt.text(53,-10, '$Y_0$', fontsize = 12, color = 'red')

#Adicional texto para el grafico
plt.text(100,20, '$LM$', fontsize = 15, color = 'red')
plt.text(100,190, '$IS$', fontsize = 15, color = 'red')
plt.text(80,100, '$II$', fontsize = 20, color = 'red')
plt.text(15,100, '$IV$', fontsize = 20, color = 'red')
plt.text(48,145, '$I$', fontsize = 20, color = 'red')
plt.text(48,45, '$III$', fontsize = 20, color = 'red')

# Título, ejes y leyenda
ax.set(title="IS-LM Model", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# Pregunta 2: Estática Comparativa:
#   
# a) Analice los efectos sobre las variables endógenas Y, r de una disminución del gasto fiscal. 
# 
# Ecuaciones a recordar:
# 
# Nuestro análisis intuitivo parte de:
# 
# -Mercado de Bienes = Go↓ →DA<Y→Y↓
# -Mercado de Dinero= Y↓ →Md↓Md<Ms→r↓
# 
# Vemos como ambas variables endógenas se reducen
# 
# El análisis matemático parte de:

# In[ ]:


# nombrar variables como símbolos
Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols ('Co Io Go Xo h r b m t beta_0, beta_1')

# nombrar variables como símbolos
k, j, Ms, P, Y = symbols('k j Ms P Y')

# Beta_0 y beta_1
beta_0 = (Co + Io - Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )

# Producto de equilibrio y la tasa de interes de equilibrio en el modelo IS-LM
Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[ ]:


df_Y_eq_Go = diff(Y_eq, Go)
print("El Diferencial del Producto con respecto al diferencial del gasto autonomo = ", df_Y_eq_Go)  # este diferencial es positivo


# In[ ]:


El Diferencial del Producto con respecto al diferencial del gasto autonomo =  -k/(h*k + j*(-(1 - t)*(b - m) + 1))
ES POSITVA


# In[ ]:


df_r_eq_Go = diff(r_eq, Go)
print("El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo = ", df_r_eq_Go)  # este diferencial es positivo


# El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo =  -j/(h*k + j*(-(1 - t)*(b - m) + 1))
# ES POSITIVA

# Gráficos

# In[21]:


#1--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.4
m = 0.5
t = 0.8

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#2--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[19]:


#--------------------------------------------------
    # NUEVA curva IS: reducción Gasto de Gobienro (Go)
    
# Definir SOLO el parámetro cambiado
Go = 20

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[22]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS_(G_0)", color = "green") #IS_orginal
ax.plot(Y, r_G, label = "IS_(G_1)", color = "green", linestyle = 'dashed') #IS_modificada

ax.plot(Y, i, label="LM", color = "C0")  #LM_original

# Texto y figuras agregadas
plt.axvline(x=51.5,  ymin= 0, ymax= 0.57, linestyle = ":", color = "green")
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "green")

plt.axvline(x=63,  ymin= 0, ymax= 0.57, linestyle = ":", color = "green")
plt.axhline(y=115, xmin= 0, xmax= 0.6, linestyle = ":", color = "green")
plt.text(60,49, '$E_1$', fontsize = 14, color = 'black')

plt.text(49,100, '$E_0$', fontsize = 14, color = 'red')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'red')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'red')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#008B8B')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#008B8B')
plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#008B8B')

#plt.text(69, 115, '→', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Politica Fiscal Contractiva", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# Como análisis intuititivo parte de
# 
# Dentro del Mercado de dinero:
# 
#                 Mso↓ →Ms>Md→r↑
# Dentro del Mercado de bienes afecta pues nuestro interés aumentó
# 
#                r↑= I↓ = DA↓ → DA<Y = Y↓  → DA=Y
# Debemos dismunir Y para poder encontrar un equilibrio

# El análisis matemático sería de la siguiente manera

# In[ ]:


Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

k, j, Ms, P, Y = symbols('k j Ms P Y')

beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )


Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[ ]:


df_Y_eq_Ms = diff(Y_eq, Ms)
print("El Diferencial del Producto con respecto al diferencial de la masa monetaria = ", df_Y_eq_Ms)


# El Diferencial del Producto con respecto al diferencial del gasto autonomo =  -(-(1 - t)*(b - m) + 1)/(P*(h*k + j*(-(1 - t)*(b - m) + 1)))
# ES POSITIVO

# In[ ]:


df_r_eq_Ms = diff(r_eq, Ms)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_Ms) 


# El Diferencial de la tasa de interes con respecto al diferencial del gasto autonomo =  h/(P*(h*k + j*(-(1 - t)*(b - m) + 1)))
# 
# ES NEGATIVO

# #Gráficos

# In[23]:


Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)



def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)



# Curva LM ORIGINAL


Y_size = 100

k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)



def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[25]:


#Nueva curva de disminución de la Masa Monetaria

Ms = 50

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_ejer = i_LM( k, j, Ms, P, Y)


# In[26]:


y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))


ax.plot(Y, r, label="IS", color = "orange")  #LM_original

ax.plot(Y, i, label = "LM_(G_0)", color = "#E9967A") #IS_orginal
ax.plot(Y, i_ejer, label = "LM_(G_1)", color = "#CD5C5C") #IS_modificada

# Texto y figuras agregadas
plt.axvline(x=50,  ymin= 0, ymax= 0.50, linestyle = ":", color = "grey")
plt.axhline(y=100, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

plt.axvline(x=40,  ymin= 0, ymax= 0.42, linestyle = ":", color = "grey")
plt.axhline(y=70, xmin= 0, xmax= 0.42, linestyle = ":", color = "grey")
plt.text(60,49, '$E_1$', fontsize = 14, color = 'black')

plt.text(50,50, '$E_0$', fontsize = 14, color = 'red')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'red')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'red')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#008B8B')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#008B8B')
plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#008B8B')

#plt.text(69, 115, '→', fontsize=15, color='grey')
#plt.text(69, 52, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title="Politica monetaria contractiva de la masa monetaria", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()


# -Analice los efectos sobre las variables endógenas Y, r de un incremento de la tasa de impuestos. . El análisis debe ser intuitivo, matemático y gráfico.

# Como el análisis intuitivo para una política fiscal de incremento de tasa de impuesto:
# 
# -Mercado de bienes
# 
#    t↑ = Co↓ = DA↓ → DA<Y = Y↓  → DA=Y
# -Mercado de dinero:
# 
#    Y↓ →Md↓Md<Ms→r↓
# Vemos que nuestras ambas variables endógenas se reducen

# Ahora matemáticamente vemos que

# In[ ]:


Co, Io, Go, Xo, h, r, b, m, t, beta_0, beta_1  = symbols('Co Io Go Xo h r b m t beta_0, beta_1')

k, j, Ms, P, Y = symbols('k j Ms P Y')

beta_0 = (Co + Io + Go + Xo)
beta_1 = ( 1-(b-m)*(1-t) )


Y_eq = (k*beta_0)/(k*h + j*beta_1) - ( beta_1 / (k*h + j*beta_1) )*(Ms/P)
r_eq = (j*beta_0)/(k*h + j*beta_1) + ( h / (k*h + j*beta_1) )*(Ms/P)


# In[ ]:


df_Y_eq_t = diff(Y_eq, t)
print("El Diferencial del Producto con respecto incremento de la tasa de impuestos = ", df_Y_eq_t)


# El Diferencial del Producto con respecto incremento de la tasa de impuestos =  Ms*j*(b - m)*(-(1 - t)*(b - m) + 1)/(P*(h*k + j*(-(1 - t)*(b - m) + 1))**2) - Ms*(b - m)/(P*(h*k + j*(-(1 - t)*(b - m) + 1))) - j*k*(b - m)*(Co + Go + Io + Xo)/(h*k + j*(-(1 - t)*(b - m) + 1))**2
# 
# ES NEGATIVO

# In[ ]:


df_r_eq_t = diff(r_eq, t)
print("El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria = ", df_r_eq_t) 


# El Diferencial de la tasa de interes con respecto al diferencial de la masa monetaria =  -Ms*h*j*(b - m)/(P*(h*k + j*(-(1 - t)*(b - m) + 1))**2) - j**2*(b - m)*(Co + Go + Io + Xo)/(h*k + j*(-(1 - t)*(b - m) + 1))**2
# ES POSITIVO

# #Gráficos 

# In[27]:


Y_size = 100 

Co = 35
Io = 40
Go = 50
Xo = 2
h = 0.8
b = 0.5
m = 0.4
t = 0.8

Y = np.arange(Y_size)



def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# Curva LM ORIGINAL

Y_size = 100
k = 2
j = 1                
Ms = 200             
P  = 20               

Y = np.arange(Y_size)



def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[28]:


#Aumento a la tasa de impuesto
t = 1.90

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_TR = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[29]:


y_max = np.max(i)
fig, ax = plt.subplots(figsize=(10, 8))


ax.plot(Y, r, label = "IS_(G_0)", color = "#7D3C98") 
ax.plot(Y, r_TR, label = "IS_(G_1)", color = "#C491E5", linestyle = 'dashed') 

ax.plot(Y, i, label="LM", color = "#EE3B87")  

plt.axvline(x=51.5,  ymin= 0, ymax= 0.57, linestyle = ":", color = "grey")
plt.axhline(y=93, xmin= 0, xmax= 0.52, linestyle = ":", color = "grey")

plt.axvline(x=40,  ymin= 0, ymax= 0.42, linestyle = ":", color = "grey")
plt.axhline(y=70, xmin= 0, xmax= 0.42, linestyle = ":", color = "grey")
plt.text(60,49, '$E_1$', fontsize = 14, color = 'black')

plt.text(49,100, '$E_0$', fontsize = 14, color = 'red')
plt.text(-1,100, '$r_0$', fontsize = 12, color = 'red')
plt.text(53,-40, '$Y_0$', fontsize = 12, color = 'red')
#plt.text(50,52, '$E_1$', fontsize = 14, color = '#008B8B')
#plt.text(-1,72, '$r_1$', fontsize = 12, color = '#008B8B')
plt.text(47,-40, '$Y_1$', fontsize = 12, color = '#008B8B')


ax.set(title="Politica Fiscal Contractiva de la tasa de impuesto", xlabel= r'Y', ylabel= r'r')
ax.legend()

plt.show()

