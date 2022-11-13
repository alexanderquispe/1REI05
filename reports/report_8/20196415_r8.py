#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install causalgraphicalmodels 


# In[2]:


pip install linearmodels


# In[3]:


from causalgraphicalmodels import CausalGraphicalModel
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from itertools import combinations 
#import plotnine as p
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels.iv.model as lm
from linearmodels.iv import IV2SLS
from statsmodels.iolib.summary2 import summary_col
import seaborn as sns 


# ## REPORTE 8 (345)

# - Julia Millen Massa Coronel (20196415)

# ###### Explique en qué consiste el supuesto de "Relevance"

# Se basa en la idea de que las variables instrumentales no van a funcionar si es que la parte de X, que es el tratamiento, no se explica por Z que es el instrumento; es decir, cuando la correlación es cero. Si bien en el mundo real casi no hay correlaciones que lleguen a 0, lo que se pide es que las estimaciones sean "grandes y oscilantes".Esa suposición de que X y Z se puede verificar con facilidad cuando miramos la relación entre ambas. Mientras más fuerte sea, más certera es la relevancia y menos saltará la estimación de una muestra a la otra. 

# #### Explique en qué consiste el supuesto de "Validez"

# Este supuesto implica que el instrumento Z es una variable que no deja cabos sueltos. Es decir, cualquier camino entre el instrumento Z y el resultado Y debe pasar por el tratamiento X o estar cerrado. Las variables instrumentales no eximen que nosotros cerremos las puertas , pero trasladan esa labor del tratamiento al instrumento, con lo que es más fácil cerrar las puertas para dicho instrumento. 

# In[4]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def read_data(file): 
    return pd.read_stata("https://raw.github.com/scunning1975/mixtape/master/" + file)

card = read_data("card.dta")
card.head()


# ##### Ploteo de Histogramas

# In[12]:


sns.displot(card, x="educ", col="nearc4")


# In[11]:


sns.displot(card, x="lwage", col="nearc4")


# ##### Regresiones

# In[14]:


ols_reg = sm.OLS.from_formula("lwage ~ educ + exper + black + south + married + smsa", 
              data = card).fit()

ols_reg.summary()


# In[15]:


iv_reg = IV2SLS.from_formula("lwage ~  1 + exper + black + south + married + smsa + [educ ~ nearc2 ]", card).fit()
iv_reg.summary


# - ¿ Cambió el coeficiente relacionado con educ?, ¿por qué?

# El coeficiente relacionado con educación en efecto aumenta a 0.3566, porque se han cerrado las puertas a otras posibles variables que afectaban o que inlfuían en la variable analizada. 

# ## LECTURA

# El presente trabajo a grandes rasgos se crea a raíz del análisis del papel de las instituciones históricas en la explicaicón del subdesarrollo contemporaneo. En concreto lo que se busca saber es ¿cuál es el impacto a largo plazo de la mita (un sistema extensivo de trabajo minero forzoso vigente en Perú y Bolivia entre 1573 y 1812)? a través de la discontinuidad de la regresión. 
# 
# En general el artículo no presenta debilidades a mi parecer. Sin embargo, cuando se examina la etnicidad me parece un poco sesgado que se defina como indigena si el idioma principal que se habla en el hogar es una legnua indigena. Ahora es bien sabido que muchas personas por sus raices o su arbol familiar pueden ser catalogadas como indigenas a pesar de no hablar quechua u otra lengua. 
# 
# En general, la contribución del articulo son sus resultados. Estos revelan que la mita reduce el consumo de los hogares en torno al 25% y aumenta la prevalencia del retraso en el crecimiento de los niños en torno a 6 puntos porcentuales en los distritos sometidos en la actualidad. Además, se comprueba que redujo la educación históricamente, y hoy los distritos de mita siguen estando menos integrados en las redes de carreteras. Sumado a eso, vemos que los datos del censo agrícola más reciente demuestran que el efecto de la mita a largo plazo aumenta la prevalencia de la agricultura de subsistencia.
# 
# Para avanzar en la pregunta sería interesante analizar es por qué en los lugares en los que se vió la presencia de la institución de la mita hubo en promedio mucha menos migración que en los lugares en los que no hubo. Otra idea en la que se podría continuar la investigación es analizar  estos modelos de evolución insitucional  para ver cómo influyen en esas limitaciones las fuerzas que promueven el cambio. 

# In[ ]:




