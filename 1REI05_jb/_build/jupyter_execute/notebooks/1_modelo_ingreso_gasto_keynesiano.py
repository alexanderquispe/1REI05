#!/usr/bin/env python
# coding: utf-8

# # Apuntes de Macroeconomía Intermedia: El modelo de Ingreso-Gasto Keynesiano

# ## Introducción

# El modelo es también conocido como modelo de cruz keynesiana, en alusión a John M. Keynes, autor de la Teoría General de la ocupación, el interés y el dinero (1936), escrito y publicado durante la depresión de los años 1930’s.
# 
# El modelo supone que el **nivel de precios está fijo**, que el **nivel del producto se adapta a los cambios en la demanda agregada** y que **la tasa de interés está determinada fuera del 
# modelo** (en el mercado monetario). Es un **modelo de corto plazo**.
# 

# ## El modelo de Ingreso-Gasto Keynesiano

# ### Las ecuaciones del modelo de Ingreso - Gasto y la función de Demanda Agregada
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=G_{0}$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda  | $DA=C+I+G+X-M$ |
# 
# 
# Entonces, haciendo reemplazos y algunas operaciones algebraicas, se obtiene la función de Demanda Agregada siguiente:
# 
# $$ DA = C_{0} + bY^{d} + I_{0} - hr + G_{0} + X_{0} - mY^{d} $$ 
# 
# $$ DA = (b-m)(1-t)Y + C_{0} + I_{0} - hr + G_{0} + X_{0} $$ 
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# 
# En forma breve:  $ DA =\alpha_{0} + \alpha_{1}Y $,
# 
# Donde $ \alpha_{0} = (C_{0} + I_{0} + G_{0} + X_{0} - hr) $ es el intercepto y  $ \alpha_{1} = (b-m)(1-t) $ es la pendiente de la función.

# ### La determinación del Ingreso de equilibrio
# 
# 
# En equilibrio, el ingreso agregado, Y, debe ser igual a la demanda agregada $DA$. De la condición de equilibrio $Y = DA$, se obtiene la ecuación del ingreso o producto de equilibrio a corto plazo. 
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# $$ Y = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# $$ [1 - (b-m)(1-t)]Y = C_{0} + I_{0} + G_{0} + X_{0} - hr $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t)]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# A corto plazo se supone, por ahora, que los precios son fijos y que la economía se encuentra por debajo de la producción de pleno empleo. También se puede decir que la oferta agregada es infinitamente elástica al nivel de precios dado. En estas condiciones, el producto e ingreso de equilibrio está determinado por la demanda agregada o por sus componentes. El ajuste en el mercado de bienes es por cantidades y no por precios
# 
# <center>
# <figure>
#     <figcaption><center><b>El ingreso de Equilibrio a Corto Plazo</b></center></figcaption>
# <img title="a title" src="../figs/1.png">
# <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 54.</figcaption>
# </figure>
# </center>

# ### La estabilidad del equilibrio
# 
# En ausencia de equilibrio, la $DA$ puede ser mayor o igual que el ingreso o producto. Estas situaciones de desequilibrio desaparecen. 
#       
# El modelo es estable porque hay convergencia al equilibrio. 
#       
# Los aumentos de la demanda agregada ($DA$) que dan lugar a excesos de demanda, provocan la caída de los inventarios o existencias de las empresas y, por lo tanto, la decisión de las empresas de aumentar su producción en un monto similar al aumento de la demanda. 
#       
# En otras palabras, los excesos de demanda se enfrentan con aumentos de la producción. Cada vez que aumenta la producción, aumenta el empleo, disminuye la tasa de desempleo y aumenta el ingreso disponible de las familias. Las familias distribuyen el incremento de su ingreso disponible entre consumo de bienes y servicios, y ahorro. 
# 
# Lo contrario ocurre cuando se produce un déficit de demanda o exceso de producción. 
# 
# En este caso, aumentan los inventarios y las empresas deciden disminuir su producción y el empleo de trabajadores. En ambos casos, la oferta de producción se ajusta a los cambios en la demanda agregada.
# 
# En el gráfico siguiente se muestra el proceso de convergencia hacia el punto de equilibrio “$E$”, partiendo de un punto “$A$” (exceso de demanda) o del punto “$B$” (exceso de producción o déficit de demanda). 
# 
# En el punto “$A$”, la demanda agregada es mayor que el nivel de producción. Hay exceso de demanda que provoca una disminución de los inventarios y aumentos de la producción hasta llegar al equilibrio. De otro lado, en el punto “$B$” hay un déficit de demanda o un exceso de producción que da lugar a un aumento de los inventarios y a disminuciones de la producción hasta llegar al equilibrio.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Determinación y Estabilidad del equilibrio</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/2.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 55.</figcaption>
# </figure>
# </center>

# ### El multiplicador Keynesiano
# 
# Un cambio en la magnitud de cualquiera de los componentes autónomos de la demanda agregada (que conforman el intercepto), genera un proceso multiplicador del ingreso hasta converger al nuevo ingreso y producto de equilibrio. Así, el aumento multiplicado del ingreso es resultado de los efectos directos e indirectos ocasionados por el aumento de cualquiera de los componentes autónomos de la demanda agregada.
# 
# De la ecuación de la demanda agregada, $DA = \alpha_{0} + \alpha_{1}Y$ , y de la condición de equilibrio $Y = DA$ , se obtiene $(1-\alpha_{1})Y = \alpha_{0}$. Por lo tanto, el multiplicador es: 
# 
# $$ \dfrac{1}{1-\alpha_{1}} = \dfrac{1}{1-(b-m)(1-t)} $$
# 
# Un cambio en $\alpha_{0}$ provoca un cambio multiplicado en $Y$.
# 
# $$ \Delta Y = \dfrac{1}{1-\alpha_{1}} \Delta \alpha_{0} $$
# 
# Nótese que el tamaño de este multiplicador depende de la magnitud de los parámetros “$b$”, “$m$” y “$t$”. Las filtraciones de demanda son las que reducen su tamaño: estas se expresan en la propensión marginal a ahorrar, la propensión marginal a importar y la tasa promedio de impuestos.
# 
# Keynes desarrolla el concepto de multiplicador de inversión partiendo de una economía cerrada y sin gobierno, y con dos sectores de producción: uno de bienes de Inversión y el otro de bienes de Consumo. 
# 
# “Llamemos a $k$ el multiplicador de inversión - dice Keynes. Éste nos indica que, cuando existe un incremento en la inversión total, el ingreso aumentará en una cantidad que es $k$ veces el incremento de la inversión” (Keynes, p. 108). 
# 
# Si la propensión marginal a consumir es “$0<b<1$”, el multiplicador que tiene en mente Keynes es $k = \dfrac{1}{1-b} $.
# 
# Este multiplicador es resultado de un proceso que empieza con el aumento de la demanda de inversión, $\Delta I$, que genera un incremento en la producción e ingreso en el sector que produce bienes de inversión, igual a: $\Delta I = \Delta Y_{1}$.
# 
# Este incremento del ingreso, genera un aumento de la demanda de consumo y, por lo tanto, de la producción e ingreso en el sector que produce bienes de consumo, igual a: $\Delta Y_{2} = b\Delta Y_{1} $.
# 
# Este segundo incremento del ingreso, aumenta la demanda de bienes de consumo y, por consiguiente, de la producción e ingreso en el sector que produce bienes de consumo, igual a: $\Delta Y_{3} = b\Delta Y_{2} $. 
# 
# El proceso continúa como se describe a continuación:
# 
# $$\Delta I = \Delta Y_{1}$$ 
# $$\Delta Y_{2} = b\Delta Y_{1} = b\Delta I $$
# $$\Delta Y_{3} = b\Delta Y_{2} = b^{2}\Delta I $$
# $$\Delta Y_{4} = b\Delta Y_{3} = b^{3}\Delta I $$
# $$...$$
# $$\Delta Y_{n} =  b^{n-1}\Delta I $$
# 
# En consecuencia, el incremento total del ingreso será igual a:
# 
# $${\displaystyle \sum_{i=1}^{n}\Delta Y_{i}=\Delta I+b\Delta I+b^{2}\Delta I+...+b^{n-1}\Delta I}$$
# 
# $$\Delta Y = (1+b+b^{2}+...+b^{n-1})\Delta I$$
# 
# El factor entre paréntesis es la suma de los términos de una progresión geométrica que es igual a: $\dfrac{1-b^{n}}{1-b}$. Como $0<b<1$, cuando “$n$” tiende a infinito, entonces $b^{n} = 0$. Por lo tanto, el ingreso aumentará en una cantidad que es $k$ veces el incremento de la inversión, es decir:
# 
# $$ \Delta Y = (\dfrac{1}{1-b})\Delta I $$
#     
# $$ \Delta Y = k\Delta I $$
# 
# A modo de ejemplo supongamos el siguiente modelo de una economía cerrada y sin gobierno, con una función consumo igual a: $C = 100 + 0.75Y$ y una inversión autónoma igual a: $I_{0} = 200$.
# 
# En este caso el ingreso de equilibrio es 1200 y el multiplicador es 4. Con este valor del ingreso de equilibrio se obtiene un consumo igual a 1000. 
# 
# Por último, si la inversión aumenta en 100, el ingreso de equilibrio aumentaría 4 veces más.
# 
# En esta economía el ingreso de equilibrio está dado por:
# 
# $$Y=(\dfrac{1}{1-0.75})(C_{0}+I_{0})$$
# $$Y=(\dfrac{1}{0.25})(C_{0}+I_{0})$$
# $$\Delta Y = 4\Delta I_{0}$$
# 
# 
# Con una propensión marginal a consumir igual a $b = 0.75$, la propensión marginal a ahorrar es igual a $s = 0.25$.
# 
# Cuanto más alta es la propensión marginal a consumir (o cuanto más baja es la propensión marginal a ahorrar), mayor es el multiplicador.
# 
# Por ejemplo, supongamos que la propensión marginal a consumir es igual a $b=0.8$. En este caso el multiplicador será igual a 5, mayor que cuando $b=0.75$. La propensión marginal a ahorrar disminuye a 0.20. Con una propensión marginal a consumir igual a 0.6 (propensión marginal a ahorrar igual a 0.4), el multiplicador se reduce a 2.5. Además, gráficamente se puede observar que el ingreso de equilibrio disminuye de 1500 a 750. 
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos del cambio en la Propensión Marginal a Consumir</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/3.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 58.</figcaption>
# </figure>
# </center>

# ## Estática comparativa en el modelo de Ingreso-Gasto keynesiano
# 
# Para una comprensión gráfica del análisis de estática comparativa, es importante describir el sentido de los cambios en la posición de la función de $DA$ en el plano ($Y$, $DA$), cuando se produce una modificación en las magnitudes de su intercepto o de su pendiente.
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# El intercepto de la función de $DA$ esta influído por los valores de los gastos autónomos y la tasa de interés. El aumento de cualquier gasto autónomo desplaza a la función de $DA$ hacia arriba en forma paralela. Lo contrario ocurre si algún gasto autónomo disminuye. La función de $DA$ mantiene su pendiente. 
# 
# Todos los desplazamientos hacia arriba de la función de $DA$, cortan a la línea de $45_{0}$ en niveles de ingreso de equilibrio más altos. Todos los puntos de la línea de $45_{0}$ son de equilibrio $DA=Y$. Con los desplazamientos hacia abajo ocurre lo contrario; los niveles de ingreso de equilibrio se reducen. Por ejemplo, un aumento en la inversión autónoma tiene una efecto expansivo sobre el producto o ingreso.
# 
# De otro lado, entre las magnitudes del intercepto y de la tasa de interés real, hay una relación inversa. Un aumento de la tasa de interés tiene un efecto contractivo sobre el producto porque desplaza a la función de $DA$ hacia abajo.
# 
# Por último, el tamaño de la pendiente de la función de $DA$ depende de la magnitud de los parámetros “$b$”, “$m$” y “$t$”. 
# 
# Por ejemplo, un aumento de la propensión marginal a consumir, aumenta su pendiente. En este caso la recta de la $DA$ gira en el sentido contrario a la rotación de las agujas del reloj, manteniendo su intercepto constante. Asimismo, un aumento en la propensión a importar (o en la tasa de impuestos), reduce su pendiente con lo cual la recta de la $DA$ gira en el sentido de las agujas del reloj. \item El aumento de la propensión marginal a importar, disminuye el producto de equilibrio.

# ### Política Fiscal Expansiva con aumento del Gasto del Gobierno
# 
# Dado el siguiente modelo de Ingreso-Gasto:
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=G_{0}$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda  | $DA=C+I+G+X-M$ |
# 
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t)]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# 
# Supongamos que el gobierno decide aumentar su gasto $(\Delta G > 0)$ para expandir la producción y el empleo en la economía. Las variables exógenas que no cambian son: la tasa de interés real, el consumo, la inversión y las exportaciones autónomas. También se supone constantes la presión tributaria y las propensiones marginales a consumir e importar. 
# 
# Este aumento del gasto del gobierno hace que se eleve la magnitud del intercepto de la función de la DA, razón por lo cual se desplaza hacia arriba. 
# 
# Esta es una política fiscal expansiva porque al aumentar el gasto y, por lo tanto, la demanda agregada, aumenta el ingreso de equilibrio.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos de una expansión del Gasto Público</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/4.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 60.</figcaption>
# </figure>
# </center>
# 
# Matemáticamente, se puede mostrar que el incremento del ingreso es igual al multiplicador por el incremento del gasto. En efecto:
# 
# $$ \partial Y = \dfrac{1}{[1 - (b-m)(1-t)]} (\partial C_{0} + \partial I_{0} + \partial G_{0} + \partial X_{0} - h\partial r) $$ 
# 
# Puesto que $ \partial C_{0} = \partial I_{0} = \partial X_{0} = \partial r = 0$ y $\partial G_{0} > 0$, entonces:
# 
# $$ \partial Y = \dfrac{1}{[1 - (b-m)(1-t)]} (\partial G) $$ 
# 
# El multiplicador del gasto es mayor que cero:
# 
# $$\dfrac{\partial Y}{\partial G} = \dfrac{1}{[1 - (b-m)(1-t)]} > 0 $$

# ### Política fiscal expansiva con una reducción de la Tasa de Tributación
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=G_{0}$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda  | $DA=C+I+G+X-M$ |
# 
# 
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + (b-m)(1-t)Y $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t)]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# 
# La política fiscal consistente en una reducción de la tasa de impuestos es expansiva, en el sentido que incrementa el ingreso de equilibrio, porque incrementa el ingreso disponible y, por lo tanto, la demanda de consumo y consecuentemente la demanda agregada.
# 
# La tributación antes de la aplicación de la política fiscal es igual a $T_{0} = t_{0}Y$ . Cuando se reduce la tasa de impuestos, $(t_{1} < t_{0})$, la ecuación de la tributación será: $T_{1} = t_{1}Y$. La reducción de la tasa de impuestos da lugar a un aumento tanto de la pendiente de la $DA$ como del valor del multiplicador.
# 
# Gráficamente la *política fiscal expansiva* consistente en una disminución de la tasa de tributación, incrementa la pendiente de la $DA$. La función de demanda $DA$ gira en sentido contrario a la rotación de las agujas del reloj, aumentando así el ingreso de equilibrio. 
# 
# Hay una relación inversa entre el multiplicador y la tasa de impuestos:
# 
# $$\dfrac{\partial [1-(b-m)(1-t)]^{-1}} {\partial t} = -\dfrac{b-m}{[1-(b-m)(1-t)]^{2}} < 0 $$
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos de la Disminución de la tasa de tributación $(t)$</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/5.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 63.</figcaption>
# </figure>
# </center>

# ### Política Comercial que aumenta la propensión marginal a importar
# 
# Esta sería una política de fomento de la libre importación.
# 
# Cuando aumenta la propensión marginal a importar se reduce la pendiente de la función de $DA$ y el valor del multiplicador. 
# 
# En el siguiente gráfico, se observa un desplazamiento de la función de $DA$ en el sentido de las agujas del reloj, provocado por un aumento de la propensión marginal a importar. El resultado es una disminución del ingreso de equilibrio.
#  
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos del aumento de la Propensión Marginal a Importar ($m$)</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/6.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 63.</figcaption>
# </figure>
# </center>
# 
# Hay una relación inversa entre el multiplicador y la magnitud de la propensión marginal a importar.
# 
# $$\dfrac{\partial [1-(b-m)(1-t)]^{-1}} {\partial m} = -\dfrac{1-t}{[1-(b-m)(1-t)]^{2}} < 0 $$

# ### Política Fiscal con Regla Contracíclica
# 
# Este tipo de política fiscal hace que los auges y las depresiones no sean muy pronunciados. 
# 
# Se establece como una regla consistente en una variación del Gasto ($G$) en sentido contrario a la variación del Producto, $Y$. 
# 
# Esta regla es igual a: $G = G_{0} - g_{Y}$. 
# 
# Cuando la economía entra en recesión y/o en depresión, disminuye la producción y aumenta el desempleo de la fuerza laboral. Ante esta situación el gobierno incrementa su gasto para estimular la recuperación de la producción. Asimismo, si la economía entra en la fase de expansión y auge, el producto y el empleo aumentan. En esta fase del ciclo, el gobierno tiene que disminuir su gasto y acumular recursos tributarios.
#         
# La política fiscal de gasto contra cíclico afecta la pendiente de la curva de $DA$ y el valor del multiplicador. Ambos disminuyen. Este tipo de política fiscal, entonces, morigera las fluctuaciones del producto o ingreso.
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=G_{0} - gY$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda  | $DA=C+I+G+X-M$ |
# 
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + [(b-m)(1-t)-g]Y $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t) + g]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# Como se puede observar, en las dos ecuaciones anteriores aparece el parámetro “$g$” de la regla contracíclica. 
# 
# En la primera, este parámetro disminuye la magnitud de la pendiente de la función de $DA$. Y, en la segunda, dicho parámetro aparece aumentando la magnitud del denominador del multiplicador y, por lo tanto, reduciendo su tamaño. 

# ### Política Fiscal con Regla Procíclica
# 
# Esta política también puede formularse como una regla de aumento del gasto cuando aumenta el producto o ingreso.
# 
# Regla del Gasto Procíclico: $G=G_{0}+gY$.
# 
# En este caso, el parámetro de la regla “$g$” aparece aumentando tanto la pendiente de la función de $DA$ como la magnitud del multiplicador:
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + [(b-m)(1-t)+g]Y $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t) - g]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# La política fiscal procíclica, entonces, prolonga los auges y las depresiones.

# ### Política Fiscal con Regla de Presupuesto Equilibrado
# 
# Esta política se expresa en una regla del gasto igual al total de los impuestos; es decir: $G = T$ o $G = tY$. 
# 
# El gasto varía en la misma dirección de la variación del producto o ingreso. 
# 
# Esta regla es, por lo tanto, igualmente procíclica.
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=T$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda  | $DA=C+I+G+X-M$ |
# 
# $$ DA = (C_{0} + I_{0} + G_{0} + X_{0} - hr) + [(b-m)(1-t)+t]Y $$ 
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t) - t]} (C_{0} + I_{0} + G_{0} + X_{0} - hr) $$ 
# 
# El parámetro “$t$” aparece aumentando tanto la pendiente de la función de $DA$ como la magnitud del multiplicador.

# ## El Modelo Ingreso-Gasto: la Curva IS
# 
# En el modelo Ingreso-Gasto el equilibrio entre el Ingreso y la $DA$ representa el equilibrio en el mercado de bienes. 
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=C_{0}+bY^{d}$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# | Gasto del Gobierno:   | $G=G_{0}$ |
# | Tributación:    | $T=tY$ |
# | Exportaciones:   | $X=X_{0}$ |
# | Importaciones:  | $M=mY^{d}$ |
# | Gasto o Demanda Agregada  | $DA=C+I+G+X-M$ |
# | Condición de equilibrio:  | $Y=C+I+G+X-M$ |
# 
# Haciendo reemplazos y algunas operaciones algebraicas en esta condición de equilibro, se obtiene:
# 
# $$ Y = \dfrac{1}{[1 - (b-m)(1-t)]} (C_{0} + I_{0} + G_{0} + X_{0}) - \dfrac{h}{1-(b-m)(1-t)}r $$ 
# 
# O, lo que es lo mismo:
#         
# $$ r = \dfrac{1}{h}(C_{0} + I_{0} + G_{0} + X_{0}) - \dfrac{1-(b-m)(1-t)}{h}Y $$
# 
# Esta ecuación indica que existen pares de valores ordenados de ingreso, $Y$, y de tasa de interés, $r$ que equilibran el mercado de bienes.
# 
# Esta relación inversa se puede ilustrar gráficamente a partir del equilibro $DA=Y$ en la recta de $45^{0}$.
# 
# Dado el ingreso de equilibro “$Y_{0}$” que corresponde a una tasa de interés real igual a “$r_{0}$” (véase parte superior del gráfico), supongamos que se produce una reducción de la tasa de interés que ahora será igual a $r_{1}$; es decir: $r_{0} > r_{1}$.
# 
# Esta reducción de la tasa de interés eleva la magnitud del intercepto de la función de $DA$ desplazándola hacia arriba dando lugar a un ingreso de equilibrio de mayor magnitud e igual a $Y_{1}$. 
# 
# Las desigualdades $r_{0} > r_{1}$ y $Y_{0} < Y_{1}$ que corresponden a valores de la tasa de interés y del ingreso que equilibran el mercado de bienes, se pueden representar en el plano ($Y$, $r$) como una recta con pendiente negativa (véase parte inferior del gráfico) que pasa por los puntos $(Y_{0}, r_{0})$ y $(Y_{1}, r_{1})$. Esta recta es conocida como la Curva IS que representa el equilibrio en el mercado de bienes.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Derivación de la curva IS a partir del equilibrio $Y=DA$</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/7.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 67.</figcaption>
# </figure>
# </center>

# ### La Curva IS o el equilibrio Ahorro-Inversión
# 
# El equilibrio entre el ingreso y la demanda agregada es equivalente al equilibrio entre el ahorro total y la inversión. En efecto, si se parte de condición de equilibro $DA = Y$:
#         
# $$ Y = C + I + G + X - M $$
#         
# haciendo las algunas operaciones se puede llegar al equilibrio Ahorro-Inversión. Restamos la tributación “$T$” de ambos miembros de la igualdad, para obtener la ecuación del ingreso disponible.
# 
# $$ Y - T = C + I - T + G + X - M $$
#         
# $$ Y^{d} = C + I - T + G + X - M $$
#         
# Esta igualdad se puede reescribir como sigue:
# 
# $$ (Y^{d} - C) + (T-G) + (M-X) = I $$
# 
# El lado derecho constituye el ahorro total que es igual a la inversión. El ahorro total incluye el *Ahorro Privado*, el *Ahorro del Gobierno* y el *Ahorro Externo*.
#         
# $$ S_{p} + S_{g} + S_{e} = I_{0} - hr $$ $$ S = I_{0} - hr $$ 
# 
# Esta igualdad entre el ahorro y la inversión representa precisamente la curva IS:
#         
# $$ S(Y) = I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$ (Y^{d} - C_{0} - bY^{d}) + (T - G_{0}) + (mY^{d} - X_{0}) = I_{0} - hr $$ 
# 
# A partir de aquí, mediante algunas operaciones algebraicas, se obtiene la ecuación de la Curva IS.
#         
# $$ [1-(b-m)(1-t)]Y - (C_{0}+G_{0}+X_{0}) = I_{0}-hr  $$ 
# 
# Haciendo algunas operaciones, la Curva IS puede expresarse con una ecuación donde la tasa de interés se encuentra en función del ingreso:
#         
# $$ r = \dfrac{1}{h}(C_{0}+ I_{0} + G_{0}+X_{0}) - \dfrac{1-(b-m)(1-t)}{h}Y $$
# 
# Esta relación se puede expresar en forma breve, como:
# 
# $$ r = \dfrac{\beta_{0}}{h} - \dfrac{\beta_{1}}{h}Y $$
# 
# donde $\beta_{0} = C_{0} + I_{0} + G_{0} + X_{0} $ y $\beta_{1} = 1-(b-m)(1-t) $
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>La Curva IS de Equilibrio en el Mercado de Bienes</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/8.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 69.</figcaption>
# </figure>
# </center>
# 
# La pendiente de la Curva $IS$ es mayor que cero y puede graficarse en el plano $(Y,r)$ con una recta con pendiente negativa:
# 
# $$ \left. \dfrac{\partial r}{\partial Y} \right|_{IS} = -\dfrac{\beta_{1}}{h} < 0 $$
# 

# ### Derivación gráfica de la Curva IS a partir de las funciones de inversión y ahorro
# 
# Se sabe que las funciones de ahorro total y de inversión, son:
# 
# |  |  |
# | :------: | :-----------:|
# | Función Consumo:   | $C=-(C_{0}+G_{0}+X_{0})+[1-(b-m)(1-t)]Y$ |
# | Función Inversión: | $I=I_{0}-hr$ |
# 
# El siguiente gráfico contiene las funciones de ahorro y de inversión. 
# 
# En el eje vertical se representa la igualdad $S=I$. El equilibrio ocurre en los puntos $A$ y $B$ que corresponden a los pares ordenados $(Y_{0}, r_{0})$, y $(Y_{1}, r_{1})$. Hay, como se puede observar, una relación inversa entre la tasa de interés real y el ingreso. Tasas de interés más bajas corresponden a ingresos más altos.   
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Funciones de Ahorro e Inversión</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/9.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 70.</figcaption>
# </figure>
# </center>
# 
# Estos pares de puntos $(Y_{0}, r_{0})$, y $(Y_{1}, r_{1})$ se pueden graficar en el plano $(Y, r)$ y la unión de estos puntos representa la Curva $IS$. Vea el siguiente gráfico.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Curva IS</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/10.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 71.</figcaption>
# </figure>
# </center>

# ### Desequilibrios en el mercado de bienes
# 
# Todos los puntos de la Curva $IS$ corresponden a pares ordenados $(Y, r)$ que equilibran el mercado de bienes. En consecuencia, los puntos fuera de esta curva son de desequilibrio en el mercado. 
# 
# Es importante identificar el tipo de desequilibrio de mercado (exceso de oferta o exceso de demanda), a la derecha y a la izquierda de la curva IS.
# 
# El punto “$A$” del siguiente gráfico, es de equilibrio: el par ordenado $(Y_{A}, r_{A})$ equilibra el ahorro con la inversión $(I_{A}, S_{A})$
# 
# El punto “$B$”, que se encuentra a la derecha de la $IS$ corresponde al par ordenado $(Y_{B}, r_{A})$. Como se mantiene la tasa de interés, la inversión permanece constante, pero el ahorro es mayor que el que corresponde al punto “$A$” porque el ingreso en el punto “$B$” es más alto. 
# 
# En el punto “$B$”, entonces, $(I_{A} < S_{A})$ lo tanto, a la derecha de la $IS$ el desequilibrio del mercado de bienes es de exceso de oferta. 
# 
# El lector puede mostrar que a la izquierda de la curva $IS$ el desequilibrio de mercado es de exceso de demanda.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Equilibrio y Desequilibrio en el Mercado de Bienes</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/11.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 72.</figcaption>
# </figure>
# </center>

# ### Movimientos de la curva IS
# 
# La expresión matemática de la *Curva IS* es:
# 
# $$ r = \dfrac{\beta_{0}}{h} - \dfrac{\beta_{1}}{h}Y $$
# 
# donde $\beta_{0} = C_{0} + I_{0} + G_{0} + X_{0} $ y $\beta_{1} = 1-(b-m)(1-t) $
# 
# Supongamos, en primer lugar, que no hay cambios en los parámetros “$b$”, “$m$”, “$t$” y “$h$”. 
# 
# Los cambios en el intercepto provocados por los cambios en los componentes autónomos de la demanda agregada (consumo autónomo, la confianza de los inversionistas, el gasto del gobierno y las exportaciones autónomas), son los que pueden dar lugar a desplazamientos paralelos de la curva IS, hacia a la derecha o hacia a la izquierda.
# 
# El incremento de cualquier gasto autónomo, aumenta la demanda agregada de bienes generando así un exceso de demanda que da lugar a un desplazamiento de la curva IS hacia la derecha, donde, como se ha mostrado anteriormente, hay exceso de oferta.
# 
# Por ejemplo, si el gobierno aplica una política fiscal expansiva aumentado el gasto, $G$, la curva IS debe desplazarse hacia la derecha con una distancia equivalente a la magnitud de dicho gasto. Vea el siguiente gráfico.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos del aumento del Gasto del Gobierno</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/12.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 73.</figcaption>
# </figure>
# </center>
# 
# Finalmente, hay que mencionar que los cambios en los parámetros “$b$”, “$m$” y “$t$” modifican la pendiente de la curva IS. 
# 
# Si se adopta una política fiscal restrictiva aumentando la tasa de tributación, la pendiente de la IS aumenta, provocando un movimiento en el sentido de las agujas del reloj. Vea el siguiente gráfico. 
# 
# Recuérdese que un aumento de la tasa de tributación reduce tamaño del multiplicador y la pendiente de la demanda agregada. Se puede observar en el gráfico que, para una misma tasa de interés, el ingreso de equilibrio será menor.
#  
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos de un aumento en la Tasa de Impuestos $(\Delta t > 0)$</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/13.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 73.</figcaption>
# </figure> 
# </center>
# 
# Los cambios en la propensión marginal a importar “$m$”, tienen efectos en la pendiente de la curva IS parecidos a los provocados por los cambios en la tasa de tributación “$t$”: sus aumentos tienen efectos contractivos y sus disminuciones tienen efectos expansivos sobre el producto.
# 
# Cuando aumenta (disminuye) “$m$”, la pendiente de la curva IS se incrementa (reduce).
# 
# Por el contrario, el aumento de la propensión marginal a consumir, “b”, aumenta el tamaño del multiplicador, aumenta la pendiente de la $DA$ y reduce la pendiente de la curva $IS$.
# 
# Esta gira en sentido contrario a las agujas del reloj. Vea el siguiente gráfico.
# 
# <center>
# <figure>
#     <figcaption>
#         <center>
#             <b>Efectos de un aumento en la PMgC $(\Delta b < 0)$</b>
#         </center>
#     </figcaption>
#     <img title="a title" src="../figs/14.png">
#     <figcaption>Apuntes de Macroeconomía Intermedia (Jiménez 2020), Page 74.</figcaption>
# </figure>
# </center>    

# In[1]:


import ipywidgets as widgets


# In[2]:


import panel as pn
import numpy as np
import holoviews as hv

pn.extension(sizing_mode = 'stretch_width')


# In[3]:


import numpy as np               # Package for scientific computing with Python
import matplotlib.pyplot as plt  # Matplotlib is a 2D plotting library


# In[4]:


"2|DEFINE PARAMETERS AND ARRAYS"
# Parameters
Y_size = 100
a = 20                 # Autonomous consumption
b = 0.2                # Marginal propensity to consume
alpha = 5              # Autonomous imports
beta  = 0.2            # Marginal propensity to import
T = 1                  # Taxes
I_bar = 10             # Investment intercept (when i = 0)
G_bar = 8              # Government spending
X_bar = 2              # Exports (given)
d = 5                  # Investment slope wrt to i
# Arrays
Y = np.arange(Y_size)  # Array values of Y

"3|DEFINE AND POPULATE THE IS-SCHEDULE"
def i_IS(a, alpha, b, beta, T, I_bar, G_bar, X_bar, d, Y):
    i_IS = ((a-alpha)-(b-beta)*T + I_bar + G_bar + X_bar - (1-b+beta)*Y)/d
    return i_IS

def Y_IS(a, alpha, b, beta, T, I_bar, G_bar, X_bar, d, i):
    Y_IS = ((a-alpha)-(b-beta)*T + I_bar + G_bar + X_bar - d*i)/(1-b+beta)
    return Y_IS

i = i_IS(a, alpha, b, beta, T, I_bar, G_bar, X_bar, d, Y)

"4|PLOT THE IS-SCHEDULE"
y_max = np.max(i)
x_max = Y_IS(a, alpha, b, beta, T, I_bar, G_bar, X_bar, d, 0)

v = [0, x_max, 0, y_max]                        # Set the axes range
fig, ax = plt.subplots(figsize=(10, 8))
ax.set(title="IS SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax.plot(Y, i, "k-")
ax.yaxis.set_major_locator(plt.NullLocator())   # Hide ticks
ax.xaxis.set_major_locator(plt.NullLocator())   # Hide ticks
plt.axis(v)                                     # Use 'v' as the axes range
plt.show()


# In[5]:


bootstrap = pn.template.BootstrapTemplate(title='Bootstrap Template')

xs = np.linspace(0, np.pi)
xs = xs[1:10]
freq = pn.widgets.FloatSlider(name="Frequency", start=0, end=10, value=2)
phase = pn.widgets.FloatSlider(name="Phase", start=0, end=np.pi)

@pn.depends(freq=freq, phase=phase)
def sine(freq, phase):
    return hv.Curve((xs, xs*freq+phase)).opts(
        responsive=True, min_height=400)

@pn.depends(freq=freq, phase=phase)
def cosine(freq, phase):
    return hv.Curve((xs, -xs*freq+phase)).opts(
        responsive=True, min_height=400)

bootstrap.sidebar.append(freq)
# bootstrap.sidebar.append(phase)

bootstrap.main.append(
    pn.Row(
        pn.Card(hv.DynamicMap(sine), title='IS'),
        pn.Card(hv.DynamicMap(cosine), title='LM')
    )
)
bootstrap.servable()


# In[6]:


bootstrap.show();


# In[ ]:





# In[ ]:




