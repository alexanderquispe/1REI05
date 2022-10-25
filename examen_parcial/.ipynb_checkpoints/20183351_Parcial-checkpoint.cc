{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c2cf5691",
   "metadata": {},
   "source": [
    "# $$EXAMEN - PARCIAL$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a3568eb",
   "metadata": {},
   "source": [
    "## $Claudia - Cabrel$\n",
    "\n",
    "## $20183351$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf8cc3d7",
   "metadata": {},
   "source": [
    "## Teoría (11 puntos)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5449d33d",
   "metadata": {},
   "source": [
    "### 1. (2 puntos) Defina el Multiplicador Keynesiano y dé un ejemplo de cómo influye este en el modelo Ingreso-Gasto Keynesiano."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71992407",
   "metadata": {},
   "source": [
    "    La definición de el Multiplicador Keynesiano se refiere al cambio de los componentes de la demanda agregada (DA) que conforma el intercepto; a su vez, el aumento multiplicado del ingreso es un resultado de los efectos directos e indirectos que se dan por el aumento de los componentes propios de la demanda agregada; asimismo, este es un. modelo que influey en el modelo Ingreso-Gasto Keynesiano."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "002ac844",
   "metadata": {},
   "source": [
    "### 2. (2 puntos) Grafique y explique cúando sucede un exceso y déficit de demanda en el modelo Ingreso-Gasto. Señale las áreas donde ocurre cada caso. Explique cómo se converge al equilibrio a partir de estos dos escenarios."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "d4085188",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import sympy as sy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "1e9ed9b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y_size = 100\n",
    "\n",
    "C0 = 20   \n",
    "I0 = 10  \n",
    "G0 = 40   \n",
    "X0 = 2   \n",
    "b = 0.4 \n",
    "m = 0.2   \n",
    "t = 0.4   \n",
    "h = 1     \n",
    "r = 0.6  \n",
    "\n",
    "\n",
    "α_0 = C0 + I0 + G0 + X0 - (h*r)\n",
    "α_1 = (b-m)*(1-t)\n",
    "\n",
    "Y = np.arange(Y_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "c6f8dad8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def DA(α_0, α_1, Y):\n",
    "    DA = α_0 + (α_1*Y)\n",
    "    return DA\n",
    "\n",
    "DA = DA(α_0, α_1, Y)\n",
    "\n",
    "Y_size = 100\n",
    "\n",
    "a = 1    \n",
    "\n",
    "Y = np.arange(Y_size)\n",
    "\n",
    "def Y_45(a, Y):\n",
    "    Y_45 = a*Y\n",
    "    return Y_45\n",
    "\n",
    "Y_45 = Y_45(a, Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "98d4dc91",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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mEB6enq348888wxXXHHFoF5bRETkjJyDY7WhlmC9zSkKg5k9iF+ckAEcAr4C/AbYAOQAe4FrnXNHI4+/E/gM0A7c6pz73ZneY/ny5e6VV17pdqykpISCgoKofY541dDQwC233EJKSgorVqw45bJpf73++uvcc889PPDAA30+prq6mjvvvJOnnnqKv/iLv+COO+4427JPuvrqq/na177G/PnzT7lP51hERKKitQ1KSuFYHZy/CNJSh/TtzOxV59zynsdj5rKpc+6GPu5a2dtB59x6YP3QVZRYxowZw09/+tNBv87SpUu59NJL6ejo6PPy6+TJk/nhD3846Pc6obW1lY997GO9BjcREZGoqK2H7buhvR3m5w55cDudmAlvMnx85jOfCfT9Ro0axU033RToe4qISIJwDg4cgd37YHQyLC2A9LRQS1J4ExEREelNRwe8sxcOVcOk8bAgD5LDj07hVyAiIiISa5qa/WXShibInQE5mdBL+6owKLyJiIiIdFV1DN7a47vKnjPXj7rFEIU3EREREfDz28oOwN6Dfl7botmQMjrsqk6h8CYiIiLS1gYle6CmFqZnwNwcGBGb7XAV3kRERCSx1TbAjt2+j9u8WZA5JeyKTkvhTURERBKTc3CwCnbthVHJsHQBjB0TdlVnFJvjgQnEOceKFSv43e/e3SBiw4YNXH755X0+p6Ojg6VLl3LllVeePHbXXXeRlZVFYWEhhYWFPPHEEyfv+/u//3uWL1/Os88+OzQfQkREJN50dMLOMninHCaMhWUL4yK4gUbeQmdm/PCHP+Taa689uTPBnXfeyaZNm/p8zre//W0KCgqore2+t9oXv/hF/u7v/q7bsbfeeguA5557jrVr13LxxRdH/0OIiIjEk6YW2LEL6ptgVibMmhEzbUD6Q+Gtq117ob4xuq+ZngZzck77kMWLF3PVVVfxjW98g4aGBm666SZmz57d62MrKip4/PHHufPOO7nnnnvO+PYdHR2MGDECMyNW9rEVEREJTfUx3wYEYPEcmDwhzGrOisJbjPjKV77Ceeedx6hRo3jllVf6fNytt97K3XffTV1d3Sn3fe973+P+++9n+fLl/Pu//zsTJ05k0aJFNDY2smLFCv7t3/5tKD+CiIhI7HIOyg9A+UFIT4WFcyA19tqA9IfCW1dnGCEbSmPGjOETn/gE6enpjB7d+x+m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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_max = np.max(DA)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 8))\n",
    "ax.set(title=\"MODELO INGRESO-GASTO\", xlabel= 'Y', ylabel= 'DA')\n",
    "ax.plot(Y, DA, label = \"$DA = α_0 + (α_1*Y)$\", color = \"ORANGE\", lw = 2) \n",
    "ax.plot(Y, Y_45, label = \"Y_45°\", color = \"PINK\") \n",
    "\n",
    "plt.axvline(x=81.2,  ymin= 0, ymax= 0.80, linestyle = \":\", color = \"gray\")\n",
    "plt.axhline(y=81.2, xmin= 0, xmax= 0.80, linestyle = \":\", color = \"gray\")\n",
    "plt.plot(81.2, 81.2, marker=\"o\", color=\"black\", label = \"Eo\")\n",
    "plt.plot(70.2, 80.2, marker=\"o\", color=\"black\", label = \"L\")\n",
    "plt.plot(94, 81.8, marker=\"o\", color=\"black\", label = \"H\")\n",
    "\n",
    "plt.text(80, -15, '$Y^e$', fontsize = 16, color = 'black')\n",
    "plt.text(-15, 80, '$DA^e$', fontsize = 16, color = 'black')\n",
    "plt.text(79, 83, 'Eo', fontsize = 14, color = 'black')\n",
    "plt.text(70, 85, 'L', fontsize = 14, color = 'black')\n",
    "plt.text(94, 84, 'H', fontsize = 14, color = 'black')\n",
    "\n",
    "ax.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86fc7af7",
   "metadata": {},
   "source": [
    "    El punto E0 representa el equilibrio del modelo de Ingreso-Gasto. Es en este punto donde todo lo demandado en la economía es igual a todo lo producido. Es decir, se cumple la condición de equilibrio Y = DA. Sin embargo, pueden ocurrir situaciones donde la demanda puede ser mayor o menor que el producto (situaciones de exceso o déficit). Igualmente, el modelo es estable porque hay convergencia hacia el punto E0 (el equilibrio). \n",
    "\n",
    "    En el caso de un exceso de demanda, lo que sucede es que los inventarios de las empresas son menores a la cantidad demanda por el mercado y, por tanto, estas deciden incrementar la producción en un monto similar al que aumentó la demanda. \n",
    "    - ↑DA --> Y < DA -->↑Y\n",
    "    - En el gráfico, el punto L representa cuando la demanda agregada es mayor que el nivel de producción. Tras el déficit de producción, las empresas hacen aumentos de su producción hasta llegar al equilibrio E0.\n",
    "\n",
    "    En el caso de un déficit de demanda, esto se debe a que los inventarios de las empresas con mayores que las cantidades demandas por el mercado, es decir, hay un déficit de demanda. Las empresas deciden reducir su producción hasta que se iguale a la cantidad demanda en la economía.\n",
    "    - ↓DA --> Y>DA --> ↓Y\n",
    "    - En el gráfico, el punto H representa cuando la demanda agregada es menor que el nivel de producción. Tras el exceso de producción, las empresas hacen reducciones de su producción hasta llegar al equilibrio E0."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b55930c7",
   "metadata": {},
   "source": [
    "### 3. (2 puntos) ¿Cuáles son las herramientas de política fiscal y política monetaria? Dea un ejemplo para cada una de ellas dentro del contexto peruano. Coloque su fuente en caso sea necesario. (Solo necesita 1 para pol. Fiscal y 1 para Monetaria)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f22b65dd",
   "metadata": {},
   "source": [
    "    Las herramientas de la política fiscal son un conjunto de acciones gubernamentales en donde se utilizan instrumentos discrecionales para modificar ingresos y gastos. Por otro lado, una política fiscal se refuere a una herramnienta para el gasto público."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98152257",
   "metadata": {},
   "source": [
    "### 4. (2) Explique qué funciones del dinero respaldan el éxito de de Yape y Plin(Aplicativos moviles para intercambio de dinero)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49bae060",
   "metadata": {},
   "source": [
    "    En el texto, el autor Jiménez, menciona que la economía tiene 4 funciones. La primera sostiene que es la unidad de cuenta y patrón de precios, en donde las aplicaciones Yape y Plin resaltan su función como aplicaciones capaces de generar intercambios de dinero por medio de sus transacciones, así como el autor menciona que el dinero puede disminuir los costos de transacción, debido a que reduce el número de precios existentes en la economía, facilitando las transacciones. Asimismo, Yape y Plin cumplen con la segunda función del dinero, que seria basicamente la concepción del dinero como medio de intercambio, lo cual propone una eficiencia económica, ya que gracias a estas aplicaciones que no tienen un cobro extra por hacer transacciones de dinero, eliminan propiamente estos costos de los intercambios de bienes y servicios. En cuanto a la tercera función que el autor menciona que sería que el dinero sirve para cancelar deudas, estos aplicativos sirven para poder utilizarse como medios de pagos de varios locales o establecimientos. Finalmente, la cuarta funcón sería la del depósito y la conservación del dinero, ya que Yape y Plin funcionan como aplicaciones que derivan de cuentas bancarias, donde, tal cual como dice el autor, el poder adquisitivo es decir el dinero, sirve para que pueda guardarse. En conclusión, estas aplicaciones han facilitado en gran medida las funciones principales que el dinero tiene y hasta el día de hoy han logrado en buena medida que más personas puedan utilizarlos gracias a su funcionalidad."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37f569e1",
   "metadata": {},
   "source": [
    "### 5.(3 puntos) Cuáles son las conclusiones principales del paper de Dancurt \"Inflation Targeting in Peru: The Reasons for the Success\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "270cad0e",
   "metadata": {},
   "source": [
    "    En el artículo en de Oscar Dancourt, el autor concluye que la política monetaria por parte del Banco Central de Reserva del Perú generó un buen desempeño macroeconómico en nuestro país durante la década de 2002-2013. Es decir, se mantuvo una baja inflación, alto crecimiento del PBI y se impidieron recesiones severas. Esto proviene del análisis de los, según el autor,  dos canales más importantes e influyentes en la actividad económica y el nivel de precios: el canal de crédito y el canal de tipo de cambio. Por otro lado, tenemos el canal de crédito en donde diferentes medidas que impactan en la oferta y demanda del credito son el coeficiente de encaje, el ajuste de la tasa de interés y el desarrollo del mercado local de bonos públicos. FInalmente, el autor sostiene que para poder mantener cierta estabilidad en el tipo de cambio, fue necesario acumular reservas de divisas. Sin embargo, las políticas monetarias del BCR pudieron tener un mejor manejo. Y finalmente, el \"desdolarizar\" el crédito es una medida viable para el control de la caída del dolar."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2b07c7d",
   "metadata": {},
   "source": [
    "## Ejercicio 2 (5 puntos). Modelo de OA-DA y el nuevo esquema institucional de Política Monetaria, donde la tasa de interés de referencia $ r_0 $ es usada como instrumento."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edf41b44",
   "metadata": {},
   "source": [
    "Utilice las siguientes ecuaciones\n",
    "\n",
    "    Demnanda Agregada\n",
    "    \n",
    "$$ IS:[1-b(1-t)]Y=C_0+I_0+G_0-hr_0$$\n",
    "\n",
    "$$LM: {M_0^s}-{P}=kY-jr_o $$\n",
    "    \n",
    "    Oferta Agregadda\n",
    "    \n",
    "$$ P=P^e+θ(Y-\\bar{Y})$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b252a089",
   "metadata": {},
   "source": [
    "### (2 puntos) Encuentre las ecuaciones de equilibrio de $ Y^{eq\\_da\\_oa}$, $P^{eq}$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "113ae2d4",
   "metadata": {},
   "source": [
    "$ IS: [1-b(1-t)]Y= C_o + I_o + G_o -hr_0$\n",
    "\n",
    "$ IS: \\beta_1Y=\\beta_0-hr_0$\n",
    "\n",
    "$ IS: r_0= \\frac{\\beta_0}{h}-\\frac{\\beta_1}{h}Y  $\n",
    "\n",
    "En ese sentido, tenemos como consiguiente:\n",
    "\n",
    "$ \\beta_0 = C_o + I_o + G_o $ y $ \\beta_1 = 1 - (b)(1 - t) $\n",
    "\n",
    "$ LM: M_0^s-P=kY-jr_0$\n",
    "\n",
    "\n",
    "Y siguiendo esa linea, reemplazamos($r$)\n",
    "Lo que nos da la ecuacón ($DA$) \n",
    "\n",
    "$ M_0^s-P=kY-j\\frac{\\beta_0}{h}+j\\frac{\\beta_1}{h}Y $\n",
    "\n",
    "$ P=M_0^s+j\\frac{\\beta_0}{h}-j\\frac{\\beta_1}{h}Y -kY$\n",
    "\n",
    "$P=\\frac{hM_0^s+j\\beta_0}{h}-\\frac{j\\beta_1+hk}{h}Y$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc0d3a73",
   "metadata": {},
   "source": [
    "En cuanto a la Oferta Agregada\n",
    "\n",
    "$ P=P^e+θ(Y-\\bar{Y})$\n",
    "\n",
    "\n",
    "Finalmente para hallar $ Y^{eq\\_da\\_oa}$ tenemos:\n",
    "\n",
    "$\\frac{h Mo^s + jB_o}{h} - \\frac{jB_1 + hk}{h}Y = P^e + θ(Y - \\bar{Y})$\n",
    "\n",
    "$Y^{eq\\_da\\_oa} = [ \\frac{1}{(θ + \\frac{jB_1 + hk}{h})} ]*[(\\frac{h Mo^s + jB_o}{h} - P^e + θ\\bar{Y})]$\n",
    "\n",
    "\n",
    " Y para $P^{eq}$ reemplazamos $ Y^{eq}$ de la oferta agregada\n",
    "\n",
    "$P^{eq\\_da\\_oa} = P^e + θ( Y^{eq\\_da\\_oa} - \\bar{Y} )$\n",
    "\n",
    "$P^{eq\\_da\\_oa} = P^e + θ( [ \\frac{1}{(θ + \\frac{jB_1 + hk}{h})} ]*[(\\frac{h Mo^s + jB_o}{h} - P^e + θ\\bar{Y})] - \\bar{Y} )$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d187f00",
   "metadata": {},
   "source": [
    "### (3 puntos) Analice los efectos sobre las variables endógenas Y, P ante una subida de la tasa de interés de referencia $ r_0 $. $(Δr_o > 0)$. El análisis debe ser intuitivo, matemático y gráfico."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf823840",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
