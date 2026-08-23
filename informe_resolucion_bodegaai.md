# Análisis de series temporales del tráfico productivo de BodegaAI

**Trabajo Práctico N.º 1 — Análisis de Series Temporales**
  
**Autores:** Juan Martin Pedrozo, Carlos Aular  

**Institución:** Universidad Austral

**Fecha:** 23/08/2026

## Resumen

Este informe presenta una resolución técnica del análisis de series temporales aplicado al tráfico productivo de BodegaAI. Se estudian dos series horarias: requests totales del Application Load Balancer (ALB) durante julio de 2026 y requests por target del POS API durante el período disponible entre julio y agosto de 2026. El objetivo es caracterizar su dinámica temporal, evaluar estacionariedad, identificar dependencia y estacionalidad, ajustar modelos SARIMA para pronóstico univariado y complementar el análisis con un modelo VAR sobre el período común de observación.

Los resultados muestran una estructura horaria compatible con estacionalidad diaria en ambas series. Para ALB, el modelo seleccionado fue `SARIMA(1, 1, 1)x(1, 1, 1, 24)`, con MAPE de test cercano a `0,79%`. Para POS API, el modelo seleccionado fue `SARIMA(1, 1, 1)x(1, 0, 1, 24)`, con MAPE de test cercano a `3,90%`. La evaluación de residuos indica que los modelos capturan una parte relevante de la dinámica, aunque todavía queda autocorrelación remanente, especialmente en rezagos diarios. El análisis VAR, aplicado sobre `log1p + diff(24)`, aporta una lectura exploratoria de interacción predictiva, pero no permite afirmar causalidad operacional.

## 1. Introducción y problema analítico

El tráfico de una aplicación productiva suele presentar patrones recurrentes asociados a horarios de uso, ciclos diarios, cambios de carga y eventos puntuales. En este contexto, el análisis de series temporales permite describir la evolución del volumen de requests, detectar estructura temporal y construir pronósticos de corto plazo que apoyen decisiones operativas.

El caso de estudio se concentra en BodegaAI, tomando como unidad de análisis registros horarios de requests. La problemática abordada consiste en determinar si las series presentan comportamiento estacionario, si existe una estacionalidad relevante, qué modelos permiten representar mejor la dinámica observada y qué grado de precisión puede obtenerse al pronosticar una ventana corta fuera de muestra.

El trabajo se organiza en bloques: descripción de datos, análisis de estacionariedad, funciones de autocovarianza/autocorrelación/autocorrelación parcial, selección y evaluación SARIMA, diagnóstico de residuos, pronóstico y análisis multivariado mediante VAR, impulso-respuesta y pruebas de causalidad de Granger.

## 2. Descripción de los datos

Se trabajó con dos fuentes de tráfico productivo a frecuencia horaria:

| Serie | Archivo | Variable usada | Rango temporal | Observaciones | Faltantes |
|---|---|---:|---|---:|---:|
| ALB total | `data/alb-prod-requests-2026-07.csv` | `requests` | 2026-07-01 00:00 UTC a 2026-07-31 23:00 UTC | 744 | 0 |
| POS API por target | `data/bodegaai-production-pos-api-request-count-per-target-last-30-days-hourly.csv` | `value` | 2026-07-18 23:00 UTC a 2026-08-17 22:00 UTC | 720 | 0 |

La preparación consistió en ordenar los registros por `period_start_utc`, construir un índice temporal horario y reindexar cada serie a una grilla completa de una hora. Esta decisión evita que los modelos interpreten un calendario implícito incorrecto y permite verificar continuidad temporal.

En términos descriptivos, ALB presenta una media aproximada de `2.114.963` requests por hora, con mínimo de `883.045` y máximo de `2.536.631`. POS API presenta una media aproximada de `829.950` requests por target por hora, con mínimo de `318.307,50` y máximo de `924.146,50`. En ambas series se observan patrones horarios compatibles con ciclos diarios, además de caídas puntuales que deben considerarse al interpretar el ajuste.

**[Insertar Figura 1 aquí]**  
**Figura 1**  
*Series horarias originales de requests para ALB y POS API.*  
Nota. Usar el gráfico de niveles del notebook para mostrar escala, ciclo diario y caídas puntuales.


## 3. Metodología

### 3.1 Preparación y exploración inicial

Las series se analizaron en niveles y mediante transformaciones por diferencia regular `diff(1)`, diferencia estacional diaria `diff(24)` y doble diferencia `diff(24).diff(1)`. Dado que los datos son horarios, el período estacional natural considerado fue `s = 24`.

La inspección gráfica se usó como primer criterio para observar cambios de nivel, ciclos recurrentes y valores extremos. Luego, la evidencia visual se contrastó con pruebas formales y con el diagnóstico de los modelos ajustados.

### 3.2 Estacionariedad y raíces unitarias

La estacionariedad se evaluó con dos pruebas complementarias:

- **ADF:** su hipótesis nula sostiene que la serie presenta raíz unitaria.
- **KPSS:** su hipótesis nula sostiene que la serie es estacionaria.

La lectura conjunta es importante porque ambas pruebas responden preguntas inversas. En ALB, la serie en niveles ya presenta evidencia compatible con estacionariedad al 5% según ADF y KPSS, aunque mantiene una estructura diaria clara. En POS API, la evidencia en niveles es mixta: ADF rechaza raíz unitaria, pero KPSS rechaza estacionariedad. Luego de transformar la serie, especialmente con diferencias, la evidencia mejora.

Por este motivo, la decisión metodológica no se apoyó exclusivamente en un test. Se combinaron pruebas, gráficos, estructura del negocio y diagnóstico posterior.

### 3.3 FAC, FACP y estructura temporal

Se analizaron FAS, FAC y FACP hasta rezagos de 72 horas, marcando especialmente los rezagos 24, 48 y 72. La presencia de dependencia persistente y señales en rezagos diarios reforzó la elección de modelos con componente estacional.

La FAC mostró que las observaciones no se comportan como ruido independiente. La FACP ayudó a acotar órdenes autorregresivos simples, pero la señal dominante fue la periodicidad diaria. Por eso, los modelos candidatos incorporaron estacionalidad con `s = 24`.

### 3.4 Modelado SARIMA

Se comparó una grilla acotada de modelos SARIMA para cada serie. La muestra tiene aproximadamente un mes de observaciones, por lo que una búsqueda excesivamente amplia aumentaría el riesgo de sobreajuste y el costo computacional sin garantizar mejor generalización.

La evaluación combinó criterios de información en entrenamiento y desempeño fuera de muestra. Se reservaron las últimas 48 horas como conjunto de test, coherente con la frecuencia horaria y con un horizonte operativo de corto plazo. Las métricas empleadas fueron MAE, RMSE y MAPE.

Cuando los criterios de información y el desempeño en test no coincidieron, se priorizó el MAPE fuera de muestra, porque el objetivo del bloque era validar capacidad predictiva y no solo ajuste interno.

### 3.5 VAR, impulso-respuesta y Granger

El análisis multivariado se realizó con VAR sobre el tramo temporal común entre ambas series: desde 2026-07-18 23:00 UTC hasta 2026-07-31 23:00 UTC. Antes de ajustar el VAR, se aplicó `log1p` y diferencia estacional de 24 horas. Luego de esta transformación quedaron 289 observaciones.

Esta transformación estabiliza la escala y centra el análisis en variaciones estacionales, no en niveles absolutos. Por lo tanto, las métricas del VAR se reportan en escala transformada y no se comparan directamente con las métricas SARIMA en requests. Tampoco se reporta MAPE para VAR, porque las diferencias logarítmicas pueden acercarse a cero o cambiar de signo, lo que vuelve poco interpretable el error porcentual.

## 4. Resultados

### 4.1 Series originales y diferenciación

La inspección de las series originales muestra que ambas contienen ciclos horarios compatibles con estacionalidad diaria. ALB presenta un cambio de nivel al inicio de julio y luego ciclos regulares. POS API conserva un nivel más estable, aunque registra caídas puntuales fuertes.

La diferencia estacional `diff(24)` reduce buena parte del ciclo diario. En ALB todavía se observan efectos asociados al cambio inicial de régimen; en POS API permanecen shocks aislados vinculados a caídas abruptas. La doble diferencia estabiliza más la media, pero también amplifica movimientos locales, por lo que no se la adopta de manera automática sin contrastarla con el desempeño del modelo.

**[Insertar Figura 2 aquí]**  
**Figura 2**  
*Transformaciones por diferencia estacional y doble diferencia para ALB y POS API.*  
Nota. Esta figura debe respaldar la decisión de evaluar modelos con componente estacional diario.


### 4.2 Pruebas de estacionariedad

Los resultados de ADF y KPSS confirman que la lectura debe ser matizada. ALB en niveles presenta ADF estadístico de aproximadamente `-4,53` con p-valor `0,00`, y KPSS estadístico de `0,41` con p-valor `0,07`. Esta combinación es compatible con estacionariedad al 5%, aunque la estacionalidad diaria sigue siendo visible y relevante.

POS API en niveles presenta ADF estadístico de aproximadamente `-9,55` con p-valor `0,00`, pero KPSS estadístico de `2,33` con p-valor `0,01`. Por lo tanto, la evidencia formal es mixta. Las transformaciones por diferencia regular y estacional muestran resultados más consistentes para trabajar con modelos que requieren mayor estabilidad.

**[Insertar Tabla 1 aquí]**  
**Tabla 1**  
*Pruebas ADF y KPSS para series originales y transformadas.*  
Nota. Usar la tabla ejecutada del notebook para documentar estadísticos y p-valores.


### 4.3 Selección de modelos SARIMA

Los modelos seleccionados fueron:

| Serie | Modelo seleccionado | AIC | BIC | MAE test | RMSE test | MAPE test |
|---|---|---:|---:|---:|---:|---:|
| ALB | `SARIMA(1, 1, 1)x(1, 1, 1, 24)` | `15.491,11` | `15.513,46` | `16.662,34` | `20.309,65` | `0,79%` |
| POS API | `SARIMA(1, 1, 1)x(1, 0, 1, 24)` | `15.737,10` | `15.759,45` | `29.640,50` | `58.453,68` | `3,90%` |

En ALB, el modelo seleccionado también fue el de mejor desempeño predictivo dentro de la grilla. En POS API, existe otro candidato con menor AIC/BIC, pero con peor MAPE de test. Por esa razón se eligió `SARIMA(1, 1, 1)x(1, 0, 1, 24)`, priorizando validación fuera de muestra.

La interpretación de parámetros se realizó con cautela. No todos los coeficientes individuales son relevantes al 5%, y por eso la selección no depende de un único parámetro, sino del balance entre desempeño predictivo, parsimonia y diagnóstico.

**[Insertar Figura 3 aquí]**  
**Figura 3**  
*FAS, FAC y FACP de las series originales.*  
Nota. Incluir los paneles de ALB y POS API para evidenciar dependencia temporal y rezagos diarios.

**[Insertar Tabla 2 aquí]**  
**Tabla 2**  
*Comparación de modelos SARIMA candidatos por AIC, BIC, MAE, RMSE y MAPE.*  
Nota. Esta tabla debe mostrar por qué la selección final prioriza validación fuera de muestra cuando corresponde.


**[Insertar Figura 4 aquí]**  
**Figura 4**  
*Comparación visual de MAPE en test para los modelos SARIMA candidatos.*  
Nota. Usar el gráfico de barras del notebook para reforzar la selección predictiva de los modelos.


### 4.4 Evaluación train/test y pronóstico

El conjunto de prueba correspondió a las últimas 48 horas de cada serie. Este horizonte cubre dos ciclos diarios completos y evita una extrapolación excesiva con una muestra histórica corta.

Los pronósticos resultaron útiles para ambas series, con mejor precisión relativa en ALB. El MAPE de `0,79%` sugiere que el modelo captura de forma adecuada la escala y el patrón horario de esa serie. En POS API, el MAPE de `3,90%` es mayor, pero razonable dada la presencia de caídas abruptas recientes y una dinámica más irregular.

El pronóstico a 48 horas mantiene la escala esperada y reproduce el ciclo diario, aunque la incertidumbre crece al avanzar el horizonte. Por lo tanto, estos modelos son más adecuados para planificación operativa de corto plazo que para inferencias extensas.

**[Insertar Figura 5 aquí]**  
**Figura 5**  
*Evaluación train/test de los modelos SARIMA seleccionados.*  
Nota. Mostrar serie reciente, valores reales de test y pronóstico para ALB y POS API.

**[Insertar Figura 6 aquí]**  
**Figura 6**  
*Pronóstico SARIMA a 48 horas con intervalos de confianza.*  
Nota. Usar los gráficos finales de pronóstico del notebook.


### 4.5 Diagnóstico de residuos

El diagnóstico residual incluyó serie temporal de residuos, histograma, FAC de residuos y prueba de Ljung-Box en rezagos 12, 24 y 48.

Los resultados indican autocorrelación remanente. En ALB, Ljung-Box no rechaza la hipótesis nula de ausencia de autocorrelación en el rezago 12, pero sí rechaza en 24 y 48. En POS API, Ljung-Box rechaza en los tres rezagos evaluados. Esto significa que los SARIMA seleccionados constituyen una base predictiva razonable, pero no eliminan toda la estructura temporal de los residuos.

Esta evidencia es relevante porque evita una lectura triunfalista del MAPE. Un error porcentual bajo no implica que el modelo sea estadísticamente perfecto; puede pronosticar bien en la ventana de test y, al mismo tiempo, dejar dependencia residual que convendría revisar con más historia o con especificaciones alternativas.

**[Insertar Figura 7 aquí]**  
**Figura 7**  
*Diagnóstico de residuos de los modelos SARIMA seleccionados.*  
Nota. Incluir serie de residuos, distribución y FAC de residuos para cada serie.

**[Insertar Tabla 3 aquí]**  
**Tabla 3**  
*Prueba de Ljung-Box en rezagos 12, 24 y 48.*  
Nota. La tabla debe sostener la conclusión de autocorrelación remanente.


### 4.6 VAR, impulso-respuesta y causalidad de Granger

El VAR se ajustó sobre las dos series alineadas en el período común y transformadas con `log1p + diff(24)`. El criterio AIC seleccionó 24 rezagos, capturando un ciclo diario completo. Sin embargo, esta selección debe interpretarse con cautela porque el tramo transformado contiene 289 observaciones, y un VAR con 24 rezagos puede ser sensible a la especificación y a la cantidad de parámetros.

**[Insertar Figura 8 aquí]**  
**Figura 8**  
*Series transformadas usadas como entrada del VAR.*  
Nota. Mostrar `log1p + diff(24)` sobre el período temporal común.

Las métricas del VAR fueron:

| Serie transformada | MAE | RMSE | Escala |
|---|---:|---:|---|
| `requests_alb` | `0,01` | `0,01` | `log1p + diff(24)` |
| `requests_pos_api` | `0,04` | `0,10` | `log1p + diff(24)` |

**[Insertar Figura 9 aquí]**  
**Figura 9**  
*Forecast VAR sobre series transformadas.*  
Nota. Usar esta figura solo como evidencia exploratoria; no compararla directamente con los pronósticos SARIMA en niveles.

La función impulso-respuesta se interpretó como herramienta exploratoria para observar co-movimientos ante shocks dentro del sistema VAR. No se la considera evidencia causal por sí misma.

**[Insertar Figura 10 aquí]**  
**Figura 10**  
*Funciones impulso-respuesta del VAR.*  
Nota. Presentar la respuesta dinámica ante shocks dentro del sistema ajustado.


Las pruebas de causalidad de Granger mostraron una lectura dependiente de la especificación. En el VAR ajustado, los rezagos de POS API ayudan a predecir ALB, con rechazo de la hipótesis nula al 5%. En sentido inverso no se rechaza la hipótesis nula. Sin embargo, las pruebas bivariadas en rezagos 6, 12 y 24 no confirman una relación estable en ninguna dirección. Por lo tanto, la conclusión correcta es prudente: existe evidencia exploratoria de capacidad predictiva condicionada al modelo, no una afirmación de causalidad operacional.

**[Insertar Tabla 4 aquí]**  
**Tabla 4**  
*Resultados de causalidad de Granger para VAR ajustado y pruebas bivariadas.*  
Nota. Esta tabla debe diferenciar la señal del VAR ajustado de las pruebas bivariadas por rezago.


## 5. Discusión

Los resultados sostienen la conveniencia de usar modelos SARIMA estacionales para series horarias de tráfico productivo. La estructura diaria aparece en gráficos, funciones de autocorrelación y desempeño comparativo de modelos. En ese sentido, tratar las series como procesos no estacionales simples perdería información importante.

El caso ALB muestra un ajuste predictivo fuerte en la ventana evaluada. Aun así, la autocorrelación residual en rezagos 24 y 48 indica que el modelo no agota la dinámica diaria. El caso POS API es más exigente: aunque el modelo elegido minimiza MAPE fuera de muestra, la serie presenta shocks más marcados y residuos con autocorrelación significativa en todos los rezagos evaluados.

La diferencia entre AIC/BIC y MAPE en POS API también es metodológicamente importante. Los criterios de información evalúan ajuste penalizado dentro de muestra, mientras que MAPE mide desempeño predictivo en una ventana reservada. Para una consigna centrada en pronóstico, priorizar desempeño fuera de muestra es una decisión defendible, siempre que se documente la tensión entre métricas.

El análisis VAR complementa, pero no reemplaza, los modelos univariados. Al trabajar sobre series transformadas y sobre un período común más corto, sus resultados deben leerse como evidencia exploratoria de interacción predictiva. Esta distinción es clave: Granger no prueba causalidad de negocio, sino utilidad predictiva de rezagos bajo una especificación determinada.

## 6. Limitaciones

La principal limitación es la longitud de la muestra: aproximadamente un mes de observaciones por serie. Esto alcanza para detectar patrones diarios y construir un primer pronóstico de corto plazo, pero limita la evaluación de ciclos más largos, cambios de régimen y validaciones temporales más robustas.

También deben considerarse los eventos puntuales observados en las series. Las caídas marcadas pueden responder a fenómenos operativos externos que no están incluidos como variables explicativas. SARIMA y VAR modelan la estructura temporal disponible, pero no incorporan información de despliegues, incidentes, campañas, calendario comercial u otros factores potencialmente relevantes.

Finalmente, los residuos con autocorrelación remanente indican que los modelos todavía dejan estructura sin explicar. Para una aplicación productiva, convendría evaluar validación rolling, ventanas históricas más largas, variables exógenas y especificaciones alternativas antes de usar estos resultados como soporte exclusivo para decisiones operativas.

## 7. Conclusiones

El análisis muestra que las series horarias de tráfico productivo de BodegaAI presentan estructura temporal clara, con evidencia compatible con estacionalidad diaria. La aproximación SARIMA con período `s = 24` resulta metodológicamente adecuada para representar esta dinámica y construir pronósticos de corto plazo.

Para ALB, el modelo `SARIMA(1, 1, 1)x(1, 1, 1, 24)` obtuvo el mejor desempeño de test, con MAPE cercano a `0,79%`. Para POS API, el modelo `SARIMA(1, 1, 1)x(1, 0, 1, 24)` fue seleccionado por MAPE fuera de muestra, cercano a `3,90%`, aunque no fuera el candidato con menor AIC/BIC.

El diagnóstico de residuos obliga a mantener una interpretación cuidadosa: los modelos pronostican razonablemente bien en la ventana evaluada, pero no eliminan toda la autocorrelación. En consecuencia, deben entenderse como una primera solución sólida para el alcance académico del trabajo, no como modelos definitivos de operación productiva.

El VAR aporta una lectura conjunta sobre series alineadas y transformadas. Sus resultados sugieren señales exploratorias de interacción predictiva, especialmente desde POS API hacia ALB dentro del VAR ajustado, pero las pruebas bivariadas de Granger no confirman una relación estable. Por lo tanto, no corresponde afirmar causalidad operacional.

## 8. Referencias

> Completar y adaptar a formato APA en la versión final.

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*. Springer.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Statsmodels Developers. (s. f.). *statsmodels: Statistical modeling and econometrics in Python*. https://www.statsmodels.org/
