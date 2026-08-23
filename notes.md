# Notas de resolución — Series temporales BodegaAI

Este documento resume los hallazgos, decisiones metodológicas y criterios usados en `resolucion_consigna_series_bodegaai.ipynb`. Sirve como guía rápida para revisar la entrega sin tener que reconstruir toda la lógica desde cero.

## Resumen ejecutivo

Se resolvió la consigna usando dos series horarias reales:

| Serie | Archivo | Variable usada | Rango | Observaciones |
|---|---|---:|---|---:|
| ALB producción | `data/alb-prod-requests-2026-07.csv` | `requests` | 2026-07-01 00:00 UTC a 2026-07-31 23:00 UTC | 744 |
| POS API producción | `data/bodegaai-production-pos-api-request-count-per-target-last-30-days-hourly.csv` | `value` | 2026-07-18 23:00 UTC a 2026-08-17 22:00 UTC | 720 |

Decisión central: analizar cada serie por separado con SARIMA y usar VAR solo sobre el período común, porque los rangos temporales no coinciden completos. La ejecución end-to-end del notebook pasó sin errores.

## Hallazgos principales

| Tema | ALB | POS API |
|---|---|---|
| Calidad temporal | Serie horaria continua, sin faltantes ni duplicados. | Serie horaria continua, sin faltantes ni duplicados. |
| Escala promedio | ~2,114,963 requests/hora. | ~829,950 requests por target/hora. |
| Mínimo observado | 883,045 requests, 2026-07-01 15:00 UTC. | 318,307.50 requests por target, 2026-08-07 17:00 UTC. |
| Máximo observado | 2,536,631 requests, 2026-07-11 20:00 UTC. | 924,146.50 requests por target, 2026-08-13 21:00 UTC. |
| Hora promedio más alta | 21:00 UTC. | 19:00 UTC. |
| Hora promedio más baja | 07:00 UTC. | 08:00 UTC. |
| Atípicos por IQR | 43 horas marcadas, principalmente bajas. | 23 horas marcadas, principalmente bajas. |
| Estacionalidad | Evidencia compatible con ciclo diario. | Evidencia compatible con ciclo diario. |

Conclusión práctica: ambas series son de tráfico horario con patrón diario; no conviene modelarlas como ruido independiente ni como series puramente no estacionales.

## Decisiones metodológicas

| Decisión | Criterio |
|---|---|
| Usar frecuencia horaria (`1h`) | Los datos están medidos por ventanas de una hora. Mantener la frecuencia evita distorsionar FAC/FACP y SARIMA. |
| Reindexar a índice horario completo | Permite detectar huecos temporales y evita que los modelos trabajen sobre un calendario implícito incorrecto. |
| Evitar notación científica | Los conteos son grandes; la lectura humana mejora con separadores de miles. |
| Analizar estacionariedad antes de modelar | Una serie no estacionaria puede producir relaciones espurias y modelos mal diagnosticados; aun así, los tests se interpretan junto con la estructura diaria observada. |
| Probar diferencia regular y diferencia estacional | En tráfico horario, `diff(24)` captura el ciclo diario; `diff(1)` captura cambios locales. |
| Usar ADF + KPSS | ADF tiene nula de raíz unitaria; KPSS tiene nula de estacionariedad. Juntas reducen una lectura mecánica: ALB en niveles pasa ambos tests al 5%, pero conserva estructura diaria; POS API muestra evidencia mixta en niveles y mejora con transformación. |
| Usar SARIMA con período estacional `s = 24` | La periodicidad natural esperada es diaria: 24 observaciones por día. |
| Mantener grilla SARIMA chica | La muestra es corta; una búsqueda enorme sobreactúa precisión y aumenta sobreajuste/costo. |
| Separar últimas 48 horas como test | Ventana razonable para validar pronóstico horario sin extrapolar demasiado lejos. |
| Calcular MAE, RMSE y MAPE manualmente | Hace explícita la métrica y evita depender de magia de librerías. |
| Diagnosticar residuos con FAC y Ljung-Box | Si queda autocorrelación en residuos, el modelo todavía dejó estructura sin explicar. |
| Pronosticar 48 horas | Horizonte coherente con frecuencia horaria y tamaño muestral. |
| VAR solo sobre rango común | VAR exige series alineadas en el mismo calendario. Usar rangos distintos sería técnicamente incorrecto. |
| Transformar VAR con `log1p` + `diff(24)` | Estabiliza escala y enfoca el VAR en variaciones estacionales, no en niveles absolutos. |
| Seleccionar rezagos VAR por AIC hasta 24 | Permite capturar dependencia intradía sin abrir una búsqueda excesiva. |
| Interpretar Granger con cautela | Granger indica poder predictivo por rezagos, no causalidad de negocio real. |

## Modelos seleccionados

| Serie | Modelo seleccionado | Motivo |
|---|---|---|
| ALB | `SARIMA(1, 1, 1)x(1, 1, 1, 24)` | Mejor desempeño fuera de muestra dentro de la grilla, con MAPE de test ≈ 0,79%. Incluye diferencia regular y estacional. |
| POS API | `SARIMA(1, 1, 1)x(1, 0, 1, 24)` | Mejor MAPE de test dentro de la grilla, ≈ 3,90%. Existe otro candidato con mejor AIC/BIC, pero peor MAPE fuera de muestra. |
| VAR conjunto | VAR sobre `log1p(series).diff(24)` con 24 rezagos | Seleccionado por AIC sobre el tramo temporal común, con cautela por sensibilidad y posible sobreparametrización. |

## Resolución por punto de la consigna

| Punto | Resolución aplicada |
|---|---|
| Series originales | Se graficaron ALB y POS API en niveles. |
| Estacionariedad | Se explicó el concepto y se evaluaron diferencias `diff(1)`, `diff(24)` y `diff(24).diff(1)`. |
| FAS, FAC y FACP | Se graficaron juntas para cada serie con rezagos hasta 72 horas y marcas en 24/48/72. |
| Raíces unitarias | Se aplicaron ADF y KPSS a versiones originales y diferenciadas. |
| SARIMA | Se comparó una grilla pequeña de modelos con estacionalidad diaria `s=24`. |
| Training/testing | Se reservaron las últimas 48 horas como test. |
| Métricas | Se calcularon MAE, RMSE y MAPE. |
| Comparación de modelos | Se comparó desempeño por MAPE y criterios de información; cuando MAPE fuera de muestra diverge de AIC/BIC, se prioriza MAPE. |
| Diagnóstico | Se revisaron residuos, histograma, FAC de residuos y Ljung-Box. |
| Pronóstico | Se generaron forecasts SARIMA de 48 horas para ambas series. |
| VAR | Se alinearon ambas series por intersección temporal y se ajustó VAR sobre transformación estacionaria. |
| Impulso-respuesta y causalidad | Se graficaron IRF y se aplicaron pruebas de causalidad de Granger. |
| Estacionalidad | Se justificó SARIMA estacional por evidencia de patrón diario. |

## Hallazgos del VAR y causalidad

- El rango común usado para VAR va desde 2026-07-18 23:00 UTC hasta 2026-07-31 23:00 UTC.
- Luego de `log1p` y diferencia estacional de 24 horas quedaron 289 observaciones para VAR.
- Las métricas VAR reportadas son MAE/RMSE en escala transformada; MAPE se omite intencionalmente porque los valores transformados pueden acercarse a cero o cambiar de signo.
- El AIC seleccionó 24 rezagos, aunque BIC/HQIC favorecen rezagos más cortos; por eso el resultado puede ser sensible y estar sobreparametrizado.
- El VAR ajustado sugiere dirección predictiva POS API -> ALB, pero las pruebas bivariadas de Granger en rezagos 6/12/24 no lo confirman.
- La lectura final debe ser prudente porque el tramo común entre ambas series es corto y la señal causal es exploratoria/modelo-dependiente, no causalidad operacional.

## Riesgos y caveats

- La muestra es corta: aproximadamente 30/31 días por serie.
- Las series no cubren exactamente el mismo período; esto limita el análisis conjunto.
- Los atípicos detectados por IQR no prueban incidentes; solo marcan horas que merecen inspección.
- SARIMA y VAR pueden ser sensibles a eventos externos no presentes en el dataset.
- Ljung-Box en residuos muestra autocorrelación remanente: ALB rechaza en rezagos 24/48, no en 12; POS API rechaza en 12/24/48. Por eso el pronóstico debe tratarse con cautela diagnóstica.
- Para una decisión productiva real harían falta más historia, validación rolling y contexto operacional.

## Archivos relacionados

| Archivo | Rol |
|---|---|
| `alb_prod_requests_eda_2026_07.ipynb` | EDA individual de requests del ALB. |
| `pos_api_requests_eda_2026_07_08.ipynb` | EDA individual de requests por target del POS API. |
| `resolucion_consigna_series_bodegaai.ipynb` | Resolución principal de la consigna para ambas series. |
| `consigna.md` | Enunciado de la consigna. |
| `data/alb-prod-requests-2026-07.csv` | Dataset ALB. |
| `data/bodegaai-production-pos-api-request-count-per-target-last-30-days-hourly.csv` | Dataset POS API. |

## Decisión final

La representación más adecuada para estas series, dentro del alcance académico y de datos disponibles, es usar modelos SARIMA estacionales por serie con período diario `s = 24`, y complementar con VAR solo para estudiar interacción predictiva en el tramo temporal común. No se interpreta VAR/Granger como causalidad operacional fuerte: se usa como evidencia estadística condicionada a la muestra.
