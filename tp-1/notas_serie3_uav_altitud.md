# Notas de resolución — Tercera serie: altitud de UAV

Documento de decisiones de la tercera serie del TP1 de Análisis de Series Temporales. Registra procedencia y tratamiento de los datos, decisiones metodológicas con su criterio, resultados y limitaciones.

Alcance acotado a esta serie. Las series ALB y POS API están documentadas en `notes.md`.

Notebook asociado: `notebooks/uav_altitude_series_tp1.ipynb`.

> **Cómo usar este documento.** Cada sección indica a qué parte del informe final alimenta, según la estructura del punto 14 de la consigna. El informe se redacta más adelante, una vez cerradas las tres series; acá sólo se registra el material.

---

## Resumen ejecutivo

> Alimenta: **Resumen Ejecutivo** del informe.

La tercera serie es la **altitud relativa de un vehículo aéreo no tripulado de ala fija** durante un vuelo completo, extraída de un registro de telemetría MAVLink real de julio de 2018. La serie cuenta con 457 observaciones a 1 Hz y recorre un rango de 4,9 a 82,8 metros.

La serie resulta no estacionaria en niveles y requiere una diferencia regular. El modelo seleccionado por criterio de Akaike es **ARIMA(1, 1, 3)**, con AIC de 1.379,95, todos sus parámetros individualmente significativos al 5 % y residuos indistinguibles de ruido blanco según la prueba de Ljung-Box. Sobre un horizonte de prueba de 15 segundos alcanza un MAE de 6,01 metros y un RMSE de 7,53 metros.

La serie cubre los puntos 1 a 9 y 12 de la consigna. No participa del modelo VAR de los puntos 10 y 11 por razones de calendario que se detallan más abajo.

---

## Procedencia de los datos

> Alimenta: **Introducción** del informe.

| Aspecto | Detalle |
|---|---|
| Ubicación | `data/telemetry/` |
| Formato | `.tlog` — registro de telemetría MAVLink |
| Contenido de la carpeta | 41 vuelos repartidos en 5 jornadas de 2018: `2018-07-18`, `2018-07-30`, `2018-09-11`, `2018-10-05`, `2018-10-18` |
| Vuelos utilizables | 35 (seis archivos `.tlog` están vacíos) |
| Peso total | 430 MB |
| Archivos por vuelo | `flight.tlog`, `flight.tlog.raw`, `mav.parm` |

**Qué es un tlog.** MAVLink es el protocolo con el que un piloto automático transmite su estado a la estación de tierra. Un `.tlog` es el registro binario de esa transmisión: cada mensaje va precedido de una marca temporal de 8 bytes en microsegundos. El archivo `mav.parm` acompaña a cada vuelo con el volcado completo de parámetros de configuración del piloto automático.

**Qué aeronave los generó.** Los parámetros identifican una aeronave de **ala fija con ArduPlane**, no un multirrotor:

| Parámetro | Valor | Lectura |
|---|---|---|
| `Q_ENABLE` | 0 | No es un quadplane |
| `TRIM_ARSPD_CM` | 1500 | Velocidad de crucero de 15 m/s |
| `WP_LOITER_RAD` | 40 | Radio de espera de 40 metros |
| `LIM_ROLL_CD` | 4500 | Alabeo máximo de 45 grados |
| `ARSPD_USE` / `ARSPD_TYPE` | 1 / 1 | Sensor de velocidad aerodinámica configurado y en uso |

Cada tlog contiene alrededor de veinticinco tipos de mensaje distintos. Las fuentes de altitud disponibles son `GLOBAL_POSITION_INT.relative_alt` (milímetros sobre el punto de despegue), `VFR_HUD.alt` (metros sobre el nivel del mar) y `SCALED_PRESSURE.press_abs` (presión barométrica).

**Los datos no se versionan.** La regla `data/*` de `.gitignore` los excluye y no hay ningún archivo de datos registrado en el historial. Pesan 482 MB entre los tlogs y el CSV descartado, contra 2,1 MB del repositorio completo. La trazabilidad se resuelve por esta documentación: la sección siguiente detalla la cadena de tratamiento con precisión suficiente para reconstruir la serie desde el archivo crudo.

---

## Selección y tratamiento

> Alimenta: **Introducción** y **Apéndices** del informe.

### Cadena completa

| Paso | Operación | Resultado |
|---|---|---|
| 1 | Lectura de `data/telemetry/2018-07-18/flight7/flight.tlog` con `pymavlink` (desde `notebooks/`, la ruta es `../data/...`) | — |
| 2 | Filtrado de mensajes `GLOBAL_POSITION_INT`, campo `relative_alt` | 3.784 mensajes |
| 3 | Conversión de milímetros a metros | — |
| 4 | Indexación por marca temporal | 19:52:17 a 20:07:45 UTC, intervalo mediano de 0,241 s (4,15 Hz) |
| 5 | Remuestreo a grilla regular de 1 Hz por promedio | 0 intervalos vacíos |
| 6 | Máscara de vuelo `altitud > 3 m` y bloque contiguo más largo | **457 observaciones** |

### Serie resultante

| Característica | Valor |
|---|---|
| Observaciones | 457 |
| Frecuencia | 1 Hz |
| Duración | 457 s (7,6 min) |
| Altitud mínima | 4,9 m |
| Altitud máxima | 82,8 m |
| Altitud media | 46,9 m |
| Rango | 77,9 m |
| Valores faltantes | Ninguno |

El vuelo recorre un perfil completo de despegue, ascenso, crucero y descenso.

### Relevamiento previo de vuelos

El relevamiento abarcó los 35 tlogs no vacíos y midió el bloque contiguo en vuelo, el rango de altitud y la disponibilidad de sensores. Los candidatos finales fueron:

| Vuelo | n a 1 Hz | Rango altitud | ADF (p) | KPSS (p) | Ljung-Box(20) con d=1 |
|---|---:|---|---:|---:|---:|
| **2018-07-18/flight7** | **457** | 5 a 83 m | 0,124 | 0,100 | 1e-62 |
| 2018-07-18/flight3 | 428 | 4 a 85 m | 0,139 | 0,100 | 9e-75 |
| 2018-07-30/flight1 | 583 | 6 a 80 m | 0,017 | 0,100 | 4e-68 |
| 2018-07-30/flight4 | 210 | 3 a 86 m | 0,349 | 0,010 | 2e-40 |
| 2018-07-30/flight5 | 226 | 5 a 91 m | 0,856 | 0,073 | 9e-92 |

Dos hallazgos del relevamiento condicionaron la selección:

**La mayor parte de cada registro corresponde a la aeronave en tierra.** En `2018-09-11/flight6`, por ejemplo, el registro dura 2.880 segundos pero la mediana de velocidad respecto del suelo es de 0,16 m/s: casi todo el archivo es deriva del barómetro con el motor en ralentí. Sin filtrar, se modelaría ruido de sensor en lugar de vuelo.

**El sensor de velocidad aerodinámica dejó de reportar a partir de septiembre de 2018.** Todos los vuelos de `2018-09-11`, `2018-10-05` y `2018-10-18` registran `airspeed = 0`, pese a que los parámetros indican el sensor configurado y en uso. Se trata de una falla del pitot, no de configuración. El dato restringe cualquier análisis multivariado a las jornadas de julio.

---

## Fuente descartada: `SurveilDrone-Net23.csv`

> Alimenta: **Introducción** del informe.

El conjunto `data/SurveilDrone-Net23.csv` (52 MB, 140.256 filas, 34 columnas, grilla de 15 minutos entre 2021-01-01 y 2024-12-31) fue la primera candidata para esta tercera serie. Fue **descartado por carecer de estructura temporal**.

### Evidencia

Las veinticinco variables numéricas constituyen ruido blanco independiente e idénticamente distribuido. La función de autocorrelación permanece por debajo de la banda de significatividad, que para n = 140.256 se ubica en ±0,0052:

| Nivel de agregación | ACF rezago 1 | ACF rezago estacional | ACF rezago semanal |
|---|---|---|---|
| Grilla de 15 min (n = 140.256) | \|r\| < 0,008 | \|r\| < 0,006 (96 = día) | \|r\| < 0,006 (672 = semana) |
| Media diaria (n = 1.461) | \|r\| < 0,053 | \|r\| < 0,072 (7 = semana) | \|r\| < 0,040 (365 = año) |

Las pruebas de hipótesis confirman la lectura sobre todas las candidatas evaluadas:

| Serie candidata | n | ADF (p) | KPSS (p) | Ljung-Box(10) | Ljung-Box(20) |
|---|---:|---:|---:|---:|---:|
| `power_consumption_watts` 15 min | 140.256 | ~0 | 0,10 | 0,660 | 0,754 |
| `power_consumption_watts` diaria | 1.461 | ~0 | 0,10 | 0,859 | 0,788 |
| `detected_object_count` diaria | 1.461 | ~0 | 0,10 | 0,372 | 0,496 |
| `ambient_temp_C` diaria | 1.461 | ~0 | 0,10 | 0,588 | 0,564 |
| `detection_confidence_avg` diaria | 1.461 | ~0 | 0,10 | 0,442 | 0,708 |

La prueba de Ljung-Box **nunca** rechaza la hipótesis de incorrelación, la ADF rechaza la raíz unitaria con contundencia y la KPSS no rechaza la estacionariedad. Es la firma de un proceso sin memoria.

### Indicios de origen sintético

Tres elementos adicionales confirman que el conjunto no proviene de un proceso generador temporal:

1. **`ambient_temp_C` permanece plana en 25,0 °C ± 0,15 durante los 48 meses** de 2021 a 2024. Una temperatura ambiente real presenta ciclo anual.
2. **Contiene exactamente 96 registros por día durante los 1.461 días**, sin huecos ni duplicados. La grilla es perfectamente rectangular.
3. **`mission_id` abarca una mediana de 1.303 días por misión**, con un máximo de 1.459. Las misiones no son bloques contiguos: la etiqueta fue asignada al azar sobre todo el rango temporal.

### Consecuencia

Una serie de ruido blanco es estacionaria por construcción. Con ella, el punto 2 no requiere diferenciación, el punto 5 seleccionaría un ARIMA(0, 0, 0) sin parámetros, el punto 9 produciría una recta en la media y el punto 12 carecería de objeto. Los puntos 5 a 9 quedarían vacíos de contenido analítico. Por ese motivo se recurrió a la telemetría MAVLink.

---

## Decisiones metodológicas

> Alimenta: **Análisis de Resultados** del informe.

| Decisión | Criterio | Alternativa descartada |
|---|---|---|
| Usar telemetría MAVLink en lugar de `SurveilDrone-Net23.csv` | El CSV es ruido blanco i.i.d. y deja vacíos los puntos 5 a 9 | Conservar el CSV y documentar honestamente su carácter de ruido blanco. Descartada porque cinco puntos de la consigna quedarían sin contenido |
| Vuelo `2018-07-18/flight7` | Mejor equilibrio entre longitud (457 obs, comparable a las 744 y 720 de las otras series), perfil de vuelo completo y ausencia de interpolación | `2018-07-30/flight4`, que da no estacionariedad inequívoca (ADF 0,349 con KPSS 0,010) pero con 210 observaciones, menos de la mitad. `2018-07-18/flight8`, descartado porque nunca supera los 6 metros: es carreteo, no vuelo. `2018-07-30/flight1`, descartado porque su ADF rechaza la raíz unitaria (p = 0,017) y debilita la justificación de diferenciar |
| Variable `GLOBAL_POSITION_INT.relative_alt` | Referida al punto de despegue (0 a 85 m), de lectura directa; muestrea a 4,15 Hz; permite detectar el contacto con el suelo sin conocer la elevación del campo | `VFR_HUD.alt`, que arrastra un desplazamiento sobre el nivel del mar de unos 345 m sin valor informativo y muestrea a 2,6 Hz |
| Máscara de vuelo `altitud > 3 m` | Preserva despegue y descenso, que aportan la dinámica más informativa | Umbral de 15 m combinado con velocidad respecto del suelo mayor a 5 m/s, que reduce el tramo a 428 s sin mejorar el análisis |
| Bloque contiguo más largo | Evita concatenar segmentos separados por interrupciones y romper la continuidad temporal que los modelos suponen | Conservar todos los tramos que cumplen la máscara |
| Remuestreo a 1 Hz | Es un submuestreo desde los 4,15 Hz nativos y no deja intervalos vacíos: no se introducen puntos interpolados | 2 Hz, que llevaría a 915 observaciones pero exige interpolar el 1,2 % de los intervalos. Esos puntos inflan artificialmente la autocorrelación de rezago corto, justo la magnitud sobre la que se apoya la identificación del modelo |
| Diferencia regular `d = 1` fija en toda la grilla | La FAC en niveles y las pruebas de raíces unitarias ya justifican una única diferencia | Explorar también `d = 0` y `d = 2`, que agregaría filas sin información |
| Grilla ARIMA(p, 1, q) con p y q entre 0 y 3 | La muestra es corta; una búsqueda amplia sobreactúa precisión y favorece el sobreajuste | Grilla más extensa |
| Selección por AIC | Penaliza la cantidad de parámetros y permite comparar modelos no anidados | Selección por desempeño predictivo únicamente, que no discrimina entre órdenes vecinos |
| Sin componente estacional | Los picos del periodograma no se verifican en la FAC de la serie diferenciada | Forzar un período estacional arbitrario |
| Horizonte de prueba de 15 s | Poco más del 3 % de la muestra; ventana de anticipación razonable para supervisión de vuelo | 45 s, que eleva el MAE a 18,2 m porque abarca la maniobra de descenso completa |
| Notebooks agrupados en `notebooks/` | Separa el código de la documentación destinada al informe, que permanece en la raíz | Mantener los notebooks en la raíz. El movimiento obligó a reescribir las rutas de datos a `../data/` |
| Snippets de código breves y consolidados en el informe | El punto 13 pide incluir los códigos empleados, pero el límite de 40 carillas no admite volcar los notebooks completos | Transcribir el código íntegro, que consumiría el presupuesto de páginas sin aportar al análisis |
| Significatividad global por razón de verosimilitud | El punto 5 exige significatividad individual **y global**; los criterios de información por sí solos no constituyen un contraste de hipótesis | Apoyarse únicamente en AIC y BIC junto al diagnóstico de residuos, que es defendible pero no es un test global |
| Excluir la serie del modelo VAR | Calendarios disjuntos: 2018 contra 2026, sin un solo instante compartido | Construir un VAR interno al vuelo con altitud y velocidad aerodinámica, que da causalidad de Granger bidireccional con p ≈ 1e-33 e interpretación física de intercambio entre energía potencial y cinética. Descartada por decisión de alcance: los puntos 10 y 11 se resuelven con ALB y POS API |

---

## Resultados por punto de la consigna

> Alimenta: **Análisis de Resultados** del informe.

### Punto 2 — Serie original y diferenciación

La serie en niveles presenta media no constante: se desplaza de manera sostenida a lo largo del vuelo conforme al perfil de despegue, ascenso, crucero y descenso. La primera diferencia oscila alrededor de cero con dispersión estable.

### Punto 3 — FAS, FAC y FACP

En niveles, la FAC decae de manera lenta y aproximadamente lineal, con ACF(1) = 0,96, y permanece muy por encima de la banda durante decenas de rezagos. La FACP presenta un primer rezago cercano a la unidad y se corta de inmediato. La teoría asocia ese par de patrones a un **proceso integrado**: la persistencia proviene de una raíz unitaria, no de memoria autorregresiva de orden alto.

Tras diferenciar, la FAC conserva significatividad sólo en los primeros rezagos, lo cual sugiere una componente de medias móviles de orden bajo.

### Punto 4 — Raíces unitarias

| Serie | ADF (p) | ADF concluye | KPSS (p) | KPSS concluye | Ljung-Box(20) |
|---|---:|---|---:|---|---:|
| Niveles | 0,1237 | no estacionaria | 0,1000 | estacionaria | 0,000e+00 |
| Diferencia `d = 1` | 0,0000 | estacionaria | 0,1000 | estacionaria | 1,115e-62 |

**Las dos pruebas discrepan sobre la serie en niveles.** La ADF no rechaza su nula de raíz unitaria; la KPSS tampoco rechaza su nula de estacionariedad. La discrepancia es frecuente en muestras de tamaño moderado y no invalida el análisis: indica que la evidencia muestral no basta para pronunciarse de manera categórica con un único contraste.

La decisión de diferenciar se sostiene sobre tres elementos convergentes: el desplazamiento visible de la media, el decaimiento lento de la FAC junto con el corte de la FACP, y la ausencia de rechazo por parte de la ADF.

Sobre la serie diferenciada, la prueba de Ljung-Box rechaza la incorrelación con p = 1,115e-62. **Ese es el resultado central**: tras eliminar la raíz unitaria subsiste estructura autocorrelacionada genuina, y por lo tanto existe un modelo ARMA por estimar.

### Puntos 5 y 7 — Estimación y comparación

Se estimaron dieciséis modelos. Los cinco mejores por AIC:

| Modelo | AIC | BIC | MAE | RMSE |
|---|---:|---:|---:|---:|
| **ARIMA(1,1,3)** | **1.379,95** | 1.400,40 | 6,013 | 7,534 |
| ARIMA(0,1,3) | 1.380,36 | 1.396,72 | 5,989 | 7,567 |
| ARIMA(2,1,3) | 1.381,68 | 1.406,21 | 6,004 | 7,552 |
| ARIMA(3,1,3) | 1.383,19 | 1.411,81 | 5,991 | 7,553 |
| ARIMA(3,1,1) | 1.387,43 | 1.407,87 | 6,007 | 7,495 |

Las diferencias de MAE y RMSE entre los mejores candidatos son pequeñas: el desempeño predictivo no discrimina con nitidez entre órdenes vecinos. El criterio de información opera como desempate.

**Significatividad individual del modelo seleccionado.** Los cuatro parámetros resultan significativos al 5 %:

| Parámetro | Coeficiente | Error estándar | z | P>\|z\| |
|---|---:|---:|---:|---:|
| `ar.L1` | −0,2650 | 0,120 | −2,208 | 0,027 |
| `ma.L1` | 1,6263 | 0,113 | 14,432 | 0,000 |
| `ma.L2` | 1,2023 | 0,136 | 8,821 | 0,000 |
| `ma.L3` | 0,4268 | 0,061 | 6,959 | 0,000 |
| `sigma2` | 1,3022 | 0,051 | 25,344 | 0,000 |

Log-verosimilitud: −684,976. Observaciones de entrenamiento: 442.

**Significatividad global.** El contraste de razón de verosimilitud compara ARIMA(1, 1, 3) contra el modelo restringido ARIMA(0, 1, 0), un paseo aleatorio sin parámetros autorregresivos ni de medias móviles. La hipótesis nula sostiene que los cuatro parámetros son conjuntamente nulos.

| Elemento | Valor |
|---|---:|
| Log-verosimilitud ARIMA(1, 1, 3) | −684,976 |
| Log-verosimilitud ARIMA(0, 1, 0) | −968,532 |
| Estadístico LR | 567,11 |
| Grados de libertad | 4 |
| Valor p (chi-cuadrado) | 2,029e-121 |

La nula se rechaza de manera contundente: el conjunto de parámetros aporta capacidad explicativa muy por encima del modelo trivial.

### Punto 6 — Desempeño entre entrenamiento y prueba

| Métrica | Valor |
|---|---|
| Entrenamiento | 442 observaciones |
| Prueba | 15 observaciones |
| MAE | 6,01 m |
| RMSE | 7,53 m |
| Rango de la serie | 77,9 m |

El MAE representa alrededor del 7,7 % del rango recorrido por la serie.

### Punto 8 — Diagnóstico de residuos

| Rezago | Estadístico Ljung-Box | Valor p |
|---:|---:|---:|
| 10 | 3,213 | 0,976 |
| 20 | 17,337 | 0,631 |
| 30 | 29,587 | 0,487 |

Media de los residuos: 0,0136 m. Desvío estándar: 1,1644 m.

La prueba **no rechaza la incorrelación en ninguno de los rezagos examinados**: los residuos son indistinguibles de ruido blanco. La FAC de los residuos lo confirma, con todos los coeficientes dentro de la banda. El modelo no deja estructura autocorrelacionada sin explicar.

El estadístico de Jarque-Bera (431,74) rechaza la normalidad, con asimetría de 0,51 y curtosis de 7,74. Los apartamientos en las colas son atribuibles a las maniobras del vuelo y afectan la validez de los intervalos de confianza nominales, no la estimación puntual.

### Punto 9 — Pronóstico

Horizonte de 15 segundos con intervalo de confianza del 95 %. La ventana guarda coherencia con la frecuencia de muestreo de 1 Hz y con el tamaño de la serie.

### Punto 12 — Estacionalidad

**Esta serie no presenta estacionalidad.** Se examinó la presencia de periodicidad mediante periodograma sobre la serie diferenciada; los picos detectados no se verifican en la FAC, donde los rezagos correspondientes a los períodos candidatos permanecen dentro de la banda.

Un caso ilustrativo del riesgo de leer el periodograma de manera aislada: en `2018-09-11/flight6` el periodograma marcaba un período dominante de 18,3 segundos, pero la FAC de la serie diferenciada en el rezago 18 arroja 0,042, plenamente dentro de la banda. El pico era espurio, producto del ruido barométrico en tierra.

La ausencia responde a la naturaleza del fenómeno: la altitud de una aeronave durante un vuelo único obedece al plan de vuelo y a la acción del piloto automático, sin ciclo repetitivo de período fijo.

**Alcance de la comparación solicitada.** El punto 12 pide comparar el modelo estacional con "los modelos determinados en el trabajo anterior". La expresión se interpreta como referida a los modelos estimados con anterioridad **dentro de este mismo trabajo práctico**, esto es, las especificaciones no estacionales de la grilla del punto 5. No remite a un trabajo práctico previo de la materia. Dado que esta serie carece de componente estacional, la comparación no corresponde: la grilla del punto 5 ya contiene la especificación adecuada. El requerimiento queda cubierto por las otras dos series, cuyo tráfico horario presenta estacionalidad diaria con s = 24.

---

## Caveats y limitaciones

> Alimenta: **Conclusiones** del informe.

| Limitación | Alcance |
|---|---|
| Muestra de 457 observaciones | Acota la confianza de los criterios de información y limita el tamaño de la grilla de modelos que tiene sentido explorar |
| El tramo de prueba coincide con el descenso | Fase dinámicamente distinta del crucero donde el modelo se entrena mayoritariamente. La métrica obtenida es **conservadora**: sobre un tramo de crucero el error sería menor. Se informa este valor en lugar de seleccionar una ventana más favorable |
| ADF y KPSS discrepan en niveles | La evidencia de no estacionariedad es convergente pero no unánime. La decisión de diferenciar se apoya en tres elementos, no en un único contraste |
| Los residuos no son normales | Jarque-Bera rechaza con p < 0,001. Afecta la validez de los intervalos de confianza nominales del pronóstico, no la estimación puntual |
| Un único vuelo | Los resultados no se generalizan a otros vuelos ni a otras aeronaves. El modelo describe este vuelo |
| La serie queda fuera del VAR | Calendario disjunto respecto de ALB y POS API. Los puntos 10 y 11 se resuelven con esas dos series, que solapan sólo 13 días, limitación ya registrada en `notes.md` |
| Extrapolación de largo plazo inválida | La trayectoria depende de decisiones del plan de vuelo que el modelo no observa. Con horizonte de 45 s el MAE trepa a 18,2 m |
| Los datos no acompañan al repositorio | La reproducción exige disponer de `data/telemetry/`. La cadena de tratamiento documentada más arriba permite reconstruir la serie desde el archivo crudo |

---

## Archivos relacionados

| Archivo | Rol |
|---|---|
| `notebooks/uav_altitude_series_tp1.ipynb` | Resolución de los puntos 1 a 9 y 12 sobre esta serie |
| `notas_serie3_uav_altitud.md` | Este documento |
| `notebooks/resolucion_consigna_series_bodegaai.ipynb` | Series ALB y POS API; resuelve los puntos 10 y 11 |
| `notes.md` | Decisiones de las series ALB y POS API |
| `consigna.md` | Enunciado |
| `data/telemetry/2018-07-18/flight7/` | Registro de telemetría de origen (no versionado) |
