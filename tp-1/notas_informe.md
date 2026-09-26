# Notas de resolución — Informe del TP1

Documento de decisiones de la etapa de redacción del informe. Registra las correcciones aplicadas al análisis previo y las decisiones editoriales.

Alcance acotado a esta etapa. Las decisiones de las series están en `notes.md` (ALB y POS API) y `notas_serie3_uav_altitud.md` (altitud del UAV).

---

## Correcciones aplicadas antes de redactar

> Alimenta: **Análisis de Resultados** y **Conclusiones** del informe.

Dos resultados del análisis previo admitían objeción y se resolvieron antes de publicarlos.

### Corrección 1 — Selección de rezagos del VAR

**Problema.** El modelo VAR seleccionaba 24 rezagos por criterio de Akaike, lo cual implica estimar 98 parámetros con 241 observaciones de entrenamiento. La relación entre parámetros y datos resultaba insostenible.

**Evidencia reunida.** Los cuatro criterios de selección no coinciden:

| Criterio | Rezagos sugeridos |
|---|---:|
| Akaike (AIC) | 24 |
| Bayesiano (BIC) | **2** |
| Hannan-Quinn (HQIC) | **2** |
| Error de predicción final (FPE) | 24 |

El desempeño fuera de muestra a 48 horas, medido en la escala original, resuelve el empate:

| Rezagos | Parámetros | MAE ALB | MAPE ALB | MAE POS API | MAPE POS API |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 31.287 | 1,48 % | 32.128 | 4,67 % |
| **2** | **10** | **27.826** | **1,31 %** | **31.455** | **4,59 %** |
| 3 | 14 | 26.275 | 1,24 % | 30.781 | 4,51 % |
| 24 | 98 | 36.008 | 1,71 % | 26.932 | 4,03 % |

**Decisión.** Se adopta **VAR(2)**, respaldado por dos criterios de información y por el desempeño fuera de muestra.

**Criterio.** El VAR(24) exhibe sobreajuste de manual: con casi diez veces más parámetros, predice **peor** la serie ALB que el VAR(2). Su ventaja en POS API es marginal y no compensa el costo en parsimonia. Dos criterios de información convergen en 2 rezagos.

**Alternativa descartada.** VAR(3), que mejora levemente ambas métricas, pero ningún criterio de información lo respalda. La diferencia frente a VAR(2) es inferior al 6 % en MAE y no justifica apartarse del criterio.

**Consecuencia sobre los resultados.** La causalidad de Granger cambia de lectura. Con VAR(2) el resultado es **unidireccional y nítido**:

| Dirección | F | p | Decisión al 5 % |
|---|---:|---:|---|
| POS API → ALB | 3,554 | 0,0294 | rechaza H₀ |
| ALB → POS API | 0,065 | 0,9369 | no rechaza H₀ |

### Corrección 2 — MAPE calculado sobre la serie transformada

**Problema.** El MAPE de la serie ALB alcanzaba 97,20 %, cifra que sugería un modelo inservible.

**Diagnóstico.** No era un error del modelo. La métrica se calculaba sobre la serie transformada mediante `log1p` y diferencia estacional de 24 horas, que oscila en torno a cero. El MAPE divide por el valor observado y se vuelve inestable cuando el denominador se aproxima a cero.

**Decisión.** Las métricas del VAR se calculan sobre la **escala original**, en solicitudes, tras invertir la transformación.

**Criterio.** Una métrica porcentual sólo es interpretable cuando el denominador tiene una escala estable y alejada de cero. La inversión de la transformación es determinista y no introduce supuestos.

**Procedimiento de inversión.** Para cada instante se recupera el logaritmo mediante la suma del valor rezagado 24 horas más la diferencia pronosticada, y se aplica la función exponencial inversa. Más allá del horizonte de 24 pasos el procedimiento se vuelve recursivo, puesto que el valor rezagado ya es un pronóstico.

**Resultado.** El MAPE pasa de 97,20 % a **1,31 %** en ALB y **4,59 %** en POS API. La capacidad predictiva del modelo era razonable; lo defectuoso era la medición.

---

## Decisiones editoriales

> Alimenta: decisiones de forma; no se traslada al informe.

| Decisión | Criterio | Alternativa descartada |
|---|---|---|
| LaTeX con clase `apa7`, compilado con XeLaTeX | Control del límite de 40 carillas, formato APA nativo y manejo de UTF-8 para el español | HTML impreso desde el navegador, que no controla el salto de página; Word, con formato APA aproximado |
| Actualizar la distribución TeX a la versión 2026 | El gestor de paquetes reportaba instalaciones exitosas que no ocurrían, por desajuste entre la instalación local y los espejos remotos | Redactar sin `apa7` y reproducir APA a mano, con resultado más frágil |
| Una fuente LaTeX por sección | Localiza los errores de compilación y evita navegar un documento extenso | Archivo único |
| 18 figuras en el cuerpo, resto en apéndice | Una figura se gana el lugar cuando es la evidencia de una afirmación del texto | Incluir las 35, que consumirían doce carillas |
| Paneles combinados de dos series | Reduce cuatro figuras a dos y habilita la comparación directa | Extraer las figuras individuales tal como estaban |
| Comparaciones numéricas como tabla | Los gráficos de comparación por métrica ya cargaban los números en el título | Conservarlos como figura |
| Patrón por día de la semana al apéndice | Exhibir un patrón semanal sin modelarlo abre una pregunta que el trabajo no responde | Incluirlo en el cuerpo |
| Fuentes de datos identificadas sólo en Referencias | El cuerpo describe el origen en prosa; APA exige citar los conjuntos de datos | Prohibición estricta, que incumpliría APA |

---

## Observación pendiente

El mapa de calor de día por hora revela, además del ciclo diario modelado, un **efecto de día de la semana** en la serie ALB: los miércoles y jueves presentan niveles sensiblemente menores. El trabajo no modela estacionalidad semanal con período 168. Corresponde señalarlo entre las limitaciones y no omitirlo, dado que la figura lo exhibe.

---

## Integración del commit remoto `58bceca`

> Alimenta: no se traslada al informe. Registro de la resolución de la divergencia.

El repositorio remoto contenía un commit de Juan Manuel Pedrozo que no estaba en la rama local. Ambas ramas partían de `6072789`: la local agregó la tercera serie, el informe en LaTeX y la reorganización de directorios; la remota reelaboró el notebook de las series de tráfico y redactó un informe propio en Markdown.

### Hallazgo de fondo

Ambos trabajos diagnosticaron de manera independiente los mismos dos problemas del análisis original, y los resolvieron de manera distinta:

| Diagnóstico | Resolución remota | Resolución adoptada |
|---|---|---|
| El error porcentual carece de sentido sobre la serie transformada | Eliminarlo de las métricas del VAR y rotular la escala | Invertir la transformación y reportar 1,31 % en escala original |
| El VAR con 24 rezagos está sobreparametrizado | Conservarlo con una advertencia | Adoptar 2 rezagos, con la comparación fuera de muestra como evidencia |

La coincidencia de ambos diagnósticos, alcanzados por separado, respalda las decisiones metodológicas del informe.

### Resolución archivo por archivo

| Archivo | Decisión | Criterio |
|---|---|---|
| `.gitignore` | **Rechazado** el cambio remoto | La regla `*.pdf` excluiría `informe/informe.pdf`, que es el entregable evaluado |
| `consigna.md` | **Rechazado** el borrado remoto | Es la referencia contra la cual se verifica la cobertura de los catorce puntos |
| `notes.md` | Tomada la versión remota | Refina las tablas de decisión y documenta la tensión entre criterios de información y desempeño fuera de muestra |
| `resolucion_consigna_series_bodegaai.ipynb` | Tomada la versión remota, ubicada en `notebooks/` | Conserva el trabajo del coautor bajo la organización local |
| `informe_resolucion_bodegaai.md` | Incorporado como antecedente | Abarca dos de las tres series y no cumple los puntos 1, 13 ni 14; se conserva porque documenta el trabajo previo |

### Mejoras incorporadas

| Mejora | Efecto |
|---|---|
| Funciones de formato numérico castellano | Miles con punto y decimales con coma en las tablas y ejes del notebook consolidado. Se pasó de 0 a 42 valores en convención local, sin ninguno en anglosajona |
| `formatear_eje_temporal` con `ConciseDateFormatter` | Marcas de fecha distribuidas y etiquetas rotadas, sin superposición |
| Seis figuras del notebook remoto | Reemplazan a las extraídas de la versión anterior: series en niveles, diferencias, FAS/FAC/FACP de ambas series, diagnóstico de residuos y pronósticos |

### Figuras descartadas del notebook remoto

Las de las celdas 35, 39 y 42 corresponden al modelo VAR de 24 rezagos y **contradirían el informe**, que adopta 2. Las de las celdas 23 y 26 tampoco se incorporaron: el informe convirtió ambas en tablas.

Las tres figuras de elaboración propia (`panel_heatmap`, `panel_perfil_horario` y `var2_pronostico`) se regeneraron con el mismo formato. Adoptar la convención castellana en seis figuras y dejar tres en anglosajona habría producido una inconsistencia visible dentro del mismo documento.

### Verificación posterior

El informe conserva contenido, tablas, cifras y conclusiones. La integración mejoró la presentación, no el análisis. Tras recompilar: 37 páginas, cuerpo de 29 carillas, cero referencias cruzadas pendientes, 17 citas y 17 entradas bibliográficas sin huérfanas, y las seis cifras clave siguen coincidiendo con el notebook consolidado.

### Coordinación pendiente

El informe remoto abarca dos de las tres series. Conviene confirmar con el coautor que el informe final es el elaborado en LaTeX. La integración preserva ambos documentos, de modo que esa conversación no condiciona el resultado técnico.
