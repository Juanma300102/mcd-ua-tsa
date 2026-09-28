# Compilación del informe

## Requisitos

TeX Live 2023 (verificado en este entorno) o superior. Paquetes necesarios:

```
apa7  biblatex  biblatex-apa  biber  listings  setspace  csquotes
babel-spanish  fontspec  geometry  booktabs  graphicx  longtable  xcolor  amsmath
```

Instalación (si faltara alguno):

```bash
tlmgr install apa7 biblatex biblatex-apa biber listings setspace csquotes
```

Si `tlmgr` reporta la instalación como exitosa pero `kpsewhich <paquete>.sty` no
encuentra el archivo, la causa es un desajuste de versión entre la instalación
local y el espejo remoto. Se resuelve actualizando la distribución:

```bash
curl -fsSL -o update-tlmgr-latest.sh \
  https://mirror.ctan.org/systems/texlive/tlnet/update-tlmgr-latest.sh
sh update-tlmgr-latest.sh --nox11 -- --upgrade
tlmgr option repository https://mirror.ctan.org/systems/texlive/tlnet
```

## Secuencia de compilación

Se compila con XeLaTeX por el manejo nativo de UTF-8, que el texto en español
requiere. La secuencia completa, desde el directorio `informe/`:

```bash
xelatex informe.tex
biber informe
xelatex informe.tex
xelatex informe.tex
```

Las tres pasadas de XeLaTeX son necesarias: la primera genera las referencias
cruzadas, `biber` resuelve la bibliografía, y las dos siguientes estabilizan el
índice y los números de página.

## Verificación

```bash
pdfinfo informe.pdf | grep Pages
rg -c "^! " informe.log        # debe ser 0 (o sin coincidencias)
rg -i "undefined" informe.log  # no debe haber referencias ni citas sin resolver
```

Última compilación verificada: 40 páginas totales. Sin contar la carátula ni los apéndices (páginas 2 a 31, con
resumen, índices, cuerpo, declaración de IA y referencias) son 30 páginas, exactamente el límite de la consigna;
solo el cuerpo (Introducción a Conclusiones) ocupa 22. 0 errores de LaTeX, 0 referencias o citas sin resolver.

## Nota sobre la serie 3

La tercera serie es pública: visitas horarias de usuarios a `es.wikipedia` (API de Wikimedia, CC0). Se evalúa
como contraste acotado con seis modelos rápidos (`notebooks/07_wikipedia_baseline.ipynb`); las tablas y figuras
del informe salen de `results/07_wikipedia*.csv` y `informe/figuras/07_wikipedia_*.png`. Los datos se
regeneran con `experiments/00_download_wikipedia_es.py`.
