# Compilación del informe

## Requisitos

Distribución TeX Live 2026 (o TinyTeX actualizado a 2026). Paquetes necesarios:

```
apa7  biblatex  biblatex-apa  biber  apacite  listings  setspace
fancyhdr  csquotes  xstring  scalerel  pgf  titlesec  endfloat
babel-spanish  fontspec  geometry  booktabs  graphicx  caption  amsmath
```

Instalación:

```bash
tlmgr install apa7 biblatex biblatex-apa apacite listings setspace \
  fancyhdr csquotes xstring scalerel pgf titlesec endfloat
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
```
