"""Generator for the single-file deliverable notebook of TP Final (Analisis de Series Temporales).

Why this script exists
-----------------------
The course rules (`tp-final/consignas.md`) let each group member upload at most two files: the
PDF report and one script/notebook. The professor therefore never sees `tsa_final/`, the CSVs
under `results/`, or this repository -- only whatever this generator writes to
`tp-final/entrega/TP2_Series_Temporales.ipynb`. That notebook cannot contain `import tsa_final`
(there is no package to import), so this script inlines the source of every `tsa_final` module
directly into notebook cells, in one shared execution namespace.

How the inlining works (read this before touching the module-order/rename tables below)
-----------------------------------------------------------------------------------------
Every `tsa_final` submodule was written as a real Python package member, so it imports its
dependencies the normal package way: `from . import data as _data`, or
`from .evaluation import Fit, Predictor, ...`. Flattened into one notebook namespace, those
import lines are both unnecessary (the names they import are already defined by an earlier cell)
and broken (there is no `tsa_final` package for a relative import to resolve against). Two
different fixes were considered:

  (a) Rewrite every qualified call site (`_data.calendar_features(...)` -> `calendar_features(...)`,
      `_baselines.auto_arima_fit` -> `auto_arima_fit`, etc.) throughout each module's body.
  (b) Keep every qualified call site exactly as written, strip only the import *line* itself, and
      define a small `types.SimpleNamespace` right after the producer module's cell that mimics
      the module object the import used to bind (e.g. `_data = types.SimpleNamespace(load_clean=...,
      calendar_features=..., ...)`).

This script uses (b). It touches far fewer characters per module (one import line removed, one
shim cell added, versus dozens of call sites rewritten across seven files) and is easy to verify
by inspection: every qualified reference inside a module's own body (`_data.load_clean`,
`_baselines.SARIMA_SPECS`, `_hybrid.chronos_zero_shot_fit`, ...) reads identically to the original
package code, because it is the original code, byte for byte, apart from the stripped import
line. The three namespace shims are `_data` (built right after `data.py`'s cell), `_baselines`
(right after `baselines.py`'s cell) and `_hybrid` (right after `hybrid_models.py`'s cell) -- the
only three submodules ever accessed through a qualified `from . import X as _x` alias elsewhere.
Submodules imported unqualified (`from .evaluation import Fit, Predictor, ...`) need no shim at
all: once `evaluation.py`'s cell has run, `Fit`, `Predictor`, `split`, etc. already exist as plain
globals, so the import line is just deleted (replaced by an explanatory comment).

Two real name collisions surface only once everything shares one namespace (invisible inside the
original package, where each module has its own globals):

  1. `baselines.py` does `from statsforecast.models import MSTL` (a StatsForecast model class);
     `hybrid_models.py` does `from statsmodels.tsa.seasonal import MSTL` (a decomposition class).
     Inlined in cell order, the second import would silently overwrite the first, and
     `baselines.mstl_arima_fit` would call the wrong `MSTL` the next time it runs. Fixed with a
     single targeted rename in `baselines.py`'s cell: `MSTL` (the statsforecast import and its one
     call site) becomes `_SF_MSTL`. The unrelated string literal `forecast["MSTL"]` (a DataFrame
     column name StatsForecast itself assigns, not this alias) is deliberately left untouched.
  2. `ml_models.py` and `hybrid_models.py` each define their own private helpers named
     `_EXOG_DTYPES`, `_window_features`, `_exog` and `_future_exog` (near-identical lag/calendar
     feature builders for their own skforecast-based fits). Inlined in cell order, the second
     module's definitions would shadow the first module's, and any later call into
     `ml_models`'s own functions (which look up these names as globals *at call time*, not at
     definition time) would silently execute `hybrid_models`'s version instead. Fixed by
     renaming these four names to `_ml_*` inside `ml_models.py`'s cell and `_hy_*` inside
     `hybrid_models.py`'s cell (word-boundary-safe: none of the four ever appears inside a string
     literal in either file).

`DATA_DIR` (in `data.py`) and `RESULTS_DIR` (in `selection.py`) are each defined in the original
package as `Path(__file__).resolve().parent.parent / "..."` -- meaningless in a notebook cell,
which has no `__file__`. Both are rewritten to point at the notebook's own `RUTA_DATOS` /
`RUTA_RESULTADOS` configuration variables (see the config cell), so every module still reads and
writes the same `data/`/`results/` layout, just configured once at the top of the notebook instead
of derived from a package file path.

Everything else in each module's source -- docstrings, function/class bodies, constants, external
library imports -- is copied verbatim. No behaviour changes; only where names come from changes.

Re-running this script
-----------------------
This script is deterministic: given the same `tsa_final/*.py` sources, `results/*.csv` files and
`pyproject.toml`, it produces byte-identical notebook JSON (aside from a fresh `nbformat` save
timestamp-free structure -- cell `id`s are derived deterministically below). If a future change
touches a module's imports or one of the renamed private names, the assertions in
`transform_module()` fail loudly instead of silently producing a broken notebook; update the
tables in this script to match and re-run. Running this script never re-executes the notebook: it
only regenerates cell *sources*. Execute the result with
`jupyter nbconvert --to notebook --execute --inplace` (see `tp-final/status/07-notebook.md`).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import nbformat as nbf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
TP_FINAL = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
TSA_FINAL = TP_FINAL / "tsa_final"
OUTPUT_NOTEBOOK = TP_FINAL / "entrega" / "TP2_Series_Temporales.ipynb"
PYPROJECT = REPO_ROOT / "pyproject.toml"

MODULE_ORDER = [
    "data",
    "evaluation",
    "baselines",
    "ml_models",
    "dl_models",
    "hybrid_models",
    "selection",
]

MODULE_TITLES = {
    "data": "`data.py` — carga, limpieza e imputación de las series",
    "evaluation": "`evaluation.py` — protocolo de evaluación compartido",
    "baselines": "`baselines.py` — familia *naive*, refit SARIMA de TP1 y extras clásicos",
    "ml_models": "`ml_models.py` — forecasters de Machine Learning (skforecast)",
    "dl_models": "`dl_models.py` — arquitecturas de Deep Learning (darts)",
    "hybrid_models": "`hybrid_models.py` — Prophet/NeuralProphet, híbrido, ensamble, AutoML y Chronos",
    "selection": "`selection.py` — consolidación, contaminación de validación, selección y pronóstico final",
}

MODULE_SUMMARIES = {
    "data": (
        "Registro de las dos series (`SERIES_REGISTRY`), las constantes de fecha acordadas en la "
        "fase 01 (`STEADY_STATE_START`, ventana de intervención, ventana de test) y las funciones "
        "`load_raw`, `load_clean`, `impute_intervention` y `calendar_features`."
    ),
    "evaluation": (
        "El contrato `fit(train) -> predictor(horizon)` que respeta cada modelo del proyecto, el "
        "split fijo train/validación/test (`split`, `train_and_val`), las métricas MAE/RMSE/MAPE/"
        "MASE (`m=24`) y `evaluate()`, que ajusta dos veces (validación, test) y registra cada fila "
        "en un `ResultsTable`."
    ),
    "baselines": (
        "Los modelos más simples posibles (`naive`, `seasonal_naive_m24`, `drift`, `average`), el "
        "refit de las especificaciones SARIMA de TP1 y tres extras clásicos (Holt-Winters, "
        "MSTL+ARIMA, AutoARIMA)."
    ),
    "ml_models": (
        "Features de lags/calendario compartidas, cinco forecasters recursivos (LightGBM, XGBoost, "
        "CatBoost, RandomForest, Ridge), la variante *direct* del mejor de ellos, el tuneo con Optuna "
        "y un stacking de los tres mejores modelos tuneados."
    ),
    "dl_models": (
        "Seis arquitecturas de redes (LSTM, N-BEATS, N-HiTS, TCN, TiDE, TFT) con `darts`, escalado "
        "`log1p` + `Scaler`, covariables de calendario y el cierre con estado `DlModelFit` que fija "
        "la mejor época en el ajuste de validación y la reutiliza, sin *early stopping*, en el refit "
        "de test."
    ),
    "hybrid_models": (
        "Prophet (por defecto y tuneado), NeuralProphet, el híbrido MSTL+LightGBM, un ensamble de "
        "peso igual, AutoGluon TimeSeries, AutoTS y Chronos-Bolt zero-shot."
    ),
    "selection": (
        "La consolidación de los 32 modelos por serie en un solo tablero, las banderas de "
        "contaminación de validación, la regla de selección, la consistencia validación/test "
        "(Spearman), la tabla de sensibilidad y el pronóstico final a 48 h con intervalos."
    ),
}

# ---------------------------------------------------------------------------
# Text transforms applied to each module's source before inlining.
# See the module docstring above for *why* each of these exists.
# ---------------------------------------------------------------------------

# Exact literal replacements (order matters within a module's list).
LITERAL_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "baselines": [
        (
            "from statsforecast.models import MSTL\n",
            "from statsforecast.models import MSTL as _SF_MSTL"
            "  # renombrado: hybrid_models.py importa statsmodels.tsa.seasonal.MSTL bajo el mismo\n"
            "# nombre 'MSTL' -- en un notebook de una sola celda por módulo esa importación posterior\n"
            "# pisaría esta, así que aquí se usa el alias _SF_MSTL en vez de MSTL (ver docstring de\n"
            "# build_deliverable_notebook.py). El literal 'forecast[\"MSTL\"]' más abajo es un nombre\n"
            "# de columna que asigna la propia librería StatsForecast, no este alias, y no se toca.\n"
        ),
        (
            "model = MSTL(season_length=[24, 168], trend_forecaster=_SFAutoARIMA())",
            "model = _SF_MSTL(season_length=[24, 168], trend_forecaster=_SFAutoARIMA())",
        ),
    ],
}

# Word-boundary-safe identifier renames (regex pattern -> replacement), to avoid the two private
# helper modules (ml_models.py / hybrid_models.py) shadowing each other in the shared namespace.
RENAMES: dict[str, list[tuple[str, str]]] = {
    "ml_models": [
        (r"\b_EXOG_DTYPES\b", "_ml_EXOG_DTYPES"),
        (r"\b_future_exog\b", "_ml_future_exog"),
        (r"\b_window_features\b", "_ml_window_features"),
        (r"\b_exog\b", "_ml_exog"),
    ],
    "hybrid_models": [
        (r"\b_EXOG_DTYPES\b", "_hy_EXOG_DTYPES"),
        (r"\b_future_exog\b", "_hy_future_exog"),
        (r"\b_window_features\b", "_hy_window_features"),
        (r"\b_exog\b", "_hy_exog"),
    ],
}

# Intra-package import lines to strip, replaced by an explanatory comment (kept short; the full
# rationale lives in this script's own module docstring, not repeated per module).
IMPORT_STRIPS: dict[str, list[tuple[str, str]]] = {
    "evaluation": [
        (
            "from . import data as _data\n",
            "# `_data` ya quedó definido arriba, justo después de la celda de código de `data.py`\n"
            "# (namespace-shim -- ver la celda de código intermedia). Este notebook no tiene un\n"
            "# paquete `tsa_final` del que importar.\n",
        ),
    ],
    "baselines": [
        (
            "from .evaluation import SEASONAL_PERIOD, Fit, Predictor\n",
            "# SEASONAL_PERIOD, Fit y Predictor ya son nombres globales de este notebook (celda de\n"
            "# `evaluation.py`, más arriba); se usan sin calificar, no hace falta importarlos.\n",
        ),
    ],
    "ml_models": [
        (
            "from . import data as _data\n",
            "# `_data` ya quedó definido arriba (namespace-shim tras la celda de `data.py`).\n",
        ),
        (
            "from .evaluation import Fit, Predictor, ResultsTable, split, train_and_val\n",
            "# Fit, Predictor, ResultsTable, split y train_and_val ya son nombres globales de este\n"
            "# notebook (celda de `evaluation.py`, más arriba).\n",
        ),
    ],
    "dl_models": [
        (
            "from . import data as _data\n",
            "# `_data` ya quedó definido arriba (namespace-shim tras la celda de `data.py`).\n",
        ),
        (
            "from .evaluation import Predictor\n",
            "# Predictor ya es un nombre global de este notebook (celda de `evaluation.py`).\n",
        ),
    ],
    "hybrid_models": [
        (
            "from . import data as _data\n",
            "# `_data` ya quedó definido arriba (namespace-shim tras la celda de `data.py`).\n",
        ),
        (
            "from .evaluation import Fit, Predictor, ResultsTable, SEASONAL_PERIOD, split, train_and_val\n",
            "# Fit, Predictor, ResultsTable, SEASONAL_PERIOD, split y train_and_val ya son nombres\n"
            "# globales de este notebook (celda de `evaluation.py`, más arriba).\n",
        ),
    ],
    "selection": [
        (
            "from . import baselines as _baselines\n",
            "# `_baselines` ya quedó definido arriba (namespace-shim tras la celda de `baselines.py`).\n",
        ),
        (
            "from . import data as _data\n",
            "# `_data` ya quedó definido arriba (namespace-shim tras la celda de `data.py`).\n",
        ),
        (
            "from . import hybrid_models as _hybrid\n",
            "# `_hybrid` ya quedó definido arriba (namespace-shim tras la celda de `hybrid_models.py`).\n",
        ),
        (
            "from .evaluation import Fit, SEASONAL_PERIOD, split, train_and_val\n",
            "# Fit, SEASONAL_PERIOD, split y train_and_val ya son nombres globales de este notebook\n"
            "# (celda de `evaluation.py`, más arriba).\n",
        ),
    ],
}

# `Path(__file__)`-based directories rewritten to the notebook's own configuration variables.
PATH_OVERRIDES: dict[str, list[tuple[str, str]]] = {
    "data": [
        (
            'DATA_DIR = Path(__file__).resolve().parent.parent / "data"',
            "DATA_DIR = RUTA_DATOS"
            "  # override: en el paquete original era Path(__file__).resolve().parent.parent /"
            ' "data";\n'
            "# una celda de notebook no tiene __file__, así que se usa la ruta configurada arriba\n"
            "# en la celda de configuración (RUTA_DATOS).\n",
        ),
    ],
    "selection": [
        (
            'RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"',
            "RESULTS_DIR = RUTA_RESULTADOS"
            "  # override: idem DATA_DIR, ver la celda de configuración inicial (RUTA_RESULTADOS).\n",
        ),
    ],
}

# Submodules that other modules access through a qualified `from . import X as _x` alias need a
# namespace-shim cell right after their own code cell. Built automatically from each module's
# top-level names (see `top_level_names` below) so it never drifts from the real module contents.
SHIM_VAR = {"data": "_data", "baselines": "_baselines", "hybrid_models": "_hybrid"}


def top_level_names(source: str) -> list[str]:
    """Top-level function/class/constant names defined by a module's source (AST-based, no
    execution). Used to build each namespace-shim cell automatically from the real module
    contents, instead of a hand-maintained (and driftable) name list."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, ast.Tuple):
                    names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def transform_module(name: str, source: str) -> str:
    text = source
    for old, new in LITERAL_REPLACEMENTS.get(name, []):
        assert old in text, f"{name}.py: literal replacement anchor not found: {old!r}"
        text = text.replace(old, new)
    for pattern, repl in RENAMES.get(name, []):
        assert re.search(pattern, text), f"{name}.py: rename pattern not found: {pattern!r}"
        text = re.sub(pattern, repl, text)
    for old, new in IMPORT_STRIPS.get(name, []):
        assert old in text, f"{name}.py: import line to strip not found: {old!r}"
        text = text.replace(old, new)
    for old, new in PATH_OVERRIDES.get(name, []):
        assert old in text, f"{name}.py: path-override anchor not found: {old!r}"
        text = text.replace(old, new)
    return text


def build_namespace_shim_cell(module_name: str, transformed_source: str) -> str:
    var = SHIM_VAR[module_name]
    names = top_level_names(transformed_source)
    quoted = ",\n    ".join(f'"{n}"' for n in names)
    lines = [
        f"{var} = types.SimpleNamespace(**{{nombre: globals()[nombre] for nombre in [",
        f"    {quoted},",
        "]})",
        "# Namespace-shim: emula el objeto de módulo que el código original importaba como",
        f"# `from . import {module_name} as {var}`, para que las llamadas calificadas",
        f"# ({var}.algo(...)) de las celdas siguientes funcionen sin cambios en este notebook",
        "# de una sola celda por módulo (ver el docstring de este script para el porqué).",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# nbformat helpers
# ---------------------------------------------------------------------------


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n") + "\n")


def sub(template: str, **tokens: str) -> str:
    """Simple token substitution (no str.format): the generated code below is full of literal
    `{`/`}` (dict/set literals, f-strings), so `.format()` would need every one of them escaped.
    Tokens are written as __TOKEN__ placeholders instead and replaced with plain `.replace()`."""
    out = template
    for key, value in tokens.items():
        out = out.replace(f"__{key}__", value)
    return out


SERIES_NAMES = ["alb", "store_service"]
SERIES_LABELS = {
    "alb": "ALB — tráfico total (RequestCount)",
    "store_service": "store-service — tráfico por target (RequestCountPerTarget)",
}

# ---------------------------------------------------------------------------
# Section 0 — carátula, propósito, cómo ejecutar, paquetes, configuración
# ---------------------------------------------------------------------------


def build_header_cells() -> list[nbf.NotebookNode]:
    with PYPROJECT.open("rb") as fh:
        pyproject = tomllib.load(fh)
    deps = pyproject["project"]["dependencies"]
    py_req = pyproject["project"]["requires-python"]

    dep_rows = []
    for dep in deps:
        spec = dep.split(";")[0].strip()  # drop environment markers (e.g. "; sys_platform == ...")
        marker = dep[len(spec):].strip().lstrip(";").strip()
        if ">=" in spec:
            pkg_name, version = spec.split(">=", 1)
        else:
            pkg_name, version = spec, ""
        pkg_name = pkg_name.strip()
        version = version.strip()
        label = f"`{pkg_name}` (marcador: `{marker}`)" if marker else f"`{pkg_name}`"
        dep_rows.append(f"| {label} | >= {version} |")
    dep_table = "\n".join(["| Paquete | Versión mínima declarada |", "|---|---|", *dep_rows])

    cells = [
        md(
            "# Trabajo Práctico N° 2 — Análisis de Series Temporales\n\n"
            "**Maestría en Ciencia de Datos, Universidad Austral (sede Buenos Aires)**\n\n"
            "**Docente:** Rodrigo Del Rosso\n\n"
            "**Integrantes:** Carlos Aular · Juan Martín Pedrozo\n"
        ),
        md(
            "## Propósito de este notebook\n\n"
            "Este notebook es el acompañamiento en código del informe (`informe/informe.pdf`): "
            "reproduce, en un único archivo autocontenido, el análisis completo de las fases 01 a 06 "
            "del trabajo práctico para las series `alb` (tráfico total del *load balancer*) y "
            "`store_service` (tráfico por *target* del grupo `store-service`), más una tercera serie "
            "pública (`wikipedia_es`, visitas horarias de usuarios a Wikipedia en español) evaluada "
            "como contraste con seis modelos rápidos en la Sección 3. Las consignas del curso permiten subir como "
            "máximo dos archivos por integrante (el informe en PDF y un script o notebook), por lo "
            "que este archivo no puede depender del repositorio, del paquete `tsa_final/` ni de las "
            "carpetas `results/`/`data/` tal como existen en el repositorio de trabajo -- todo el "
            "código que ese paquete contenía está **inlineado** en la Sección 1, dentro del propio "
            "notebook, en el mismo orden de dependencias que tenía como paquete. No hay ninguna línea "
            "`import tsa_final` en todo el archivo (las únicas menciones a `tsa_final` son texto, "
            "en celdas Markdown o en comentarios, y no afectan la ejecución).\n\n"
            "## Cómo ejecutar\n\n"
            "1. Colocar este notebook junto a las carpetas `data/` y `results/` (o editar "
            "`RUTA_DATOS`/`RUTA_RESULTADOS` en la celda de configuración de abajo).\n"
            "2. `Kernel > Restart & Run All`. Con `REENTRENAR = False` (el valor por defecto) el "
            "notebook **no** vuelve a entrenar ningún modelo: reutiliza las tablas de resultados ya "
            "calculadas (`results/0X_*.csv`, `results/06_final_forecast.csv`) y las figuras del "
            "informe (`informe/figuras/`) para reconstruir tablas, rankings y gráficos. El tiempo de "
            "ejecución con `REENTRENAR = False` es de unos pocos minutos (ver "
            "`status/07-notebook.md` para el tiempo medido).\n"
            "3. Con `REENTRENAR = True`, cada fase vuelve a correr el entrenamiento real -- las "
            "mismas llamadas, semillas y presupuestos de tiempo que las fases 02 a 06 documentaron "
            "-- y **sobrescribe** `results/0X_*.csv` con resultados nuevos. Esto es costoso: solo la "
            "fase 04 (Deep Learning) ya toma ~28 minutos de ajuste (~31 minutos de notebook) y la "
            "fase 05 (Prophet/AutoML/híbridos) ~17 minutos de ajuste (~22 de notebook); sumando las "
            "fases 02, 03 y 06 el total estimado para todo el notebook con `REENTRENAR = True` es de "
            "aproximadamente una hora (ver `status/07-notebook.md`, sección Evidencia, para el "
            "desglose). No es necesario activarlo para revisar el análisis: todo el contenido "
            "reportable (tablas, rankings, figuras, selección, pronóstico final) se recalcula en el "
            "propio notebook a partir de esas tablas persistidas, con `REENTRENAR` en cualquiera de "
            "los dos valores.\n"
        ),
        md(
            "## Paquetes utilizados\n\n"
            f"Python `{py_req}` (entorno de desarrollo: Python 3.12). Versiones mínimas declaradas "
            f"en `pyproject.toml` del repositorio de trabajo:\n\n"
            f"{dep_table}\n\n"
            "La celda de código siguiente imprime las versiones efectivamente instaladas en el "
            "entorno donde se ejecuta este notebook, para verificar contra la tabla de arriba."
        ),
        code(
            "import importlib.metadata as _importlib_metadata\n"
            "import sys\n\n"
            "print(f\"Python {sys.version.split()[0]}\")\n"
            "for _pkg in [\n"
            '    "numpy", "pandas", "scipy", "statsmodels", "statsforecast", "scikit-learn",\n'
            '    "skforecast", "lightgbm", ("xgboost", "xgboost-cpu"), "catboost", "shap", "torch", "darts",\n'
            '    "pytorch-lightning", "prophet", "neuralprophet", "autogluon.timeseries", "autots",\n'
            '    "chronos-forecasting", "optuna", "matplotlib", "holidays", "nbformat", "nbconvert",\n'
            "]:\n"
            "    _candidatos = _pkg if isinstance(_pkg, tuple) else (_pkg,)\n"
            "    for _candidato in _candidatos:\n"
            "        try:\n"
            "            print(f\"{_candidato}: {_importlib_metadata.version(_candidato)}\")\n"
            "            break\n"
            "        except _importlib_metadata.PackageNotFoundError:\n"
            "            continue\n"
            "    else:\n"
            '        print(f"{_candidatos[0]}: no instalado en este entorno")\n'
        ),
        md(
            "## Configuración\n\n"
            "`REENTRENAR` controla si cada fase reentrena modelos reales (`True`) o reutiliza los "
            "resultados ya persistidos (`False`, por defecto). `RUTA_DATOS`/`RUTA_RESULTADOS` "
            "apuntan, por defecto, a `../data` y `../results` relativas a este notebook (la misma "
            "convención `Path.cwd().parent / \"...\"` que usaban los notebooks de cada fase, ya que "
            "este archivo también vive en una subcarpeta de `tp-final/`) -- se pueden editar para "
            "apuntar a otra ubicación si el notebook se mueve."
        ),
        code(
            "import types\n"
            "import warnings\n"
            "from pathlib import Path\n\n"
            "warnings.filterwarnings(\"ignore\")\n\n"
            "REENTRENAR = False  # True => reentrena todo con las llamadas/semillas/presupuestos originales\n\n"
            "RUTA_DATOS = Path(\"../data\")\n"
            "RUTA_RESULTADOS = Path(\"../results\")\n"
            "RUTA_FIGURAS_INFORME = Path(\"../informe/figuras\")  # solo lectura: nunca se escribe aquí\n\n"
            "RUTA_DATOS = RUTA_DATOS.resolve()\n"
            "RUTA_RESULTADOS = RUTA_RESULTADOS.resolve()\n"
            "RUTA_FIGURAS_INFORME = RUTA_FIGURAS_INFORME.resolve()\n\n"
            "SERIES_NAMES = [\"alb\", \"store_service\"]  # series de BodegaAI (fases 01 a 06); la serie pública se trata en la Sección 3\n"
            "SERIES_LABELS = {\n"
            "    \"alb\": \"ALB — tráfico total (RequestCount)\",\n"
            "    \"store_service\": \"store-service — tráfico por target (RequestCountPerTarget)\",\n"
            "}\n\n"
            'print(f"REENTRENAR = {REENTRENAR}")\n'
            'print(f"RUTA_DATOS = {RUTA_DATOS} (existe: {RUTA_DATOS.exists()})")\n'
            'print(f"RUTA_RESULTADOS = {RUTA_RESULTADOS} (existe: {RUTA_RESULTADOS.exists()})")\n'
            'print(f"RUTA_FIGURAS_INFORME = {RUTA_FIGURAS_INFORME} (existe: {RUTA_FIGURAS_INFORME.exists()})")\n'
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# Section 1 — módulos auxiliares inlineados
# ---------------------------------------------------------------------------


def build_module_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "# 1. Módulos auxiliares (código de `tsa_final/`, inlineado)\n\n"
            "Las siete celdas de código siguientes contienen, **sin modificar su lógica**, el "
            "contenido completo de los siete módulos que formaban el paquete `tsa_final/` en el "
            "repositorio de trabajo (`data.py`, `evaluation.py`, `baselines.py`, `ml_models.py`, "
            "`dl_models.py`, `hybrid_models.py`, `selection.py`), en el mismo orden de dependencias "
            "que tenían como paquete. Cada docstring de módulo se conserva íntegro.\n\n"
            "**Cómo se resolvieron las importaciones internas.** El código original importaba entre "
            "sí con sintaxis de paquete (`from . import data as _data`, "
            "`from .evaluation import Fit, Predictor, ...`), que no tiene sentido sin un paquete "
            "`tsa_final` real. Se optó por la alternativa más legible entre las dos consideradas: en "
            "vez de reescribir cada llamada calificada (`_data.calendar_features(...)` -> "
            "`calendar_features(...)`) en los siete archivos, se eliminan únicamente las líneas de "
            "importación (reemplazadas por un comentario) y se agrega, después de la celda de código "
            "de `data.py`, `baselines.py` y `hybrid_models.py`, una pequeña celda que construye un "
            "`types.SimpleNamespace` (`_data`, `_baselines`, `_hybrid`) con los mismos nombres que el "
            "módulo original exponía -- de modo que las llamadas calificadas de las celdas "
            "siguientes (`_data.load_clean(...)`, `_baselines.auto_arima_fit`, "
            "`_hybrid.chronos_zero_shot_fit`) funcionan exactamente igual que en el paquete, sin "
            "tocar el cuerpo de ninguna función. Los nombres importados sin calificar "
            "(`from .evaluation import Fit, split, ...`) no necesitan ningún shim: una vez que la "
            "celda de `evaluation.py` corrió, esos nombres ya son globales de este notebook.\n\n"
            "Dos colisiones de nombres que el paquete original evitaba automáticamente (cada módulo "
            "tenía su propio espacio de nombres) se vuelven reales al compartir un solo namespace, y "
            "se resuelven con renombres puntuales, documentados en el comentario de la celda donde "
            "ocurren: `MSTL` (`baselines.py` importa la clase de StatsForecast; `hybrid_models.py` "
            "importa, con el mismo nombre, la clase de descomposición de statsmodels -- se renombra "
            "a `_SF_MSTL` solo en `baselines.py`) y los helpers privados `_exog`/`_future_exog`/"
            "`_window_features`/`_EXOG_DTYPES` (definidos, casi idénticos, tanto en `ml_models.py` "
            "como en `hybrid_models.py` -- se renombran a `_ml_*`/`_hy_*` respectivamente). "
            "`DATA_DIR` y `RESULTS_DIR`, que en el paquete se derivaban de `Path(__file__)` (sin "
            "sentido en una celda de notebook), se redirigen a `RUTA_DATOS`/`RUTA_RESULTADOS` "
            "(sección de configuración, arriba)."
        ),
    ]

    transformed: dict[str, str] = {}
    for name in MODULE_ORDER:
        source = (TSA_FINAL / f"{name}.py").read_text()
        out = transform_module(name, source)
        transformed[name] = out
        cells.append(md(f"### {MODULE_TITLES[name]}\n\n{MODULE_SUMMARIES[name]}"))
        cells.append(code(out))
        if name in SHIM_VAR:
            cells.append(code(build_namespace_shim_cell(name, out)))
    return cells


# ---------------------------------------------------------------------------
# Section 2 — análisis por fase, con subsecciones por serie
# ---------------------------------------------------------------------------


def build_phase01_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "# 2. Análisis por fase\n\n"
            "## Fase 01 — Preparación de datos y análisis exploratorio (EDA)\n\n"
            "Documenta la preparación y el análisis exploratorio de `alb` (conteo total de "
            "solicitudes del *load balancer*) y `store_service` (conteo de solicitudes por *target* "
            "del grupo `store-service`, normalizado por la cantidad de instancias activas). No "
            "depende de `REENTRENAR`: no hay ningún modelo que entrenar, solo la carga/limpieza de "
            "datos (`load_raw`/`load_clean`, ya inlineadas en la Sección 1) y estadística descriptiva."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.ticker as mticker\n"
            "import seaborn as sns\n"
            "from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss\n"
            "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
            "import holidays as holidays_pkg\n\n"
            "sns.set_theme(style=\"whitegrid\")\n\n\n"
            "def miles(ax, eje=\"y\"):\n"
            "    formatter = mticker.FuncFormatter(lambda x, _: f\"{x:,.0f}\".replace(\",\", \".\"))\n"
            "    if eje == \"y\":\n"
            "        ax.yaxis.set_major_formatter(formatter)\n"
            "    else:\n"
            "        ax.xaxis.set_major_formatter(formatter)\n"
        ),
        md(
            "### 1.1 Serie limpia: intervención e imputación, perfil horario y descomposición MSTL\n\n"
            "Se corre, por serie, la misma secuencia que `notebooks/01_data_eda.ipynb`: zoom sobre la "
            "ventana de intervención (observado vs. imputado), perfil horario y heatmap día×hora en "
            "hora local, descomposición MSTL (periodos 24 y 168) con la fuerza de estacionalidad "
            "(Wang, Smith & Hyndman), ACF/PACF hasta el lag 336 y las pruebas de estacionariedad ADF/"
            "KPSS en nivel y primera diferencia."
        ),
        code(
            "def resumen_eda(nombre_serie: str) -> dict:\n"
            "    crudo = load_raw(nombre_serie)\n"
            "    limpia = load_clean(nombre_serie, impute=True)\n"
            "    etiqueta = SERIES_LABELS[nombre_serie]\n"
            "    print(f\"=== {etiqueta} ===\")\n"
            "    print(f\"Serie limpia: {len(limpia)} observaciones horarias, \"\n"
            "          f\"{limpia.index.min()} -> {limpia.index.max()}\")\n"
            "    print(f\"Horas de intervención imputadas: {int(intervention_mask(limpia.index).sum())}\")\n\n"
            "    zoom_ini, zoom_fin = pd.Timestamp(\"2026-09-10\", tz=\"UTC\"), pd.Timestamp(\"2026-09-27\", tz=\"UTC\")\n"
            "    crudo_zoom = crudo.loc[STEADY_STATE_START:].loc[zoom_ini:zoom_fin]\n"
            "    limpia_zoom = limpia.loc[zoom_ini:zoom_fin]\n"
            "    fig, ax = plt.subplots(figsize=(11, 3.2))\n"
            "    ax.plot(crudo_zoom.index, crudo_zoom.values, label=\"Observado (crudo)\", color=\"#1f77b4\", linewidth=1.0)\n"
            "    ax.plot(limpia_zoom.index, limpia_zoom.values, label=\"Imputado (intervención)\", color=\"darkorange\", linewidth=1.0, linestyle=\"--\")\n"
            "    ax.axvspan(INTERVENTION_START, INTERVENTION_END, color=\"grey\", alpha=0.25, label=\"Ventana de intervención\")\n"
            "    ax.set_title(f\"{etiqueta} — serie limpia vs. observada\")\n"
            "    ax.set_ylabel(\"Valor por hora\")\n"
            "    miles(ax)\n"
            "    ax.legend(loc=\"upper left\", fontsize=8)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "    cf = calendar_features(limpia.index)\n"
            "    hora_local = limpia.index.tz_convert(LOCAL_TZ)\n"
            "    perfil_horario = limpia.groupby(cf[\"hour\"]).mean()\n"
            "    medias_dow = limpia.groupby(cf[\"dayofweek\"]).mean()\n"
            "    medias_dow.index = [\"Lun\", \"Mar\", \"Mié\", \"Jue\", \"Vie\", \"Sáb\", \"Dom\"]\n"
            "    rango_relativo = (medias_dow.max() - medias_dow.min()) / medias_dow.mean() * 100\n"
            "    print(f\"Rango relativo entre días de la semana: {rango_relativo:.2f}% de la media\")\n\n"
            "    heat_df = pd.DataFrame({\"dow\": cf[\"dayofweek\"].values, \"hour\": cf[\"hour\"].values, \"value\": limpia.values})\n"
            "    heat_pivot = heat_df.pivot_table(index=\"dow\", columns=\"hour\", values=\"value\", aggfunc=\"mean\")\n"
            "    heat_pivot.index = [\"Lun\", \"Mar\", \"Mié\", \"Jue\", \"Vie\", \"Sáb\", \"Dom\"]\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
            "    axes[0].plot(perfil_horario.index, perfil_horario.values, marker=\"o\", color=\"#1f77b4\")\n"
            "    axes[0].set_title(\"Perfil horario medio (hora local)\")\n"
            "    axes[0].set_xlabel(\"Hora local\")\n"
            "    miles(axes[0])\n"
            "    sns.heatmap(heat_pivot, cmap=\"viridis\", ax=axes[1], cbar_kws={\"label\": \"Media\"})\n"
            "    axes[1].set_title(\"Media por día de semana x hora (hora local)\")\n"
            "    fig.suptitle(etiqueta, y=1.03)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "    mstl = MSTL(limpia, periods=(24, 168)).fit()\n"
            "    fig = mstl.plot()\n"
            "    fig.set_size_inches(11, 7)\n"
            "    fig.suptitle(f\"Descomposición MSTL — {etiqueta}\", y=1.01)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    resid_var = mstl.resid.var()\n"
            "    fuerza_24 = max(0.0, 1 - resid_var / (mstl.seasonal[\"seasonal_24\"] + mstl.resid).var())\n"
            "    fuerza_168 = max(0.0, 1 - resid_var / (mstl.seasonal[\"seasonal_168\"] + mstl.resid).var())\n"
            "    fuerza_tendencia = max(0.0, 1 - resid_var / (mstl.trend + mstl.resid).var())\n"
            "    print(f\"Fuerza estacional 24h = {fuerza_24:.3f} | 168h = {fuerza_168:.3f} | tendencia = {fuerza_tendencia:.3f}\")\n\n"
            "    log_diff = np.log1p(limpia).diff().dropna()\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))\n"
            "    plot_acf(log_diff, lags=336, ax=axes[0], title=f\"ACF log1p + diff(1) — {nombre_serie}\")\n"
            "    plot_pacf(log_diff, lags=336, ax=axes[1], method=\"ywm\", title=f\"PACF log1p + diff(1) — {nombre_serie}\")\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "    filas_estacionariedad = []\n"
            "    log_nivel = np.log1p(limpia)\n"
            "    for etiqueta_transf, clave, serie_ in [(\"nivel (log1p)\", \"level\", log_nivel), (\"primera diferencia (log1p)\", \"diff\", log_diff)]:\n"
            "        adf_stat, adf_p, *_ = adfuller(serie_, autolag=\"AIC\")\n"
            "        kpss_stat, kpss_p, *_ = kpss(serie_, regression=\"c\", nlags=\"auto\")\n"
            "        filas_estacionariedad.append({\"transformación\": etiqueta_transf, \"ADF p-valor\": adf_p, \"KPSS p-valor\": kpss_p})\n"
            "    tabla_estacionariedad = pd.DataFrame(filas_estacionariedad)\n"
            "    print(tabla_estacionariedad.round(3).to_string(index=False))\n\n"
            "    return {\n"
            "        \"limpia\": limpia,\n"
            "        \"fuerza_24\": fuerza_24,\n"
            "        \"fuerza_168\": fuerza_168,\n"
            "        \"estacionariedad\": tabla_estacionariedad,\n"
            "        \"rango_relativo_dow\": rango_relativo,\n"
            "    }\n"
        ),
        md("#### Serie ALB"),
        code("eda_alb = resumen_eda(\"alb\")"),
        md("#### Serie store_service"),
        code("eda_store = resumen_eda(\"store_service\")"),
        md(
            "### 1.2 Feriados argentinos y correlación cruzada ALB vs. store_service\n\n"
            "Se compara cada feriado argentino presente en el rango limpio contra el mismo día de la "
            "semana, una semana antes, y se calcula la correlación cruzada entre ambas series "
            "(diferenciadas en `log1p`) para lags de -24 a 24 h."
        ),
        code(
            "years = range(eda_alb[\"limpia\"].index.tz_convert(LOCAL_TZ).year.min(),\n"
            "              eda_alb[\"limpia\"].index.tz_convert(LOCAL_TZ).year.max() + 1)\n"
            "feriados_ar = holidays_pkg.Argentina(years=years)\n"
            "rango_local = eda_alb[\"limpia\"].index.tz_convert(LOCAL_TZ)\n"
            "feriados_en_rango = sorted({d for d in feriados_ar if rango_local.date.min() <= d <= rango_local.date.max()})\n"
            'print("Feriados argentinos en el rango de la serie limpia:", feriados_en_rango)\n\n'
            "filas_feriados = []\n"
            "for nombre_serie, resultado in [(\"alb\", eda_alb), (\"store_service\", eda_store)]:\n"
            "    idx_local = resultado[\"limpia\"].index.tz_convert(LOCAL_TZ)\n"
            "    for feriado in feriados_en_rango:\n"
            "        mascara_dia = idx_local.date == feriado\n"
            "        mascara_semana_previa = idx_local.date == (feriado - pd.Timedelta(days=7))\n"
            "        if mascara_dia.sum() == 0 or mascara_semana_previa.sum() == 0:\n"
            "            continue\n"
            "        media_feriado = resultado[\"limpia\"][mascara_dia].mean()\n"
            "        media_previa = resultado[\"limpia\"][mascara_semana_previa].mean()\n"
            "        filas_feriados.append({\"serie\": nombre_serie, \"feriado\": feriado, \"ratio_%\": 100 * media_feriado / media_previa})\n"
            "pd.DataFrame(filas_feriados)"
        ),
        code(
            "alb_diff = np.log1p(eda_alb[\"limpia\"]).diff().dropna()\n"
            "store_diff = np.log1p(eda_store[\"limpia\"]).diff().dropna()\n"
            "idx_comun = alb_diff.index.intersection(store_diff.index)\n"
            "alb_diff, store_diff = alb_diff.loc[idx_comun], store_diff.loc[idx_comun]\n\n"
            "lag_max = 24\n"
            "lags = range(-lag_max, lag_max + 1)\n"
            "valores_ccf = []\n"
            "for lag in lags:\n"
            "    if lag >= 0:\n"
            "        x, y = alb_diff.iloc[: len(alb_diff) - lag if lag else None], store_diff.shift(-lag).dropna()\n"
            "    else:\n"
            "        x, y = alb_diff.shift(lag).dropna(), store_diff\n"
            "    idx = x.index.intersection(y.index)\n"
            "    valores_ccf.append(np.corrcoef(x.loc[idx], y.loc[idx])[0, 1])\n"
            "serie_ccf = pd.Series(valores_ccf, index=list(lags))\n"
            "mejor_lag = serie_ccf.abs().idxmax()\n\n"
            "fig, ax = plt.subplots(figsize=(9, 3.5))\n"
            "ax.stem(serie_ccf.index, serie_ccf.values)\n"
            "ax.axvline(0, color=\"grey\", linewidth=0.8)\n"
            "ax.set_title(\"Correlación cruzada ALB vs. store_service (log1p + diff(1))\")\n"
            "ax.set_xlabel(\"Lag (horas); positivo = ALB adelanta a store_service\")\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            'print(f"Lag con correlación máxima en valor absoluto: {mejor_lag} h, corr = {serie_ccf[mejor_lag]:.3f}")\n'
        ),
        md(
            "### Conclusiones de la fase 01\n\n"
            "El ciclo diario (periodo 24) es el componente estacional dominante en ambas series "
            "(fuerza de estacionalidad de Wang-Smith-Hyndman muy por encima del componente semanal, "
            "periodo 168) y la estacionalidad semanal es prácticamente nula después del *go-live* del "
            "balanceador (medias por día de semana difieren menos del 3% de la media en ambas "
            "series). `log1p` + primera diferencia alcanza estacionariedad para las dos series según "
            "ADF y KPSS. La correlación cruzada ALB/store_service alcanza su máximo en el lag 0 "
            "($r \\approx 0.85$), es decir, las dos series se mueven de forma contemporánea y ALB no "
            "es un indicador adelantado de store_service a resolución horaria. Solo hay tres feriados "
            "argentinos en el rango limpio, con efecto mixto (no una reducción uniforme) -- "
            "insuficiente evidencia para una corrección específica; se deja como variable de "
            "calendario (`is_holiday`) para los modelos de las fases siguientes. Estas conclusiones "
            "son la base de las decisiones de protocolo de la fase 02: estacional-naive con "
            "`m=24` (no `m=168`) como referencia principal y como base de MASE."
        ),
    ]
    return cells


def build_phase02_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "## Fase 02 — Protocolo de evaluación y modelos base (*baselines*)\n\n"
            "Antes de modelos más sofisticados (fases 03 a 06), se corren métodos simples y clásicos "
            "como piso de comparación: si un modelo complejo no le gana a estos baselines, esa "
            "complejidad no se justifica. Protocolo (fijado en `evaluation.py`, Sección 1): un solo "
            "corte train/validación/test -- no *backtesting* expansivo, por presupuesto de tiempo "
            "(ver `status/02-evaluation-baselines.md`) -- con validación de 48 h terminando justo "
            "antes de la intervención conocida, y test de 105 h posteriores al *rollback*. Métricas: "
            "MAE, RMSE, MAPE y MASE (`m=24`, dado que la estacionalidad semanal es negligible, "
            "fase 01). Familias: `naive` (`naive`, `seasonal_naive_m24`, `drift`, `average`), el "
            "refit de las especificaciones SARIMA de TP1, y tres extras clásicos (Holt-Winters, "
            "MSTL+ARIMA, AutoARIMA)."
        ),
        code(
            "SERIES = {nombre: (eda_alb if nombre == \"alb\" else eda_store)[\"limpia\"] for nombre in SERIES_NAMES}\n\n"
            "fig, axes = plt.subplots(len(SERIES_NAMES), 1, figsize=(11, 3.2 * len(SERIES_NAMES)))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    s = SERIES[nombre_serie]\n"
            "    train, val, test = split(s)\n"
            "    ax.plot(s.index, s.values, linewidth=0.6, color=\"#444444\")\n"
            "    ax.axvspan(train.index.min(), train.index.max(), color=\"#1f77b4\", alpha=0.12, label=\"entrenamiento\")\n"
            "    ax.axvspan(val.index.min(), val.index.max(), color=\"#ff7f0e\", alpha=0.35, label=\"validación (48h)\")\n"
            "    ax.axvspan(test.index.min(), test.index.max(), color=\"#2ca02c\", alpha=0.35, label=\"test (105h)\")\n"
            "    ax.set_title(SERIES_LABELS[nombre_serie])\n"
            "    miles(ax)\n"
            "    ax.legend(loc=\"upper left\", fontsize=8, ncol=3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 2.1 Serie ALB y 2.2 Serie store_service (protocolo compartido)\n\n"
            "El bloque de entrenamiento de abajo corre, para las dos series a la vez (mismo patrón "
            "que `notebooks/02_evaluation_baselines.ipynb`), la familia *naive* completa, el refit "
            "SARIMA de TP1 y los tres extras clásicos -- solo si `REENTRENAR = True`. Con "
            "`REENTRENAR = False` se reutiliza `results/02_baselines.csv` tal cual quedó de la "
            "última corrida real."
        ),
        code(
            "resultados_02 = ResultsTable()\n"
            "if REENTRENAR:\n"
            "    for nombre_serie in SERIES_NAMES:\n"
            "        for nombre_modelo, fit_fn in NAIVE_MODELS.items():\n"
            "            evaluate(fit_fn, model=nombre_modelo, family=\"naive\", series_name=nombre_serie,\n"
            "                      series=SERIES[nombre_serie], table=resultados_02)\n"
            "        orden, orden_estacional = SARIMA_SPECS[nombre_serie]\n"
            "        evaluate(make_sarima_fit(orden, orden_estacional), model=f\"SARIMA{orden}x{orden_estacional}\",\n"
            "                  family=\"SARIMA (TP1 refit)\", series_name=nombre_serie, series=SERIES[nombre_serie],\n"
            "                  table=resultados_02)\n"
            "        for nombre_modelo in [\"holt_winters\", \"mstl_arima\", \"auto_arima\"]:\n"
            "            evaluate(CLASSICAL_MODELS[nombre_modelo], model=nombre_modelo, family=\"classical\",\n"
            "                      series_name=nombre_serie, series=SERIES[nombre_serie], table=resultados_02)\n"
            "    tabla_02 = resultados_02.to_frame()\n"
            "    tabla_02.to_csv(RUTA_RESULTADOS / \"02_baselines.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalcularon y sobrescribieron {len(tabla_02)} filas en 02_baselines.csv\")\n"
            "else:\n"
            "    print(\"REENTRENAR=False: se reutiliza results/02_baselines.csv, no se reentrena nada.\")\n\n"
            "tabla_02 = pd.read_csv(RUTA_RESULTADOS / \"02_baselines.csv\")\n"
            "tabla_val_02 = tabla_02[tabla_02[\"split\"] == \"val\"].sort_values([\"series\", \"MAE\"])\n"
            "tabla_test_02 = tabla_02[tabla_02[\"split\"] == \"test\"].sort_values([\"series\", \"MAE\"])\n"
            "tabla_val_02\n"
        ),
        md(
            "### 2.3 ¿Quién gana? Ranking de validación y control de realidad en test\n\n"
            "El gráfico de barras usa el MAPE de validación (la línea punteada marca "
            "`seasonal_naive_m24`, el piso a superar); el segundo gráfico reentrena el ganador de "
            "validación de cada serie sobre `train_and_val` y pronostica la ventana de test real "
            "(el refit más lento posible aquí es `auto_arima`, ~1 minuto; el resto es prácticamente "
            "instantáneo -- corre siempre, con cualquier valor de `REENTRENAR`, porque es la única "
            "forma de mostrar el pronóstico real contra el dato observado)."
        ),
        code(
            "fig, axes = plt.subplots(1, len(SERIES_NAMES), figsize=(12, 4.5))\n"
            "paleta_02 = {\"naive\": \"#999999\", \"SARIMA (TP1 refit)\": \"#1f77b4\", \"classical\": \"#2ca02c\"}\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    sub = tabla_val_02[tabla_val_02[\"series\"] == nombre_serie].sort_values(\"MAPE_%\")\n"
            "    colores = [paleta_02[f] for f in sub[\"family\"]]\n"
            "    ax.barh(sub[\"model\"], sub[\"MAPE_%\"], color=colores)\n"
            "    mape_naive = sub.loc[sub[\"model\"] == \"seasonal_naive_m24\", \"MAPE_%\"].iloc[0]\n"
            "    ax.axvline(mape_naive, color=\"crimson\", linestyle=\"--\")\n"
            "    ax.set_title(SERIES_LABELS[nombre_serie].split(\" — \")[0])\n"
            "    ax.set_xlabel(\"MAPE de validación (%)\")\n"
            "    ax.invert_yaxis()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        code(
            "TODOS_LOS_AJUSTES_02 = dict(NAIVE_MODELS)\n"
            "TODOS_LOS_AJUSTES_02.update(CLASSICAL_MODELS)\n\n\n"
            "def ajuste_para_02(nombre_modelo, nombre_serie):\n"
            "    if nombre_modelo in TODOS_LOS_AJUSTES_02:\n"
            "        return TODOS_LOS_AJUSTES_02[nombre_modelo]\n"
            "    orden, orden_estacional = SARIMA_SPECS[nombre_serie]\n"
            "    return make_sarima_fit(orden, orden_estacional)\n\n\n"
            "fig, axes = plt.subplots(len(SERIES_NAMES), 1, figsize=(11, 3.6 * len(SERIES_NAMES)))\n"
            "ganadores_02 = {}\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    ganador = tabla_val_02[tabla_val_02[\"series\"] == nombre_serie].iloc[0][\"model\"]\n"
            "    ganadores_02[nombre_serie] = ganador\n"
            "    s = SERIES[nombre_serie]\n"
            "    train, val, test = split(s)\n"
            "    ajuste_completo = train_and_val(s)\n"
            "    predictor = ajuste_para_02(ganador, nombre_serie)(ajuste_completo)\n"
            "    pred_test = predictor(len(test))\n"
            "    reciente = s.loc[ajuste_completo.index.max() - pd.Timedelta(hours=24 * 7):]\n"
            "    ax.plot(reciente.index, reciente.values, color=\"#444444\", linewidth=1, label=\"histórico reciente\")\n"
            "    ax.plot(test.index, test.values, color=\"#2ca02c\", linewidth=1.6, label=\"test real\")\n"
            "    ax.plot(test.index, pred_test, color=\"#d62728\", linewidth=1.6, linestyle=\"--\", label=\"pronóstico\")\n"
            "    ax.set_title(f\"{SERIES_LABELS[nombre_serie].split(' — ')[0]} — ganador en validación: {ganador}\")\n"
            "    miles(ax)\n"
            "    ax.legend(loc=\"upper left\", fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### Conclusiones de la fase 02\n\n"
            "En **ALB**, `auto_arima` (MASE val 0.475) y `seasonal_naive_m24` (MASE val 0.489) quedan "
            "prácticamente empatados en validación, y en test se invierte: el refit SARIMA de TP1 "
            "toma la delantera (MAPE 2.42%) aunque había quedado detrás en validación. En "
            "**store_service**, `seasonal_naive_m24` gana claramente en validación (MASE 0.480) y "
            "sigue siendo fuerte en test. `seasonal_naive_m24` es una referencia genuinamente sólida "
            "en las dos series (MASE < 1 en ambos splits): cualquier modelo de las fases 03-06 que no "
            "le gane no está justificando su complejidad adicional. La divergencia entre el ranking "
            "de validación y el de test (especialmente en ALB) es la primera evidencia reportable del "
            "riesgo aceptado al usar un único corte de validación en vez de *backtesting* expansivo -- "
            "se documenta, no se oculta, y vuelve a aparecer en cada fase siguiente."
        ),
    ]
    return cells


def build_phase03_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "## Fase 03 — Modelos de Machine Learning\n\n"
            "*Forecasters* de árboles y lineales (LightGBM, XGBoost, CatBoost, RandomForest, Ridge) "
            "sobre lags (1-24, 48, 168, 336) y variables de calendario, con `skforecast` "
            "`ForecasterRecursive`. Secuencia fija (ver `status/03-ml-models.md`, \"Sequencing\"): "
            "recursivo (5 modelos, hiperparámetros por defecto) -> ranking de validación -> "
            "estrategia *direct* solo para el #1 -> Optuna solo para el top-3 por MAE default -> "
            "*stacking* del top-3 tuneado -> ganador único por serie -> SHAP solo del ganador. Vara "
            "a vencer: `seasonal_naive_m24` (MASE < 1 en validación, fase 02)."
        ),
        code(
            "import shap\n"
            "from sklearn.linear_model import Ridge as _RidgeParaSHAP\n\n\n"
            "def ejecutar_pipeline_ml(nombre_serie: str) -> dict:\n"
            "    print(f\"=== {nombre_serie} ===\")\n"
            "    series = SERIES[nombre_serie]\n"
            "    tabla = ResultsTable()\n\n"
            "    mase_naive_val = tabla_02.loc[\n"
            "        (tabla_02[\"series\"] == nombre_serie) & (tabla_02[\"model\"] == \"seasonal_naive_m24\") & (tabla_02[\"split\"] == \"val\"),\n"
            "        \"MASE\",\n"
            "    ].iloc[0]\n\n"
            "    for nombre, ajuste in RECURSIVE_MODELS.items():\n"
            "        evaluate(ajuste, model=nombre, family=\"ML\", series_name=nombre_serie, series=series, table=tabla)\n\n"
            "    df = tabla.to_frame()\n"
            "    ranking_val = df[df[\"split\"] == \"val\"][[\"model\", \"MAE\", \"MASE\"]].sort_values(\"MAE\").reset_index(drop=True)\n"
            "    mejor_recursivo = ranking_val.iloc[0][\"model\"]\n"
            "    candidatos_tuneo = ranking_val.head(3)[\"model\"].tolist()\n\n"
            "    evaluate_direct(default_builder(mejor_recursivo), model=f\"{mejor_recursivo}_direct\",\n"
            "                     series_name=nombre_serie, series=series, table=tabla)\n\n"
            "    parametros_tuneados: dict[str, dict] = {}\n"
            "    nombres_tuneados = []\n"
            "    for nombre in candidatos_tuneo:\n"
            "        mejores_parametros = tune_hyperparameters(nombre, series, n_trials=20, timeout_s=120)\n"
            "        parametros_tuneados[nombre] = mejores_parametros\n"
            "        evaluate(make_tuned_fit(nombre, mejores_parametros), model=f\"{nombre}_tuned\", family=\"ML\",\n"
            "                  series_name=nombre_serie, series=series, table=tabla)\n"
            "        nombres_tuneados.append(f\"{nombre}_tuned\")\n\n"
            "    df = tabla.to_frame()\n"
            "    val_tuneado = df[(df[\"split\"] == \"val\") & (df[\"model\"].isin(nombres_tuneados))][[\"model\", \"MAE\"]].sort_values(\"MAE\")\n"
            "    top3 = [m.replace(\"_tuned\", \"\") for m in val_tuneado.head(3)[\"model\"].tolist()]\n"
            "    if len(top3) >= 2:\n"
            "        estimadores = []\n"
            "        for nombre in top3:\n"
            "            clase_est, kwargs_fijos, _ = MODEL_SPECS[nombre]\n"
            "            estimadores.append((nombre, clase_est(**kwargs_fijos, **parametros_tuneados[nombre])))\n"
            "        evaluate(make_stacking_fit(estimadores, cv=3), model=\"stacking_top3\", family=\"ML\",\n"
            "                  series_name=nombre_serie, series=series, table=tabla)\n\n"
            "    df = tabla.to_frame()\n"
            "    ranking_general = df[df[\"split\"] == \"val\"][[\"model\", \"MAE\", \"MASE\"]].sort_values(\"MAE\").reset_index(drop=True)\n"
            "    ganador = ranking_general.iloc[0][\"model\"]\n"
            "    print(f\"ganador ({nombre_serie}): {ganador} | naive a vencer: {mase_naive_val:.3f}\")\n"
            "    return {\"table\": df, \"winner\": ganador, \"tuned_params\": parametros_tuneados, \"series\": series}\n"
        ),
        md(
            "### 3.1 Serie ALB y 3.2 Serie store_service\n\n"
            "Con `REENTRENAR = False` se reutiliza `results/03_ml.csv`; los hiperparámetros tuneados "
            "por Optuna solo existen en memoria si `REENTRENAR = True` (no se persisten en el CSV), "
            "así que la figura SHAP de la siguiente subsección se recalcula únicamente en ese caso -- "
            "de lo contrario se muestra la figura ya generada por `notebooks/03_ml_models.ipynb`."
        ),
        code(
            "resultados_03 = {}\n"
            "if REENTRENAR:\n"
            "    for nombre_serie in SERIES_NAMES:\n"
            "        resultados_03[nombre_serie] = ejecutar_pipeline_ml(nombre_serie)\n"
            "    tabla_03 = pd.concat([resultados_03[s][\"table\"] for s in SERIES_NAMES], ignore_index=True)\n"
            "    tabla_03.to_csv(RUTA_RESULTADOS / \"03_ml.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalcularon y sobrescribieron {len(tabla_03)} filas en 03_ml.csv\")\n"
            "else:\n"
            "    print(\"REENTRENAR=False: se reutiliza results/03_ml.csv, no se reentrena nada.\")\n\n"
            "tabla_03 = pd.read_csv(RUTA_RESULTADOS / \"03_ml.csv\")\n"
            "tabla_val_03 = tabla_03[tabla_03[\"split\"] == \"val\"].sort_values([\"series\", \"MAE\"])\n"
            "ganadores_03 = {\n"
            "    nombre_serie: tabla_val_03[tabla_val_03[\"series\"] == nombre_serie].iloc[0][\"model\"]\n"
            "    for nombre_serie in SERIES_NAMES\n"
            "}\n"
            "print(\"Ganador por serie (menor MAE de validación):\", ganadores_03)\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    ranking = tabla_val_03[tabla_val_03[\"series\"] == nombre_serie].sort_values(\"MASE\", ascending=False)\n"
            "    mase_naive = tabla_02.loc[(tabla_02[\"series\"] == nombre_serie) & (tabla_02[\"model\"] == \"seasonal_naive_m24\") & (tabla_02[\"split\"] == \"val\"), \"MASE\"].iloc[0]\n"
            "    colores = [\"#2ca02c\" if m == ganadores_03[nombre_serie] else \"#1f77b4\" for m in ranking[\"model\"]]\n"
            "    ax.barh(ranking[\"model\"], ranking[\"MASE\"], color=colores)\n"
            "    ax.axvline(mase_naive, color=\"red\", linestyle=\"--\", label=f\"seasonal_naive_m24 ({mase_naive:.3f})\")\n"
            "    ax.set_xlabel(\"MASE de validación\")\n"
            "    ax.set_title(nombre_serie)\n"
            "    ax.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 3.3 SHAP del ganador único por serie (solo el ganador, por costo)"
        ),
        code(
            "from IPython.display import Image, display\n\n"
            "if REENTRENAR:\n"
            "    for nombre_serie in SERIES_NAMES:\n"
            "        ganador = resultados_03[nombre_serie][\"winner\"]\n"
            "        params_tuneados = resultados_03[nombre_serie][\"tuned_params\"]\n"
            "        series = resultados_03[nombre_serie][\"series\"]\n"
            "        nombre_base = ganador.replace(\"_tuned\", \"\").replace(\"_direct\", \"\")\n"
            "        if nombre_base not in MODEL_SPECS:\n"
            "            print(f\"SHAP omitido para {nombre_serie}: el ganador ({ganador}) es un stacking, sin un único estimador base\")\n"
            "            continue\n"
            "        train, val, test = split(series)\n"
            "        entrenamiento_completo = pd.concat([train, val])\n"
            "        if nombre_base in params_tuneados:\n"
            "            clase_est, kwargs_fijos, _ = MODEL_SPECS[nombre_base]\n"
            "            estimador = clase_est(**kwargs_fijos, **params_tuneados[nombre_base])\n"
            "        else:\n"
            "            estimador = default_builder(nombre_base)()\n"
            "        X_shap, y_shap = training_matrix(nombre_base, estimador, entrenamiento_completo)\n"
            "        estimador.fit(X_shap, y_shap)\n"
            "        explainer = shap.LinearExplainer(estimador, X_shap) if isinstance(estimador, _RidgeParaSHAP) else shap.TreeExplainer(estimador)\n"
            "        valores_shap = explainer.shap_values(X_shap)\n"
            "        plt.figure()\n"
            "        shap.summary_plot(valores_shap, X_shap, show=False, plot_size=(9, 6))\n"
            "        plt.title(f\"SHAP -- {ganador} ({nombre_serie})\")\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
            "else:\n"
            "    print(\"REENTRENAR=False: se muestran las figuras SHAP ya generadas por notebooks/03_ml_models.ipynb\")\n"
            "    for nombre_serie in SERIES_NAMES:\n"
            "        display(Image(filename=str(RUTA_FIGURAS_INFORME / f\"03_shap_{nombre_serie}.png\")))\n"
        ),
        md(
            "### Conclusiones de la fase 03\n\n"
            "En **ALB**, `catboost_tuned` gana en validación (MASE 0.353, le gana a la naive 0.489) "
            "pero pierde en test (MASE 2.313) -- la misma divergencia validación/test que ya "
            "documentó la fase 02, no un error. En **store_service**, `lightgbm_tuned` gana en "
            "validación (MASE 0.573) pero **no** logra bajar de la naive (0.480) en ningún split -- "
            "ningún modelo ML le ganó a la estacional-naive en esta serie. El *stacking* "
            "(`stacking_top3`) fue el peor modelo de toda la fase en ambas series: combinar tres "
            "modelos tuneados vía un meta-modelo lineal no ayudó en un pronóstico recursivo de 48 "
            "pasos. SHAP confirma que `lag_1`, `lag_2`, `lag_23/24` y `lag_48` dominan la importancia "
            "de *features*, consistente con la estacionalidad diaria ya cuantificada en la fase 01."
        ),
    ]
    return cells


def build_phase04_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "## Fase 04 — Modelos de Deep Learning\n\n"
            "Seis arquitecturas de redes (LSTM, N-BEATS, N-HiTS, TCN, TiDE, TFT, con `darts`), "
            "escalado `log1p` + `Scaler`, covariables de calendario cíclicas y 2 semillas fijas "
            "(42, 43). **Regla de fuga de datos**: el *early stopping* nunca mira la ventana de "
            "selección de 48 h -- monitorea `val_loss` en las 168 h inmediatamente anteriores, "
            "registra la mejor época y el refit de test entrena esa cantidad fija de épocas, sin "
            "*early stopping* (ver docstring de `dl_models.py`, Sección 1, y "
            "`status/04-dl-models.md`). Con `REENTRENAR = True` esto es la fase más costosa del "
            "notebook: 6 arquitecturas x 2 semillas x 2 series x 2 ajustes (validación + test) "
            "~28 minutos de ajuste medidos (~31 min de notebook, ver Evidencia en "
            "`status/04-dl-models.md`)."
        ),
        code(
            "def ejecutar_pipeline_dl(nombre_serie: str) -> dict:\n"
            "    print(f\"=== {nombre_serie} ===\")\n"
            "    series = SERIES[nombre_serie]\n"
            "    tabla_semillas = ResultsTable()\n"
            "    for nombre_arquitectura in ARCHITECTURES:\n"
            "        for semilla in SEEDS:\n"
            "            ajuste = make_fit(nombre_arquitectura, semilla)\n"
            "            evaluate(ajuste, model=f\"{nombre_arquitectura}_seed{semilla}\", family=\"DL\",\n"
            "                      series_name=nombre_serie, series=series, table=tabla_semillas)\n"
            "    df_semillas = tabla_semillas.to_frame()\n"
            "    df_semillas[\"architecture\"] = df_semillas[\"model\"].str.extract(r\"^(.*)_seed\\d+$\")\n"
            "    df_semillas[\"seed\"] = df_semillas[\"model\"].str.extract(r\"_seed(\\d+)$\").astype(int)\n"
            "    filas_principales = []\n"
            "    for nombre_arquitectura in ARCHITECTURES:\n"
            "        for split_ in [\"val\", \"test\"]:\n"
            "            sub = df_semillas[(df_semillas[\"architecture\"] == nombre_arquitectura) & (df_semillas[\"split\"] == split_)]\n"
            "            filas_principales.append({\n"
            "                \"model\": nombre_arquitectura, \"family\": \"DL\", \"series\": nombre_serie, \"split\": split_,\n"
            "                \"MAE\": sub[\"MAE\"].mean(), \"RMSE\": sub[\"RMSE\"].mean(), \"MAPE_%\": sub[\"MAPE_%\"].mean(),\n"
            "                \"MASE\": sub[\"MASE\"].mean(), \"fit_time_s\": sub[\"fit_time_s\"].sum(),\n"
            "            })\n"
            "    return {\"series\": series, \"seeds_df\": df_semillas, \"main_df\": pd.DataFrame(filas_principales)}\n"
        ),
        md(
            "### 4.1 Serie ALB y 4.2 Serie store_service\n\n"
            "Con `REENTRENAR = False` se reutilizan `results/04_dl.csv` (agregado por semilla) y "
            "`results/04_dl_seeds.csv` (por semilla) tal como quedaron de la última corrida real."
        ),
        code(
            "if REENTRENAR:\n"
            "    resultados_04 = {nombre_serie: ejecutar_pipeline_dl(nombre_serie) for nombre_serie in SERIES_NAMES}\n"
            "    seeds_04 = pd.concat([resultados_04[s][\"seeds_df\"] for s in SERIES_NAMES], ignore_index=True)\n"
            "    seeds_04.to_csv(RUTA_RESULTADOS / \"04_dl_seeds.csv\", index=False)\n"
            "    tabla_04 = pd.concat([resultados_04[s][\"main_df\"] for s in SERIES_NAMES], ignore_index=True)\n"
            "    tabla_04.to_csv(RUTA_RESULTADOS / \"04_dl.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalcularon y sobrescribieron 04_dl.csv ({len(tabla_04)} filas) y 04_dl_seeds.csv ({len(seeds_04)} filas)\")\n"
            "else:\n"
            "    print(\"REENTRENAR=False: se reutilizan results/04_dl.csv y results/04_dl_seeds.csv, no se reentrena nada.\")\n\n"
            "tabla_04 = pd.read_csv(RUTA_RESULTADOS / \"04_dl.csv\")\n"
            "seeds_04 = pd.read_csv(RUTA_RESULTADOS / \"04_dl_seeds.csv\")\n"
            "tabla_val_04 = tabla_04[tabla_04[\"split\"] == \"val\"].sort_values([\"series\", \"MASE\"])\n"
            "ganadores_04 = {\n"
            "    nombre_serie: tabla_val_04[tabla_val_04[\"series\"] == nombre_serie].iloc[0][\"model\"]\n"
            "    for nombre_serie in SERIES_NAMES\n"
            "}\n"
            "print(\"Mejor arquitectura DL por serie (menor MASE de validación):\", ganadores_04)\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    ranking = tabla_val_04[tabla_val_04[\"series\"] == nombre_serie][[\"model\", \"MASE\"]].sort_values(\"MASE\", ascending=False)\n"
            "    mase_naive = tabla_02.loc[(tabla_02[\"series\"] == nombre_serie) & (tabla_02[\"model\"] == \"seasonal_naive_m24\") & (tabla_02[\"split\"] == \"val\"), \"MASE\"].iloc[0]\n"
            "    colores = [\"#2ca02c\" if m == ganadores_04[nombre_serie] else \"#1f77b4\" for m in ranking[\"model\"]]\n"
            "    ax.barh(ranking[\"model\"], ranking[\"MASE\"], color=colores)\n"
            "    ax.axvline(mase_naive, color=\"red\", linestyle=\"--\", label=f\"seasonal_naive_m24 ({mase_naive:.3f})\")\n"
            "    ax.set_xlabel(\"MASE de validación\")\n"
            "    ax.set_title(nombre_serie)\n"
            "    ax.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 4.3 Dispersión entre semillas, mejor época y pronóstico en test\n\n"
            "La dispersión de MAE de validación entre las 2 semillas y la mejor época de "
            "*early stopping* se leen directamente de `04_dl_seeds.csv`/`04_dl_best_epochs.csv` (no "
            "requieren reentrenar). El pronóstico de test sí necesita un ajuste real: se reentrena "
            "únicamente la arquitectura ganadora de cada serie (no las seis), un refit de costo "
            "bajo -- entre 9 y 30 s por ajuste para cinco de las seis arquitecturas, ver "
            "`status/04-dl-models.md` -- así que se corre siempre, con cualquier valor de "
            "`REENTRENAR`."
        ),
        code(
            "dispersión = seeds_04[seeds_04[\"split\"] == \"val\"].groupby([\"series\", \"architecture\"])[\"MAE\"].agg([\"mean\", \"std\"])\n"
            "dispersión[\"cv_%\"] = 100 * dispersión[\"std\"] / dispersión[\"mean\"]\n"
            "print(dispersión.round(3).to_string())\n\n"
            "epocas_04 = pd.read_csv(RUTA_RESULTADOS / \"04_dl_best_epochs.csv\")\n"
            "print(epocas_04.pivot_table(index=\"architecture\", columns=[\"series\", \"seed\"], values=\"best_epoch\").to_string())\n"
        ),
        code(
            "fig, axes = plt.subplots(2, 1, figsize=(12, 9))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    ganador = ganadores_04[nombre_serie]\n"
            "    semilla = int(seeds_04[(seeds_04[\"series\"] == nombre_serie) & (seeds_04[\"architecture\"] == ganador)][\"seed\"].iloc[0])\n"
            "    series = SERIES[nombre_serie]\n"
            "    train, val, test = split(series)\n"
            "    ajuste = make_fit(ganador, semilla)\n"
            "    ajuste(train)  # ajuste de validación, descartado -- solo sirve para fijar best_epoch\n"
            "    pronostico_test = ajuste(train_and_val(series))(len(test))\n"
            "    ax.plot(test.index, test.to_numpy(), label=\"real\", color=\"black\")\n"
            "    ax.plot(test.index, pronostico_test, label=f\"pronóstico ({ganador})\", color=\"#d62728\")\n"
            "    ax.set_title(f\"{nombre_serie} -- test, mejor DL: {ganador} (semilla {semilla})\")\n"
            "    ax.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### Conclusiones de la fase 04\n\n"
            "**Ninguna arquitectura DL le gana a `seasonal_naive_m24` ni al mejor modelo de las fases "
            "02-03 en validación**, en ninguna de las dos series: ALB, mejor DL `nhits` (MASE 0.556) "
            "vs. naive 0.489 vs. `catboost_tuned` 0.353; store_service, mejor DL `nbeats` (MASE "
            "0.638) vs. naive 0.480. **En test la situación se invierte**: `nbeats` es el mejor "
            "modelo DL en ambas series y sí le gana tanto a la naive como al mejor modelo pre-DL "
            "(ALB: test MASE 1.744 vs. naive 1.990; store_service: 1.247 vs. 1.443) -- la misma "
            "asimetría validación/test que documentaron las fases 02 y 03. La dispersión entre "
            "semillas es alta para `lstm` y `nbeats` (CV hasta 57% en ALB) y TFT domina el costo de "
            "entrenamiento (10-30x el resto) sin ganar en ninguna serie ni split: la arquitectura más "
            "cara no es la más precisa. Los números de esta subsección ya incorporan la corrección "
            "del 2026-09-28 (restauración de los pesos de la mejor época tras *early stopping*, ver "
            "`status/04-dl-models.md`, sección Evidencia -> Corrección): antes de esa corrección, el "
            "mejor DL en validación era `nhits` con MASE 0.875 en ALB y `tcn` con 0.760 en "
            "store_service -- las cifras vigentes (0.556 y 0.638) son las que corresponden a este "
            "notebook."
        ),
    ]
    return cells


def build_phase05_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "## Fase 05 — Prophet, híbridos, AutoML y modelo de fundación\n\n"
            "Familias que faltaban en la comparación: Prophet/NeuralProphet (familia `Prophet`), un "
            "híbrido MSTL + LightGBM (`Hybrid`), un ensamble de peso igual entre el mejor modelo de "
            "cada fase anterior (`Ensemble`), AutoGluon TimeSeries y AutoTS (`AutoML`) y Chronos-Bolt "
            "zero-shot (`Foundation`). TimeGPT (Nixtla) se omite por completo: requiere una API "
            "externa con clave, sin aprobación explícita para este proyecto. Regla de fuga heredada "
            "de la fase 04: ningún tuneo/*early stopping*/peso de ensamble puede mirar la ventana de "
            "selección de 48 h; el tuneo de Prophet usa las 168 h inmediatamente anteriores. Con "
            "`REENTRENAR = True` es, junto con la fase 04, la fase más costosa: ~17 minutos de ajuste "
            "medidos (~22 min de notebook, `status/05-hybrid-automl-foundation.md`), dominados por "
            "AutoGluon (300 s de límite de tiempo x 2 llamados x 2 series)."
        ),
        code(
            "DEFAULT_ML_NAMES = list(RECURSIVE_MODELS)  # solo boosters/lineales sin tunear\n"
            "FAMILIAS_BASELINE = [\"naive\", \"classical\", \"SARIMA (TP1 refit)\"]\n\n\n"
            "def ajuste_baseline_para(nombre_serie, nombre_modelo):\n"
            "    if nombre_modelo in NAIVE_MODELS:\n"
            "        return NAIVE_MODELS[nombre_modelo]\n"
            "    if nombre_modelo in CLASSICAL_MODELS:\n"
            "        return CLASSICAL_MODELS[nombre_modelo]\n"
            "    orden, orden_estacional = SARIMA_SPECS[nombre_serie]\n"
            "    return make_sarima_fit(orden, orden_estacional)\n\n\n"
            "def seleccionar_miembros_ensamble(nombre_serie):\n"
            "    pool_baseline = tabla_02[(tabla_02[\"series\"] == nombre_serie) & (tabla_02[\"family\"].isin(FAMILIAS_BASELINE))]\n"
            "    fila_baseline = pool_baseline[pool_baseline[\"split\"] == \"val\"].sort_values(\"MAE\").iloc[0]\n"
            "    pool_ml = tabla_03[(tabla_03[\"series\"] == nombre_serie) & (tabla_03[\"family\"] == \"ML\") & (tabla_03[\"model\"].isin(DEFAULT_ML_NAMES))]\n"
            "    fila_ml = pool_ml[pool_ml[\"split\"] == \"val\"].sort_values(\"MAE\").iloc[0]\n"
            "    pool_dl = tabla_04[(tabla_04[\"series\"] == nombre_serie) & (tabla_04[\"split\"] == \"val\")].sort_values(\"MAE\").reset_index(drop=True)\n"
            "    fila_dl = pool_dl.iloc[0]\n"
            "    if fila_dl[\"model\"] == \"tft\":\n"
            "        fila_dl = pool_dl.iloc[1]  # tft es la arquitectura mas cara; se sustituye si ganara (no ocurrió en la práctica)\n"
            "    return {\n"
            "        \"baseline\": (fila_baseline[\"model\"], ajuste_baseline_para(nombre_serie, fila_baseline[\"model\"])),\n"
            "        \"ml\": (fila_ml[\"model\"], RECURSIVE_MODELS[fila_ml[\"model\"]]),\n"
            "        \"dl\": (fila_dl[\"model\"], make_fit(fila_dl[\"model\"], seed=SEEDS[0])),\n"
            "    }\n"
        ),
        code(
            "def ejecutar_pipeline_05(nombre_serie: str) -> dict:\n"
            "    print(f\"=== {nombre_serie} ===\")\n"
            "    series = SERIES[nombre_serie]\n"
            "    tabla = ResultsTable()\n\n"
            "    evaluate(prophet_default_fit, model=\"prophet_default\", family=\"Prophet\", series_name=nombre_serie, series=series, table=tabla)\n"
            "    prophet_tuneado = make_prophet_tuned_fit()\n"
            "    evaluate(prophet_tuneado, model=\"prophet_tuned\", family=\"Prophet\", series_name=nombre_serie, series=series, table=tabla)\n"
            "    evaluate(neuralprophet_fit, model=\"neuralprophet\", family=\"Prophet\", series_name=nombre_serie, series=series, table=tabla)\n"
            "    evaluate(hybrid_mstl_lightgbm_fit, model=\"mstl_lightgbm\", family=\"Hybrid\", series_name=nombre_serie, series=series, table=tabla)\n"
            "    modelos_autogluon = evaluate_autogluon(series, series_name=nombre_serie, table=tabla)\n"
            "    evaluate(autots_fit, model=\"autots\", family=\"AutoML\", series_name=nombre_serie, series=series, table=tabla)\n"
            "    evaluate(chronos_zero_shot_fit, model=\"chronos_bolt_base\", family=\"Foundation\", series_name=nombre_serie, series=series, table=tabla)\n\n"
            "    miembros = seleccionar_miembros_ensamble(nombre_serie)\n"
            "    ajuste_ensamble = make_ensemble_fit([ajuste for _nombre, ajuste in miembros.values()])\n"
            "    evaluate(ajuste_ensamble, model=\"ensemble_equal_weight\", family=\"Ensemble\", series_name=nombre_serie, series=series, table=tabla)\n\n"
            "    df = tabla.to_frame()\n"
            "    ganador = df[df[\"split\"] == \"val\"].sort_values(\"MAE\").iloc[0][\"model\"]\n"
            "    print(f\"ganador ({nombre_serie}): {ganador} | miembros de ensamble: {[n for n, _f in miembros.values()]} | AutoGluon: {modelos_autogluon}\")\n"
            "    return {\"table\": df, \"series\": series, \"winner\": ganador, \"ensemble_members\": miembros}\n"
        ),
        md(
            "### 5.1 Serie ALB y 5.2 Serie store_service\n\n"
            "Con `REENTRENAR = False` se reutiliza `results/05_hybrid.csv`."
        ),
        code(
            "if REENTRENAR:\n"
            "    resultados_05 = {nombre_serie: ejecutar_pipeline_05(nombre_serie) for nombre_serie in SERIES_NAMES}\n"
            "    tabla_05 = pd.concat([resultados_05[s][\"table\"] for s in SERIES_NAMES], ignore_index=True)\n"
            "    tabla_05.to_csv(RUTA_RESULTADOS / \"05_hybrid.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalcularon y sobrescribieron {len(tabla_05)} filas en 05_hybrid.csv\")\n"
            "else:\n"
            "    print(\"REENTRENAR=False: se reutiliza results/05_hybrid.csv, no se reentrena nada.\")\n"
            "    resultados_05 = None\n\n"
            "tabla_05 = pd.read_csv(RUTA_RESULTADOS / \"05_hybrid.csv\")\n"
            "tabla_val_05 = tabla_05[tabla_05[\"split\"] == \"val\"].sort_values([\"series\", \"MASE\"])\n"
            "ganadores_05 = {\n"
            "    nombre_serie: tabla_val_05[tabla_val_05[\"series\"] == nombre_serie].iloc[0][\"model\"]\n"
            "    for nombre_serie in SERIES_NAMES\n"
            "}\n"
            "print(\"Mejor modelo de fase 05 por serie:\", ganadores_05)\n"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    ranking = tabla_val_05[tabla_val_05[\"series\"] == nombre_serie][[\"model\", \"MASE\"]].sort_values(\"MASE\", ascending=False)\n"
            "    mase_naive = tabla_02.loc[(tabla_02[\"series\"] == nombre_serie) & (tabla_02[\"model\"] == \"seasonal_naive_m24\") & (tabla_02[\"split\"] == \"val\"), \"MASE\"].iloc[0]\n"
            "    colores = [\"#2ca02c\" if m == ganadores_05[nombre_serie] else \"#1f77b4\" for m in ranking[\"model\"]]\n"
            "    ax.barh(ranking[\"model\"], ranking[\"MASE\"], color=colores)\n"
            "    ax.axvline(mase_naive, color=\"red\", linestyle=\"--\", label=f\"seasonal_naive_m24 ({mase_naive:.3f})\")\n"
            "    ax.set_xlabel(\"MASE de validación\")\n"
            "    ax.set_title(nombre_serie)\n"
            "    ax.legend(fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 5.3 Pronóstico en test del mejor modelo de fase 05\n\n"
            "El ganador de las dos series es `ensemble_equal_weight`, cuyo pronóstico de test exige "
            "reentrenar sus tres miembros (uno de ellos, el `auto_arima` o el SARIMA de TP1 según la "
            "serie, cuesta ~1 minuto). Con `REENTRENAR = True` se reentrena en el momento, "
            "reutilizando los cierres (`Fit`) ya construidos en el bloque anterior. Con "
            "`REENTRENAR = False` se muestra la figura ya generada por "
            "`notebooks/05_hybrid_automl_foundation.ipynb` en vez de repetir ese costo solo para una "
            "figura."
        ),
        code(
            "if REENTRENAR:\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(12, 9))\n"
            "    for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "        series = SERIES[nombre_serie]\n"
            "        train, val, test = split(series)\n"
            "        ajuste_completo = train_and_val(series)\n"
            "        ganador = resultados_05[nombre_serie][\"winner\"]\n"
            "        miembros = resultados_05[nombre_serie][\"ensemble_members\"]\n"
            "        ajuste_ensamble = make_ensemble_fit([ajuste for _n, ajuste in miembros.values()])\n"
            "        pronostico = ajuste_ensamble(ajuste_completo)(len(test))\n"
            "        ax.plot(test.index, test.to_numpy(), label=\"real\", color=\"black\")\n"
            "        ax.plot(test.index, pronostico, label=f\"pronóstico ({ganador})\", color=\"#d62728\")\n"
            "        ax.set_title(f\"{nombre_serie} -- test, mejor fase 05: {ganador}\")\n"
            "        ax.legend(fontsize=8)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    display(Image(filename=str(RUTA_FIGURAS_INFORME / \"05_test_forecast_best_model.png\")))\n"
        ),
        md(
            "### Conclusiones de la fase 05\n\n"
            "En **ALB**, `ensemble_equal_weight` es el mejor modelo de la fase en validación (MASE "
            "0.404, le gana a la naive 0.489) pero no supera a `catboost_tuned` (0.353, fase 03); "
            "`chronos_bolt_base` (0.598) es notable por ser zero-shot puro. `prophet_tuned` rinde "
            "peor que `prophet_default` en validación (0.659 vs. 0.532) pero **mejor en test** "
            "(MASE 1.578, el mejor de la fase) -- el tuneo optimiza sobre una ventana distinta de la "
            "de selección, así que no hay garantía de que transfiera. El híbrido `mstl_lightgbm` es "
            "el peor modelo del proyecto en test hasta esta fase (MASE 5.720): extrapolar la serie "
            "desestacionalizada 105 h hacia delante amplifica mucho más el error que en la ventana de "
            "48 h de validación. En **store_service**, `ensemble_equal_weight` (0.480126) y "
            "`seasonal_naive_m24` (0.480171) quedan prácticamente empatados -- la diferencia es "
            "ruido; en test gana `neuralprophet` (MASE 1.071) a pesar de haber sido el segundo peor "
            "de su familia en validación (0.694), otra inversión validación/test. Ninguna familia de "
            "la fase 05 domina de forma consistente ambos *splits* ni ambas series -- la misma "
            "conclusión que las fases 02-04 ya establecieron."
        ),
    ]
    return cells


def build_phase06_cells() -> list[nbf.NotebookNode]:
    cells = [
        md(
            "## Fase 06 — Comparación, selección y pronóstico final\n\n"
            "Junta los 32 modelos por serie evaluados en las fases 02 a 05 en un solo tablero, marca "
            "los modelos cuya validación está contaminada (se usó la misma ventana para elegir sus "
            "propios hiperparámetros o miembros), aplica la regla de selección acordada (menor MAE "
            "de validación entre los no contaminados, sin desempate por parsimonia), confirma qué "
            "tan estable es ese ranking en test (Spearman) y reentrena el modelo elegido -- junto "
            "con el peor, el SARIMA de TP1 y Chronos-Bolt -- sobre todos los datos limpios "
            "disponibles para pronosticar las 48 h siguientes con intervalos. Todo lo de esta "
            "subsección se calcula siempre en memoria a partir de las tablas de las fases 02-05 "
            "(`REENTRENAR` no cambia estos cálculos, que son puro pandas); solo el pronóstico final "
            "reentrena modelos reales, y solo cuando `REENTRENAR = True` se vuelve a escribir en "
            "`results/06_all_models.csv`/`results/06_final_forecast.csv`."
        ),
        code(
            "pool = load_pool()\n"
            "pool = add_contamination_flags(pool)\n"
            "pool = add_rank_columns(pool)\n"
            "if REENTRENAR:\n"
            "    pool.to_csv(RUTA_RESULTADOS / \"06_all_models.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalculó y sobrescribió 06_all_models.csv ({len(pool)} filas)\")\n"
            "else:\n"
            "    print(f\"REENTRENAR=False: tablero de {len(pool)} filas calculado en memoria (no se sobrescribe 06_all_models.csv)\")\n\n"
            "for nombre_serie in SERIES_NAMES:\n"
            "    contaminados = pool[(pool.series == nombre_serie) & (pool.split == \"val\") & (pool.val_contaminated)]\n"
            "    print(f\"\\n=== {nombre_serie}: modelos con validación contaminada ===\")\n"
            "    print(contaminados[[\"model\", \"family\", \"MASE\", \"contamination_reason\"]].to_string(index=False))\n"
        ),
        md(
            "### 6.1 Mejor modelo de cada familia y regla de selección\n\n"
            "**Regla acordada**: el modelo seleccionado por serie es el de menor MAE de validación "
            "entre los modelos elegibles (no contaminados), sin desempate por parsimonia. Las "
            "métricas de test se reportan siempre, nunca se usan para seleccionar."
        ),
        code(
            "FAMILY_COLORS = {\n"
            "    \"naive\": \"#999999\", \"SARIMA (TP1 refit)\": \"#1f77b4\", \"classical\": \"#17becf\",\n"
            "    \"ML\": \"#2ca02c\", \"ML (direct)\": \"#98df8a\", \"DL\": \"#d62728\", \"Prophet\": \"#9467bd\",\n"
            "    \"Hybrid\": \"#8c564b\", \"AutoML\": \"#e377c2\", \"Foundation\": \"#bcbd22\", \"Ensemble\": \"#ff7f0e\",\n"
            "}\n\n"
            "fig, axes = plt.subplots(1, len(SERIES_NAMES), figsize=(14, 6))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    bpf = best_per_family(pool, nombre_serie).sort_values(\"MASE\", ascending=False)\n"
            "    colores = [FAMILY_COLORS.get(f, \"#333333\") for f in bpf[\"family\"]]\n"
            "    barras = ax.barh(bpf[\"family\"] + \" (\" + bpf[\"model\"] + \")\", bpf[\"MASE\"], color=colores)\n"
            "    for barra, contaminado in zip(barras, bpf[\"val_contaminated\"]):\n"
            "        barra.set_hatch(\"//\" if contaminado else \"\")\n"
            "    mase_naive = pool.loc[(pool.series == nombre_serie) & (pool.split == \"val\") & (pool.model == \"seasonal_naive_m24\"), \"MASE\"].iloc[0]\n"
            "    ax.axvline(mase_naive, color=\"red\", linestyle=\"--\", linewidth=1, label=f\"seasonal_naive_m24 ({mase_naive:.3f})\")\n"
            "    ax.set_xlabel(\"MASE de validación\")\n"
            "    ax.set_title(SERIES_LABELS[nombre_serie])\n"
            "    ax.legend(fontsize=8, loc=\"lower right\")\n"
            "plt.suptitle(\"Mejor modelo de cada familia (rayado = validación contaminada)\")\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        code(
            "for nombre_serie in SERIES_NAMES:\n"
            "    res = select_model(pool, nombre_serie)\n"
            "    ref = tp1_and_foundation_reference(pool, nombre_serie)\n"
            "    fila_tp1 = ref[ref.family == \"SARIMA (TP1 refit)\"].iloc[0]\n"
            "    fila_chronos = ref[ref.family == \"Foundation\"].iloc[0]\n"
            "    fila_test = lambda modelo: pool[(pool.series == nombre_serie) & (pool.split == \"test\") & (pool.model == modelo)].iloc[0]\n"
            "    test_sel, test_naive, test_tp1 = fila_test(res.selected[\"model\"]), fila_test(\"seasonal_naive_m24\"), fila_test(fila_tp1[\"model\"])\n"
            "    print(f\"\\n=== {nombre_serie} ===\")\n"
            "    print(f\"Seleccionado : {res.selected['model']:<24} val MASE={res.selected['MASE']:.3f}  test MASE={test_sel['MASE']:.3f}\")\n"
            "    print(f\"Subcampeón   : {res.runner_up['model']:<24} val MASE={res.runner_up['MASE']:.3f}\")\n"
            "    print(f\"Peor elegible: {res.worst['model']:<24} val MASE={res.worst['MASE']:.3f}\")\n"
            "    print(f\"TP1 SARIMA   : {fila_tp1['model']:<24} val MASE={fila_tp1['MASE']:.3f}  test MASE={test_tp1['MASE']:.3f}\")\n"
            "    print(f\"Chronos-Bolt : val MASE={fila_chronos['MASE']:.3f}  test MASE={fila_test('chronos_bolt_base')['MASE']:.3f}\")\n"
            "    print(f\"Mejora sobre TP1 SARIMA   -- val: {pct_improvement(fila_tp1['MAE'], res.selected['MAE']):+.1f}%   test: {pct_improvement(test_tp1['MAE'], test_sel['MAE']):+.1f}%\")\n"
            "    print(f\"Mejora sobre seasonal_naive_m24 -- val: {pct_improvement(pool[(pool.series == nombre_serie) & (pool.split == 'val') & (pool.model == 'seasonal_naive_m24')].iloc[0]['MAE'], res.selected['MAE']):+.1f}%   test: {pct_improvement(test_naive['MAE'], test_sel['MAE']):+.1f}%\")\n"
        ),
        md(
            "### 6.2 ¿Se sostiene el ranking en test? Correlación de Spearman\n\n"
            "Un valor bajo confirma lo que las fases 02-05 ya venían mostrando: la ventana de "
            "validación de 48 h es una sola muestra, chica y ruidosa -- razón por la cual la regla de "
            "selección nunca usa test."
        ),
        code(
            "fig, axes = plt.subplots(1, len(SERIES_NAMES), figsize=(13, 6))\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    cons_todos = val_test_consistency(pool, nombre_serie, eligible_only=False)\n"
            "    cons_elegibles = val_test_consistency(pool, nombre_serie, eligible_only=True)\n"
            "    print(f\"{nombre_serie}: Spearman (32 modelos) rho={cons_todos.rho:.3f} (p={cons_todos.p_value:.3f}) | \"\n"
            "          f\"elegibles rho={cons_elegibles.rho:.3f} (p={cons_elegibles.p_value:.3f})\")\n"
            "    val_sub = pool[(pool.series == nombre_serie) & (pool.split == \"val\")]\n"
            "    test_sub = pool[(pool.series == nombre_serie) & (pool.split == \"test\")]\n"
            "    merged = val_sub.merge(test_sub, on=\"model\", suffixes=(\"_val\", \"_test\"))\n"
            "    for _, fila in merged.iterrows():\n"
            "        marcador = \"x\" if fila[\"val_contaminated_val\"] else \"o\"\n"
            "        ax.scatter(fila[\"MASE_val\"], fila[\"MASE_test\"], color=FAMILY_COLORS.get(fila[\"family_val\"], \"#333333\"),\n"
            "                    marker=marcador, s=55, edgecolors=\"black\", linewidths=0.4)\n"
            "    limite = max(merged[\"MASE_val\"].max(), merged[\"MASE_test\"].max()) * 1.05\n"
            "    ax.plot([0, limite], [0, limite], color=\"gray\", linestyle=\":\", linewidth=1, label=\"val = test\")\n"
            "    ax.set_xlabel(\"MASE de validación\")\n"
            "    ax.set_ylabel(\"MASE de test\")\n"
            "    ax.set_title(f\"{SERIES_LABELS[nombre_serie]} (x = validación contaminada)\")\n"
            "    ax.legend(fontsize=8)\n"
            "plt.suptitle(\"Validación vs. test por modelo, coloreado por familia\")\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 6.3 Análisis de sensibilidad (informativo — **no** es la regla de selección)\n\n"
            "¿Quién ganaría bajo otro criterio? (a) la regla adoptada; (b) MAE de validación "
            "incluyendo contaminados; (c) MAE de test como oráculo (imposible en la práctica)."
        ),
        code(
            "for nombre_serie in SERIES_NAMES:\n"
            "    print(f\"\\n=== {nombre_serie} ===\")\n"
            "    print(sensitivity_table(pool, nombre_serie).to_string(index=False))\n"
        ),
        md(
            "### 6.4 Comparación en la ventana de test y pronóstico final a 48 h\n\n"
            "El resumen de test (seleccionado vs. peor vs. TP1 vs. Chronos) exige reentrenar los "
            "cuatro roles sobre `train_and_val`; el pronóstico final los reentrena de nuevo sobre "
            "todos los datos limpios disponibles. Ambos son costosos (varios refits de `auto_arima`/"
            "SARIMA por serie), así que solo se recalculan con `REENTRENAR = True` -- con "
            "`REENTRENAR = False` se muestra la figura de test ya generada por "
            "`notebooks/06_selection_forecast.ipynb` y se relee/regrafica `06_final_forecast.csv` "
            "(sin reentrenar nada, solo graficando números ya calculados)."
        ),
        code(
            "if REENTRENAR:\n"
            "    fig, axes = plt.subplots(len(SERIES_NAMES), 1, figsize=(13, 4.2 * len(SERIES_NAMES)))\n"
            "    role_labels = {\"selected\": \"seleccionado\", \"worst\": \"peor\", \"tp1_foundational\": \"SARIMA TP1\", \"foundation_model\": \"Chronos-Bolt\"}\n"
            "    role_colors = {\"selected\": \"#2ca02c\", \"worst\": \"#d62728\", \"tp1_foundational\": \"#1f77b4\", \"foundation_model\": \"#bcbd22\"}\n"
            "    for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "        pronosticos, test = test_overlay_forecasts(nombre_serie)\n"
            "        ax.plot(test.index, test.to_numpy(), color=\"black\", linewidth=1.6, label=\"real\")\n"
            "        for rol, arreglo in pronosticos.items():\n"
            "            ax.plot(test.index, arreglo, color=role_colors[rol], linestyle=\"--\", linewidth=1.2, label=role_labels[rol])\n"
            "        ax.set_title(f\"{SERIES_LABELS[nombre_serie]} -- ventana de test (105h)\")\n"
            "        ax.legend(fontsize=8, ncol=4)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "    pronostico_final = build_final_forecast_table()\n"
            "    pronostico_final.to_csv(RUTA_RESULTADOS / \"06_final_forecast.csv\", index=False)\n"
            "    print(f\"REENTRENAR=True: se recalculó y sobrescribió 06_final_forecast.csv ({len(pronostico_final)} filas)\")\n"
            "else:\n"
            "    display(Image(filename=str(RUTA_FIGURAS_INFORME / \"06_test_overlay_forecast.png\")))\n"
            "    pronostico_final = pd.read_csv(RUTA_RESULTADOS / \"06_final_forecast.csv\", parse_dates=[\"timestamp\"])\n"
            "    print(f\"REENTRENAR=False: se reutiliza results/06_final_forecast.csv ({len(pronostico_final)} filas), no se reentrena nada.\")\n"
        ),
        code(
            "role_labels = {\"selected\": \"seleccionado\", \"worst\": \"peor\", \"tp1_foundational\": \"SARIMA TP1\", \"foundation_model\": \"Chronos-Bolt\"}\n"
            "role_colors = {\"selected\": \"#2ca02c\", \"worst\": \"#d62728\", \"tp1_foundational\": \"#1f77b4\", \"foundation_model\": \"#bcbd22\"}\n\n"
            "fig, axes = plt.subplots(len(SERIES_NAMES), 1, figsize=(13, 4.5 * len(SERIES_NAMES)))\n"
            "horas_historial = 7 * 24\n"
            "for ax, nombre_serie in zip(axes, SERIES_NAMES):\n"
            "    historial = SERIES[nombre_serie].iloc[-horas_historial:]\n"
            "    ax.plot(historial.index, historial.to_numpy(), color=\"black\", linewidth=1.2, label=\"histórico reciente\")\n"
            "    sub = pronostico_final[pronostico_final.series == nombre_serie]\n"
            "    for rol in (\"selected\", \"worst\", \"tp1_foundational\", \"foundation_model\"):\n"
            "        df_rol = sub[sub.role == rol].sort_values(\"timestamp\")\n"
            "        ax.plot(df_rol.timestamp, df_rol.yhat, color=role_colors[rol], linewidth=1.4, label=role_labels[rol])\n"
            "        if rol == \"selected\":\n"
            "            tiene_95 = df_rol[\"lower_95\"].notna().all()\n"
            "            col_inf, col_sup = (\"lower_95\", \"upper_95\") if tiene_95 else (\"lower_80\", \"upper_80\")\n"
            "            ax.fill_between(df_rol.timestamp, df_rol[col_inf], df_rol[col_sup], color=role_colors[rol], alpha=0.15,\n"
            "                              label=f\"intervalo {'95' if tiene_95 else '80'}% (seleccionado)\")\n"
            "    ax.axvline(historial.index[-1], color=\"gray\", linestyle=\":\", linewidth=1)\n"
            "    ax.set_title(f\"{SERIES_LABELS[nombre_serie]} -- pronóstico 48h (2026-09-26 23:00 -> 2026-09-28 22:00 UTC)\")\n"
            "    ax.legend(fontsize=8, ncol=3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### Conclusiones de la fase 06\n\n"
            "**ALB**: seleccionado `auto_arima` (val MAE 18.807,5; MASE 0,475), apenas por debajo de "
            "`seasonal_naive_m24` (subcampeón, 0,489) y `autogluon_timeseries` (tercero, 0,499) -- "
            "una diferencia pequeña, no una goleada. Le gana al SARIMA fundacional de TP1 por 48,5% "
            "en validación, pero en test la relación se invierte: pierde contra el SARIMA de TP1 "
            "(-34,9%, el mejor modelo de los 32 en test, MASE 1,526) y contra la naive estacional "
            "(-3,4%). `catboost_tuned` (el \"ganador\" si se permitiera la contaminación) tiene MASE "
            "de validación mucho mejor (0,353) pero colapsa en test (2,313) -- justo el riesgo que la "
            "regla busca evitar. Peor modelo elegible: `average` (MASE 3,79).\n\n"
            "**store_service**: seleccionado, literalmente, `seasonal_naive_m24` -- ningún modelo más "
            "sofisticado logró bajarle el MAE de validación sin contaminar. `ensemble_equal_weight` "
            "queda estadísticamente empatado pero excluido por construcción. Mejora sobre el SARIMA "
            "de TP1: +25,8% en validación, revertida a -1,2% en test (inconsistencia mucho más leve "
            "que en ALB). Peor modelo elegible: `drift` (MASE 3,58).\n\n"
            "**¿Le ganaron los modelos avanzados al enfoque de TP1?** En validación, sí en las dos "
            "series -- pero el que gana termina siendo un modelo clásico (`auto_arima`) o la naive "
            "estacional, no ML/DL/AutoML. En test, no: el SARIMA fundacional de TP1 iguala o supera "
            "al modelo seleccionado en ambas series. Bajo el único criterio válido para elegir sin "
            "ver el futuro (MAE de validación, sin contaminar), los modelos más sofisticados de las "
            "fases 03-05 no superaron a los baselines clásicos y a la naive estacional en ninguna de "
            "las dos series -- el hallazgo más consistente de todo el proyecto, no una sorpresa de "
            "esta fase en particular. La correlación de Spearman validación/test, baja-a-moderada en "
            "ambas series, es una confirmación numérica adicional de que la ventana de validación "
            "única de 48 h es ruidosa (limitación del protocolo, aceptada por presupuesto de tiempo "
            "desde la fase 02 -- no evidencia de que el test debería usarse para elegir)."
        ),
    ]
    return cells


# ---------------------------------------------------------------------------
# Section 3 — serie pública adicional (Wikipedia en español)
# ---------------------------------------------------------------------------


def build_wikipedia_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 3. Serie pública adicional: visitas horarias de Wikipedia en español\n\n"
            "**¿Qué es y por qué está acá?** La consigna pide tres series (`consignas.md`, punto 1). "
            "Además de `alb` y `store_service` (infraestructura de BodegaAI) se incorpora una serie "
            "**pública**: las visitas por hora de usuarios a Wikipedia en español (API de métricas de "
            "páginas vistas de Wikimedia, proyecto `es.wikipedia`, acceso `all-access`, agente `user`; "
            "datos bajo licencia CC0). Comparte con las otras dos el mismo calendario (2.063 horas del "
            "2026-07-03 al 2026-09-26, sin huecos), el ciclo diario y las mismas ventanas de la fase 02 "
            "(validación de 48 h, test de 105 h), pero es tráfico externo, sin el deploy/rollback del 17 "
            "al 22 de septiembre y con una autocorrelación semanal (rezago 168) mucho más alta.\n\n"
            "**Alcance, dicho sin rodeos.** Es un *baseline de contraste*, no una réplica de las "
            "Secciones 2.1-2.6: seis modelos de cinco familias con **hiperparámetros por defecto**, sin "
            "Deep Learning, sin AutoML, sin Chronos y sin búsqueda de hiperparámetros. La regla de "
            "selección es la misma (menor MAE de validación) y los resultados de `alb` y `store_service` "
            "**no se recalculan**. Por eso esta serie **no** se agrega a `SERIES_NAMES`: las funciones "
            "de las Secciones 2.x asumen las 32 configuraciones por serie y sus tablas persistidas.\n\n"
            "**Datos.** `data/wikipedia-es-pageviews-hourly-since-2026-05-01.csv` (mismas columnas que "
            "los CSV de AWS). Se descarga de "
            "`https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/es.wikipedia/all-access/user/hourly/2026050100/2026092622`; "
            "el repositorio incluye el script `experiments/00_download_wikipedia_es.py`. El registro "
            "`wikipedia_es` de `SERIES_REGISTRY` (Sección 1, `data.py`) la marca con "
            "`\"intervention\": \"no\"`, de modo que `load_clean` no le aplica la imputación del rollback."
        ),
        md(
            "## 3.1 La serie y su estructura frente a las de BodegaAI\n\n"
            "La primera tabla compara la autocorrelación del logaritmo en los rezagos 24 (ciclo diario) "
            "y 168 (ciclo semanal). En Wikipedia casi no decae entre 24 y 168 horas; en ALB baja "
            "claramente. La segunda tabla muestra que la relación con las series de BodegaAI es de "
            "contexto (mismo ritmo diario) y no de dependencia: los cambios a 24 h prácticamente no "
            "están correlacionados."
        ),
        code(r"""
NOMBRE_WIKI = "wikipedia_es"

wiki = load_clean(NOMBRE_WIKI)  # sin imputación de intervención (registro: "intervention": "no")
wiki_train, wiki_val, wiki_test = split(wiki)
print(f"{NOMBRE_WIKI}: {len(wiki)} horas, {wiki.index[0]} -> {wiki.index[-1]}")
print(f"train {len(wiki_train)} h | val {len(wiki_val)} h | test {len(wiki_test)} h")

filas = []
for nombre in [*SERIES_NAMES, NOMBRE_WIKI]:
    x = np.log(load_clean(nombre).astype(float))
    filas.append({"serie": nombre, "autocorr_rezago_24": x.autocorr(24), "autocorr_rezago_168": x.autocorr(168)})
estructura = pd.DataFrame(filas).set_index("serie").round(3)
print()
print(estructura.to_string())

relacion = []
for nombre in SERIES_NAMES:
    conjunta = pd.concat(
        [np.log(load_clean(nombre).astype(float)).rename("x"), np.log(wiki.astype(float)).rename("w")], axis=1
    ).dropna()
    relacion.append({
        "contra": nombre,
        "corr_niveles": conjunta["x"].corr(conjunta["w"]),
        "corr_cambios_24h": conjunta["x"].diff(24).corr(conjunta["w"].diff(24)),
    })
print()
print(pd.DataFrame(relacion).set_index("contra").round(3).to_string())

fig, ax = plt.subplots(figsize=(13, 3.6))
reciente = wiki.loc[wiki.index[-1] - pd.Timedelta(days=21):]
ax.plot(reciente.index, reciente.to_numpy(), color="#1f77b4", lw=1.1)
ax.axvspan(wiki_test.index[0], wiki_test.index[-1], color="#d62728", alpha=0.12, label="ventana de test")
ax.set_title("es.wikipedia: visitas horarias de usuarios (últimas 3 semanas)")
ax.legend()
fig.tight_layout()
plt.show()
"""),
        md(
            "## 3.2 Seis modelos rápidos con hiperparámetros por defecto\n\n"
            "`seasonal_naive_m24` (naive), `holt_winters` (clásico), `SARIMA(1, 1, 1)x(1, 1, 1, 24)` "
            "(la especificación de ALB del TP1, **no ajustada a esta serie**), `ridge` y `lightgbm` "
            "(recursivos, con rezagos de hasta 336 h y calendario) y `prophet_default`. Mismo "
            "protocolo de la fase 02: un ajuste sobre `train` para validar (48 h) y otro sobre "
            "`train + val + tramo de intervención` para el test (105 h). MASE con m=24. Con "
            "`REENTRENAR = False` se reutiliza `results/07_wikipedia.csv` si existe; si no existe, o con "
            "`REENTRENAR = True`, se ajustan los seis modelos (unos 15 segundos en total)."
        ),
        code(r"""
ARCHIVO_RESULTADOS_07 = RUTA_RESULTADOS / "07_wikipedia.csv"
ARCHIVO_PRONOSTICO_07 = RUTA_RESULTADOS / "07_wikipedia_forecast.csv"

modelos_wiki = [
    ("seasonal_naive_m24", "naive", NAIVE_MODELS["seasonal_naive_m24"]),
    ("holt_winters", "classical", CLASSICAL_MODELS["holt_winters"]),
    ("SARIMA(1, 1, 1)x(1, 1, 1, 24)", "SARIMA (TP1 refit)", make_sarima_fit((1, 1, 1), (1, 1, 1, 24))),
    ("ridge", "ML", RECURSIVE_MODELS["ridge"]),
    ("lightgbm", "ML", RECURSIVE_MODELS["lightgbm"]),
    ("prophet_default", "Prophet", prophet_default_fit),
]

if REENTRENAR or not ARCHIVO_RESULTADOS_07.exists():
    tabla_wiki = ResultsTable()
    for nombre, familia, ajuste in modelos_wiki:
        evaluate(ajuste, model=nombre, family=familia, series_name=NOMBRE_WIKI, series=wiki, table=tabla_wiki)
    res_wiki = tabla_wiki.to_frame()
    res_wiki.to_csv(ARCHIVO_RESULTADOS_07, index=False)
else:
    res_wiki = pd.read_csv(ARCHIVO_RESULTADOS_07)

columnas = ["model", "family", "MAE", "MAPE_%", "MASE", "fit_time_s"]
for ventana, etiqueta in [("val", "Validación (48 h)"), ("test", "Test (105 h)")]:
    print(f"\n{etiqueta}")
    print(res_wiki[res_wiki["split"] == ventana].sort_values("MAE")[columnas].round(3).to_string(index=False))
"""),
        md(
            "## 3.3 Selección, desacuerdo validación/test y pronóstico de 48 h\n\n"
            "Misma regla que en el resto del proyecto: **el modelo seleccionado es el de menor MAE de "
            "validación**. Se reporta además el mejor y el peor en test. La validación es una sola "
            "ventana de 48 h y el test una sola de 105 h: si los rankings discrepan, se dice, no se "
            "elige por test. El pronóstico final reajusta el modelo seleccionado con toda la historia "
            "disponible y proyecta 48 horas fuera de muestra (sin intervalos de predicción)."
        ),
        code(r"""
rank_val = res_wiki[res_wiki["split"] == "val"].sort_values("MAE").reset_index(drop=True)
rank_test = res_wiki[res_wiki["split"] == "test"].sort_values("MAE").reset_index(drop=True)
seleccionado_wiki = rank_val.loc[0, "model"]
print(f"Seleccionado por validación : {seleccionado_wiki} (MAE val {rank_val.loc[0, 'MAE']:,.0f})")
print(f"Mejor en test               : {rank_test.loc[0, 'model']} (MAE test {rank_test.loc[0, 'MAE']:,.0f})")
print(f"Peor en test                : {rank_test.iloc[-1]['model']} (MAE test {rank_test.iloc[-1]['MAE']:,.0f})")

unidos = rank_val[["model", "MAE"]].merge(rank_test[["model", "MAE"]], on="model", suffixes=("_val", "_test"))
rho_wiki = unidos["MAE_val"].rank().corr(unidos["MAE_test"].rank())
print(f"Correlación de Spearman entre rankings val y test (6 modelos): {rho_wiki:.2f}")

horizonte_wiki = 48
if REENTRENAR or not ARCHIVO_PRONOSTICO_07.exists():
    ajuste_por_nombre = {nombre: ajuste for nombre, _, ajuste in modelos_wiki}
    indice_futuro = pd.date_range(wiki.index[-1] + pd.Timedelta(hours=1), periods=horizonte_wiki, freq="h", tz="UTC")
    pron_wiki = pd.DataFrame({
        "timestamp": indice_futuro,
        "series": NOMBRE_WIKI,
        "role": "selected",
        "model": seleccionado_wiki,
        "yhat": ajuste_por_nombre[seleccionado_wiki](wiki)(horizonte_wiki),
    })
    pron_wiki.to_csv(ARCHIVO_PRONOSTICO_07, index=False)
else:
    pron_wiki = pd.read_csv(ARCHIVO_PRONOSTICO_07, parse_dates=["timestamp"])
print(f"\nPronóstico final: {len(pron_wiki)} h desde {pron_wiki['timestamp'].iloc[0]} hasta {pron_wiki['timestamp'].iloc[-1]}")

# Figuras del informe (solo lectura, como en la Sección 2.3): MAE de test y pronóstico en test de los 3 mejores,
# y pronóstico final de 48 h.
from IPython.display import Image, display

for archivo_figura in ["07_wikipedia_test.png", "07_wikipedia_forecast.png"]:
    ruta_figura = RUTA_FIGURAS_INFORME / archivo_figura
    if ruta_figura.exists():
        display(Image(filename=str(ruta_figura)))
    else:
        print(f"(figura no encontrada: {ruta_figura})")

assert len(wiki) == 2063
assert len(res_wiki) == len(modelos_wiki) * 2
assert len(pron_wiki) == horizonte_wiki and pron_wiki["yhat"].notna().all()
"""),
        md(
            "## 3.4 Comparación de los seis modelos comunes en las tres series\n\n"
            "La tabla reúne el MASE de validación y de test de los seis modelos que se ejecutaron sobre las "
            "tres series (para `alb` y `store_service` se toman de `results/06_all_models.csv`, la batería "
            "completa de 32 configuraciones; para `wikipedia_es`, de los resultados de la sección 3.2). El SARIMA "
            "usa la especificación de ALB en `alb` y en `wikipedia_es`, y la del servicio POS API del TP1 en "
            "`store_service`. Es una lectura descriptiva: no se recalcula ninguna serie."
        ),
        code(r"""
pool_06 = pd.read_csv(RUTA_RESULTADOS / "06_all_models.csv")
MODELOS_COMUNES = ["seasonal_naive_m24", "prophet_default", "lightgbm", "SARIMA (TP1 refit)", "ridge", "holt_winters"]


def mase_comun(df, modelo, ventana):
    if modelo == "SARIMA (TP1 refit)":
        filas = df[df["family"] == "SARIMA (TP1 refit)"]
    else:
        filas = df[df["model"] == modelo]
    return float(filas[filas["split"] == ventana]["MASE"].iloc[0])


comparacion_tres = {}
for serie, df_serie in [
    ("alb", pool_06[pool_06["series"] == "alb"]),
    ("store_service", pool_06[pool_06["series"] == "store_service"]),
    (NOMBRE_WIKI, res_wiki),
]:
    for ventana in ("val", "test"):
        comparacion_tres[(serie, ventana)] = {m: mase_comun(df_serie, m, ventana) for m in MODELOS_COMUNES}

tabla_tres = pd.DataFrame(comparacion_tres).loc[MODELOS_COMUNES]
print("MASE (menor es mejor)")
print(tabla_tres.round(3).to_string())
print("\nMejor modelo por columna:")
print(tabla_tres.idxmin().to_string())
"""),
        md(
            "### Conclusiones de la sección 3\n\n"
            "Por la regla de selección adoptada, el modelo seleccionado es la referencia estacional "
            "ingenua (`seasonal_naive_m24`, MAE de validación 21.146,77; MASE 0,438), seguida por "
            "Prophet (0,655) y LightGBM (0,727). En test, en cambio, el mejor modelo es `ridge` "
            "(MASE 0,784), la referencia ingenua queda segunda (0,924) y el SARIMA del TP1 es el peor "
            "(MASE 2,513), precedido por Holt-Winters (2,264). La correlación de Spearman entre ambos "
            "rankings, sobre solo seis modelos, es 0,09: el desacuerdo validación/test de las series de "
            "BodegaAI se repite y, con una sola ventana de cada tipo, no permite declarar un ganador "
            "firme.\n\n"
            "El contraste principal es que el SARIMA del TP1 fue el mejor modelo de test en ALB (MASE "
            "1,526) y es el peor en Wikipedia: el desempeño relativo del enfoque clásico depende de la "
            "serie. Holt-Winters y SARIMA, que solo usan estacionalidad de período 24, ocupan los "
            "últimos lugares, mientras que `ridge` usa rezagos de hasta 336 h y calendario y Prophet "
            "incluye por defecto un componente semanal; es una explicación plausible, **no contrastada "
            "en este trabajo**. Entre los seis modelos comunes a las tres series (sección 3.4), la referencia "
            "estacional ingenua tiene el menor MASE de validación en todas, pero ningún modelo es el mejor en test en "
            "más de una serie (SARIMA en ALB, Prophet en store_service y Ridge en Wikipedia). Límites: seis modelos sin ajuste de hiperparámetros, SARIMA con la "
            "especificación de ALB, una sola ventana de validación de 48 h y una de test de 105 h, y "
            "pronóstico final sin intervalos de predicción. Ridge y LightGBM reciben como variable exógena la marca de intervención del despliegue, que vale 1 entre el 17 y el 22 de septiembre aunque esta serie no fue afectada; con la marca en cero, el MASE de prueba de Ridge pasa de 0,784 a 0,756 y el de LightGBM no cambia, sin alterar el orden de los modelos ni el de validación."
        ),
    ]


# ---------------------------------------------------------------------------
# Section 4 — conclusiones y limitaciones
# ---------------------------------------------------------------------------


def build_conclusion_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            "# 4. Conclusiones generales y limitaciones\n\n"
            "## Síntesis por serie\n\n"
            "**ALB**: el modelo seleccionado (`auto_arima`, MASE de validación 0,475) le gana al "
            "refit SARIMA de TP1 en validación por 48,5%, pero pierde contra él en test por 34,9% -- "
            "el SARIMA fundacional de TP1 termina siendo el mejor de los 32 modelos evaluados en la "
            "ventana de test real. **store_service**: el modelo seleccionado es la propia "
            "`seasonal_naive_m24` -- ningún modelo ML, DL, Prophet, AutoML o de fundación logró "
            "bajarle el MAE de validación de forma no contaminada; en test, NeuralProphet gana "
            "(MASE 1,071) a pesar de haber sido el 14° de 32 en validación, evidencia de que el "
            "modelo que hubiera elegido la regla honesta no es el mejor en test, pero tampoco había "
            "forma de saberlo sin ver el futuro. **wikipedia_es** (serie pública, Sección 3, seis modelos "
            "sin ajuste de hiperparámetros): la regla de validación elige `seasonal_naive_m24`, pero en "
            "test gana `ridge` (MASE 0,784) y el SARIMA del TP1 es el peor (2,513): el desempeño "
            "relativo del enfoque clásico depende de la serie (mejor en ALB, peor en Wikipedia).\n\n"
            "## Consistencia entre validación y test\n\n"
            "La correlación de Spearman entre el ranking de validación y el de test es baja y no "
            "significativa en ALB ($\\rho=0.237$, $p=0.192$, sobre 32 modelos) y moderada y "
            "significativa en store_service ($\\rho=0.654$, $p<0.001$). En ambas series, y en cada "
            "fase desde la 02, el modelo que gana en la ventana de validación de 48 h no es "
            "necesariamente el que mejor pronostica la ventana de test de 105 h -- la conclusión "
            "transversal más consistente de todo el proyecto es que el enfoque clásico de TP1 "
            "(SARIMA) no fue superado de forma confiable por los modelos de ML/DL/AutoML/fundación "
            "más sofisticados de las fases 03 a 05, bajo el único criterio honesto disponible para "
            "elegir un modelo sin conocer el futuro.\n\n"
            "## Límites del trabajo\n\n"
            "1. Una sola ventana de validación de 48 h (no *backtesting* expansivo de múltiples "
            "ventanas), decidida por presupuesto de tiempo frente a ~15-32 modelos por serie, varios "
            "de ellos costosos de entrenar (fase 02, `status/02-evaluation-baselines.md`) -- la causa "
            "principal de la divergencia validación/test documentada en cada fase.\n"
            "2. Historia acotada: ~86 días de régimen estable (2.063 observaciones horarias por "
            "serie), insuficiente para estimar estacionalidad anual y limitante para arquitecturas "
            "DL de mayor capacidad.\n"
            "3. Existen modelos con validación contaminada (tuneados o seleccionados sobre la misma "
            "ventana de validación); se muestran en las tablas completas pero se excluyen de la "
            "regla de selección, nunca al revés.\n"
            "4. Posible anomalía sin verificar en las últimas horas observadas de `store_service` "
            "(20:00-22:00 UTC del 2026-09-26); no se investigó en profundidad si se trata de un "
            "artefacto de exportación de datos o de una señal real.\n"
            "5. La estacionalidad semanal (periodo 168), documentada como negligible en la fase 01, "
            "no se explotó como componente principal en ningún modelo -- una limitación conocida de "
            "TP1 que, con ~13 semanas de historia disponibles, podría revisarse con más datos.\n"
            "6. La serie pública de Wikipedia (Sección 3) se evaluó como contraste acotado: seis modelos "
            "por defecto, sin DL, AutoML ni fundación, con el SARIMA de ALB, una ventana de validación de "
            "48 h y una de test de 105 h; no es comparable uno a uno con las 32 configuraciones de las "
            "series de BodegaAI.\n\n"
            "## Líneas de trabajo futuro\n\n"
            "Validación cruzada de múltiples orígenes (*origin*) en el sentido de Hyndman & "
            "Athanasopoulos, en vez de un único corte, si el presupuesto de tiempo lo permite; "
            "extender la historia disponible más allá de los ~86 días actuales; y extender a la serie "
            "pública de Wikipedia (Sección 3) la batería completa de las fases 03 a 06."
        ),
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    cells.extend(build_header_cells())
    cells.extend(build_module_cells())
    cells.extend(build_phase01_cells())
    cells.extend(build_phase02_cells())
    cells.extend(build_phase03_cells())
    cells.extend(build_phase04_cells())
    cells.extend(build_phase05_cells())
    cells.extend(build_phase06_cells())
    cells.extend(build_wikipedia_cells())
    cells.extend(build_conclusion_cells())

    # Deterministic cell ids (nbformat >= 4.5 requires an `id` per cell); derived from position so
    # re-running this generator on unchanged inputs produces a byte-identical notebook.
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:04d}"

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (mcd-ua-tsa)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT_NOTEBOOK)
    print(f"Wrote {len(cells)} cells to {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
