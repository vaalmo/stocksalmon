from pathlib import Path
import pandas as pd
PD_KW = dict(dtype_backend="pyarrow")
DATA_DIR = Path("../data/raw")

path_chess   = DATA_DIR / "chessData.csv"
path_random  = DATA_DIR / "random_evals.csv"
path_tactic  = DATA_DIR / "tactic_evals.csv"

def load_csv(path: Path, usecols=None, dtypes=None, nrows=None):
    """
    Carga un CSV a DataFrame de forma robusta.
    - usecols: columnas a leer (si quieres probar con un subconjunto).
    - dtypes: mapeo de tipos por columna si ya los conoces.
    - nrows: para muestrear primeras N filas si necesitas una carga rápida.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    try:
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtypes,
            low_memory=False,    # evita inferencias parciales
            on_bad_lines="warn", # avisa si hay líneas corruptas
            **PD_KW               # usa backend pyarrow si está disponible
        )
    except TypeError:
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtypes,
            low_memory=False,
            on_bad_lines="warn",
        )
    return df
