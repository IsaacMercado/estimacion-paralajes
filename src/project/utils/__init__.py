from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.parent

is_project_folder = bool(
    {
        path
        for path in ROOT_DIR.iterdir()
        if path.is_file() and path.name == "pyproject.toml"
    }
)

if not is_project_folder:
    ROOT_DIR = Path.cwd().resolve()

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_RAW_DIR = DATA_DIR / "raw"

if not is_project_folder:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
