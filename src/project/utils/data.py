import hashlib

from astropy.table import Table
from astropy.visualization import quantity_support
from astroquery.gaia import Gaia

from project.utils import DATA_RAW_DIR
from project.utils.load import fast_load_table, load_with_polars

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
quantity_support()


output_dir = DATA_RAW_DIR


def filename_from_query(query: str) -> str:
    _hash = hashlib.md5(query.encode()).hexdigest()
    return f"GaiaQuery_{_hash}.ecsv"


def query_data(query: str, filename: str | None = None, fast: bool = False):
    """
    Consulta datos de Gaia utilizando la API y guarda el resultado en un archivo.
    Si el archivo ya existe, lo lee y devuelve el resultado sin volver a consultar.
    """
    if filename is None:
        path = output_dir / filename_from_query(query)
    else:
        path = output_dir / filename
        if not path.exists():
            query_filename = filename_from_query(query)
            query_path = output_dir / query_filename
            if query_path.exists():
                query_path.rename(path)

    if path.exists():
        return fast_load_table(path) if fast else Table.read(path)

    job = Gaia.launch_job_async(query)
    job.get_results().write(path)
    return job.get_results()


def query_data_with_header(query: str):
    return load_with_polars(output_dir / filename_from_query(query))
