import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from astropy.table import Table

from project.utils import DATA_RAW_DIR

polars_dtypes = {
    np.dtype("bool"): pl.Boolean,
    np.dtype("int8"): pl.Int8,
    np.dtype("int16"): pl.Int16,
    np.dtype("int64"): pl.Int64,
    np.dtype("float32"): pl.Float32,
    np.dtype("float64"): pl.Float64,
    np.dtype("<U1"): pl.String,
}


def dict_dtypes(columns: Table.TableColumns) -> dict[str, pl.DataType]:
    return {
        cname: polars_dtypes[columns[cname].dtype]
        for cname in columns
        if columns[cname].dtype in polars_dtypes
    }


def load_header_lines(file):
    while True:
        tell = file.tell()
        line = file.readline()
        yield line
        if not line.startswith("#"):
            file.seek(tell)
            break


def dict_units(columns: Table.TableColumns) -> dict[str, str]:
    return {cname: columns[cname].unit for cname in columns if columns[cname].unit}


def load_gz_with_polars(
    pattern: str, columns: list[str] | None = None
) -> tuple[pl.LazyFrame, Table.TableColumns]:
    filename = next(DATA_RAW_DIR.glob(pattern))

    with gzip.open(filename, "rt") as file:
        table = Table.read(
            tuple(load_header_lines(file)),
            format="ascii.ecsv",
        )

    df = pl.scan_csv(
        str(DATA_RAW_DIR / pattern),
        separator=",",
        comment_prefix="#",
        schema=dict_dtypes(table.columns),
        null_values=("null",),
    )

    if columns is not None:
        df = df.select(columns)

    return df, table.columns


def load_with_polars(filename: str | Path) -> tuple[pl.DataFrame, Table.TableColumns]:
    with open(filename, "r") as file:
        table = Table.read(
            tuple(load_header_lines(file)),
            format="ascii.ecsv",
        )

        df = pl.read_csv(
            file,
            separator=" ",
            comment_prefix="#",
            schema=dict_dtypes(table.columns),
        )

    return df, table.columns


def fast_load_table(filename: str | Path, index: bool = False) -> Table:
    df, columns = load_with_polars(filename)
    return Table.from_pandas(df.to_pandas(), index=index, units=dict_units(columns))


def get_gaia_md5sum(path: str, has_md5sum: bool = True):
    url_prefix = f"http://cdn.gea.esac.esa.int/Gaia/{path.strip('/')}"
    df = pd.read_csv(
        f"{url_prefix}/_MD5SUM.txt",
        header=None,
        sep=r"\s+",
        names=["md5Sum", "file"],
    )

    if has_md5sum:
        # The last row in the "_MD5SUM.txt" file in the DR3 directories includes the md5Sum value of the _MD5SUM.txt file
        df.drop(df.tail(1).index, inplace=True)

    df["url"] = df["file"].apply(lambda x: f"{url_prefix}/{x}")

    df[["healpix8_min", "healpix8_max"]] = (
        df["file"]
        .str.extract(r"_(?P<healpix8_min>\d+)-(?P<healpix8_max>\d+).csv")
        .astype(int)
    )

    # Compute HEALPix levels 6,7, and 9 ===========================================
    df["healpix7_min"] = df["healpix8_min"].apply(lambda x: x >> 2)
    df["healpix7_max"] = df["healpix8_max"].apply(lambda x: x >> 2)

    df["healpix6_min"] = df["healpix7_min"].apply(lambda x: x >> 2)
    df["healpix6_max"] = df["healpix7_max"].apply(lambda x: x >> 2)

    df["healpix9_min"] = df["healpix8_min"].apply(lambda x: x << 2)
    df["healpix9_max"] = df["healpix8_max"].apply(lambda x: (x << 2) + 3)

    return df


if __name__ == "__main__":
    md5sum = get_gaia_md5sum("gdr3/gaia_source", has_md5sum=True)
    print(md5sum)
