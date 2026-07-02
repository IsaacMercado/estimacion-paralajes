import contextlib
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_dir: Path):
    name = url.strip().split("/")[-1]
    filename = output_dir / name

    if filename.exists():
        statinfo = filename.stat()

        with contextlib.closing(urlopen(url)) as fp:
            headers = fp.info()
            length = int(headers["content-length"])

        if statinfo.st_size == length:
            print(f"Skipping {name}, file already downloaded")
            return
        else:
            print(f"Redownloading {name}")

    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=name,
    ) as bar:
        urlretrieve(url, filename=filename, reporthook=bar.update_to)
