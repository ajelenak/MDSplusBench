import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict, namedtuple
from configparser import ConfigParser
from dataclasses import dataclass, field
from itertools import batched, chain, product
from multiprocessing import cpu_count
from pathlib import Path
from typing import Generator

import h5py
import numpy as np
import pandas as pd
from dask.distributed import Client

lggr = logging.getLogger("mdsplusml-bench")

MiB = 1024 * 1024


def parse_cli() -> argparse.Namespace:
    """Process command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MDSplusML HDF5 benchmark script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infiles",
        help="Input HDF5 files to run benchmarks on.",
        type=str,
        nargs="+",
        metavar="INFILES",
    )
    parser.add_argument(
        "--pb",
        help="Page buffer size in bytes.",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--to-csv",
        help="Save benchmark data to a CSV file",
        type=str,
        metavar="CSVFILE",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )

    return parser.parse_args()


def bench_params(**kwargs) -> Generator:
    """Return a tuple of namedtuples with benchmark runtime parameters."""
    BenchParams = namedtuple("BenchParams", kwargs.keys())
    return (BenchParams(*_) for _ in product(*kwargs.values()))


@dataclass
class Timer:
    """Timer for measuring elapsed time."""

    name: str
    _tic: float | None = field(default=None, init=False, repr=False)
    _toc: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start timer"""
        if self._tic is not None:
            raise RuntimeWarning(
                f'Timer "{self.name}" is running. Stop it with .stop().'
            )
        self._tic = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._tic is None:
            raise RuntimeError(
                f'Timer "{self.name}" not running. Use .start() to start it.'
            )
        self._toc = time.perf_counter()
        return self._toc - self._tic

    def elapsed(self) -> float:
        """Elapsed time of the timer in seconds."""
        if self._tic is None:
            raise RuntimeError(
                f'Timer "{self.name}" not running. Use .start() to start it.'
            )
        if self._toc is None:
            raise ValueError(f'Timer "{self.name}" still running.')
        return self._toc - self._tic

    def __repr__(self) -> str:
        pre = f'Timer "{self.name}" at {hex(id(self))}'
        if self._tic is None:
            return f"<{pre} not yet started>"
        elif self._toc is None:
            return f"<{pre} running, started at {self._tic} seconds>"
        else:
            return f"<{pre}: Elapsed {self.elapsed():.4f} seconds>"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stop()


def get_s3_params(need_region: bool = False) -> dict[str, bytes]:
    """Collect S3 connection parameters."""
    s3p = dict()

    # Read AWS credentials and config files...
    home = Path.home()
    creds = ConfigParser()
    creds.read(
        os.getenv("AWS_SHARED_CREDENTIALS_FILE", home.joinpath(".aws", "credentials"))
    )
    config = ConfigParser()
    config.read(os.getenv("AWS_CONFIG_FILE", home.joinpath(".aws", "config")))

    profile = os.getenv("AWS_PROFILE", "default")
    s3p["secret_id"] = os.getenv(
        "AWS_ACCESS_KEY_ID", creds.get(profile, "aws_access_key_id", fallback="")
    ).encode("ascii")
    s3p["secret_key"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY",
        creds.get(profile, "aws_secret_access_key", fallback=""),
    ).encode("ascii")
    s3p["session_token"] = os.getenv(
        "AWS_SESSION_TOKEN",
        creds.get(profile, "aws_session_token", fallback=""),
    ).encode("ascii")
    if need_region:
        s3p["aws_region"] = os.getenv(
            "AWS_REGION", config.get(profile, "region")
        ).encode("ascii")

    return s3p


def gather_dset_info(h5f: h5py.File) -> dict[int, list[str]]:
    """Discover all shot signal datasets in the HDF5 file.

    Every dataset not a dimension scale is assumed to be holding signal data. It
    also needs to have a `shot` attribute.
    """
    shots = defaultdict(list)

    def dset_info(name: str, h5obj: h5py.HLObject):
        if isinstance(h5obj, h5py.Dataset):
            if not h5obj.is_scale:
                shots[h5obj.attrs["shot"]].append(h5obj.name)

    h5f.visititems(dset_info)
    return shots


def reader(h5file: str, obj: dict[int, list[str]], **h5f_kwargs) -> dict[str, float]:
    """Read data for supplied selection of shots/signals in the given HDF5 file."""
    bench_data = dict()
    with Timer("open-file-time") as timer:
        f = h5py.File(h5file, mode="r", **h5f_kwargs)
    bench_data[timer.name] = timer.elapsed()
    h5py._errors.unsilence_errors()  # Enable displaying full libhdf5 error stack
    num_dsets = 0
    with Timer("read-data-time") as timer:
        for s in chain.from_iterable(obj.values()):
            sig_dset = f[s]
            sig_dset[...]
            num_dsets += 1
            for dim in sig_dset.dims:
                for scale in dim.values():
                    scale[...]
                    num_dsets += 1

    # Collect page buffer cache stats only for a paged file...
    if (
        f.id.get_create_plist().get_file_space_strategy()[0]
        == h5py.h5f.FSPACE_STRATEGY_PAGE
    ):
        bench_data["pb-size"] = f.id.get_access_plist().get_page_buffer_size()[0]
        if bench_data["pb-size"] != 0:
            hit_rate = lambda page_stats: 100 * (  # noqa: E731
                page_stats.hits / (page_stats.accesses - page_stats.bypasses)
            )
            pb_stats = f.id.get_page_buffering_stats()
            bench_data["pb-meta-accesses"] = pb_stats.meta.accesses
            bench_data["pb-meta-hitrate"] = hit_rate(pb_stats.meta)
            bench_data["pb-meta-evicts"] = pb_stats.meta.evictions
            bench_data["pb-raw-accesses"] = pb_stats.raw.accesses
            bench_data["pb-raw-hitrate"] = hit_rate(pb_stats.raw)
            bench_data["pb-raw-evicts"] = pb_stats.raw.evictions

    f.close()

    bench_data[timer.name] = timer.elapsed()
    bench_data["wrkr-num-objs"] = len(obj)
    bench_data["mean-obj-time"] = timer.elapsed() / len(obj)
    bench_data["num-dsets"] = num_dsets
    bench_data["mean-dset-time"] = timer.elapsed() / num_dsets
    return bench_data


# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    cli = parse_cli()
    logging.basicConfig(
        level=cli.loglevel.upper(),
        stream=sys.stdout,
        format="%(name)s:%(levelname)s:%(funcName)s:%(message)s",
    )
    lggr.debug("Runtime options: %s", cli)

    cpus = cpu_count()

    # Run the benchmarks with different parameters...
    data = list()
    for rp in bench_params(
        pb_size=cli.pb,  # libhdf5 page buffer size
        num_workers=[1, 2, 4, 8, 16],  # number of Dask workers
        # shots=[None, 150, 300, 0],  # number of shots to read (0 means all)
        shots=[None, 0],  # number of shots to read (0 means all)
        signals=[None, 0],  # number of signals to read (0 means all)
        infile=cli.infiles,
    ):
        # Keep only cases of interest...
        if (rp.shots is None and rp.signals is None) or (
            rp.shots is not None and rp.signals is not None
        ):
            continue
        lggr.info("Benchmark run parameters: %s", rp)
        if rp.num_workers > cpus:
            lggr.warning(
                "Number of workers %d greater than CPU cores %d",
                rp.num_workers,
                cpu_count(),
            )

        h5py_kwargs = {"rdcc_nbytes": 8 * MiB, "page_buf_size": rp.pb_size}
        if rp.infile.startswith(("https://", "s3://")):
            h5py_kwargs["driver"] = "ros3"
            h5py_kwargs.update(get_s3_params(need_region=rp.infile.startswith("s3://")))
        lggr.debug("h5py.File settings: %s", h5py_kwargs)

        # Gather info about the datasets in the file for later...
        lggr.info("Gathering file content info...")
        with (
            Timer("gather-info") as gather,
            h5py.File(
                rp.infile,
                mode="r",
                **h5py_kwargs,
            ) as f,
        ):
            objs = gather_dset_info(f)
        lggr.info("Gathering file content time = %.4f seconds", gather.elapsed())

        if rp.shots is None and rp.signals is not None:
            # Re-arrange per-shot HDF5 dataset info into per-signal...
            signals = defaultdict(list)
            for s in chain.from_iterable(objs.values()):
                name_parts = Path(s).parts
                if "signals" in name_parts:
                    signals[name_parts[-1]].append(s)
            objs = signals
            del signals

        # Randomize and select the shots/signals to read...
        use_objs = list(objs.keys())
        np.random.shuffle(use_objs)
        if rp.shots is not None:
            obj_type = "shots"
            use_objs = use_objs[slice(None if rp.shots == 0 else rp.shots)]
        elif rp.signals is not None:
            obj_type = "signals"
            use_objs = use_objs[slice(None if rp.signals == 0 else rp.signals)]

        lggr.info(
            "Starting to read %d %s and their scales with %d worker(s)",
            len(use_objs),
            obj_type,
            rp.num_workers,
        )
        dask_client = Client(
            processes=True, n_workers=rp.num_workers, threads_per_worker=1
        )
        bench_futures = list()
        with Timer("total-runtime") as timer:
            for batch in batched(use_objs, (len(use_objs) // rp.num_workers) + 1):
                bf = dask_client.submit(
                    reader,
                    rp.infile,
                    dict((_, objs[_]) for _ in batch),
                    **h5py_kwargs,
                )
                bench_futures.append(bf)
            bench_data = dask_client.gather(bench_futures)
        dask_client.close()
        lggr.debug("Worker benchmark results: %s", bench_data)
        lggr.info("Benchmark case runtime = %.4f seconds", timer.elapsed())
        for wrkr, _ in enumerate(bench_data):
            _.update(
                {
                    "worker#": wrkr,
                    "num-workers": rp.num_workers,
                    "file": rp.infile,
                    "obj-type": obj_type,
                    "tot-num-obj": len(use_objs),
                    "gather-time": gather.elapsed(),
                    "tot-runtime": timer.elapsed(),
                }
            )
        data.extend(bench_data)
        if cli.to_csv:
            lggr.debug("Checkpoint benchmark data so far...")
            Path(cli.to_csv).with_suffix(".checkpoint.json").write_text(
                json.dumps(data, indent=None)
            )

    df = pd.DataFrame.from_records(data)
    if cli.to_csv:
        lggr.info("Benchmark results saved to file: %s", cli.to_csv)
        df.to_csv(cli.to_csv, index=False)
        Path(cli.to_csv).with_suffix(".checkpoint.json").unlink(missing_ok=True)
    else:
        pd.set_option("display.max_columns", 16)
        pd.set_option("display.max_colwidth", 200)
        print(df)
        lggr.info("Benchmark results not saved.")
