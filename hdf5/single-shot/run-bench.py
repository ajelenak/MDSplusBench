import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict, namedtuple
from configparser import ConfigParser
from dataclasses import dataclass, field
from itertools import batched, product
from multiprocessing import cpu_count
from pathlib import Path
import fsspec
from typing import Generator, Union

import h5py
import numpy as np
import pandas as pd
from dask.distributed import Client

lggr = logging.getLogger("mdsplusml-bench")

MiB = 1024 * 1024


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


def parse_cli() -> argparse.Namespace:
    """Process command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MDSplusML h5py benchmark script for single-shot HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infolder",
        help="A folder with single-shot HDF5 files.",
        type=str,
        metavar="FOLDER",
    )
    # parser.add_argument(
    #     "--pb",
    #     help="Page buffer size in bytes. Multiple values allowed.",
    #     nargs="+",
    #     type=int,
    #     default=[0],
    # )
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


def gather_dset_info(h5f: h5py.File) -> dict[int, dict[str, Union[str, list[str]]]]:
    """Discover all signal datasets in the HDF5 file.

    Every dataset not a dimension scale and with a `shot` attribute is assumed
    to be holding signal data.
    """
    shots = defaultdict(list)
    fname = h5f.filename

    def dset_info(name: str, h5obj: h5py.HLObject):
        if isinstance(h5obj, h5py.Dataset):
            if not h5obj.is_scale:
                shots[h5obj.attrs["shot"]].append(h5obj.name)

    h5f.visititems(dset_info)
    keys = list(shots.keys())
    if len(keys) != 1:
        raise ValueError(f"File {h5f.filename} holds more than one shot")
    return {keys[0]: {"h5path": next(iter(shots.values())), "fname": fname}}


def reader(obj_id: str, obj: dict, worker: int, **h5f_kwargs) -> dict[str, float]:
    """Read data for supplied selection of shots/signals in the given HDF5 file(s)."""
    h5py._errors.unsilence_errors()  # enable displaying full libhdf5 error stack
    bench_data = dict()
    # with Timer("open-file-time") as timer:
    #     f = h5py.File(h5file, mode="r", **h5f_kwargs)
    # bench_data[timer.name] = timer.elapsed()
    num_dsets = 0
    with Timer("open+read-data-time") as timer:
        for fname, signals in obj.items():
            with h5py.File(fname, mode="r", **h5f_kwargs) as f:
                for s in signals:
                    sig_dset = f[s]
                    sig_dset[...]
                    num_dsets += 1
                    for dim in sig_dset.dims:
                        for scale in dim.values():
                            scale[...]
                            num_dsets += 1

    # Collect page buffer cache stats only for a paged file...
    # if (
    #     f.id.get_create_plist().get_file_space_strategy()[0]
    #     == h5py.h5f.FSPACE_STRATEGY_PAGE
    # ):
    #     bench_data["pb-size"] = f.id.get_access_plist().get_page_buffer_size()[0]
    #     if bench_data["pb-size"] != 0:
    #         hit_rate = lambda page_stats: 100 * (  # noqa: E731
    #             page_stats.hits / (page_stats.accesses - page_stats.bypasses)
    #         )
    #         pb_stats = f.id.get_page_buffering_stats()
    #         bench_data["pb-meta-accesses"] = pb_stats.meta.accesses
    #         bench_data["pb-meta-hitrate"] = hit_rate(pb_stats.meta)
    #         bench_data["pb-meta-evicts"] = pb_stats.meta.evictions
    #         bench_data["pb-raw-accesses"] = pb_stats.raw.accesses
    #         bench_data["pb-raw-hitrate"] = hit_rate(pb_stats.raw)
    #         bench_data["pb-raw-evicts"] = pb_stats.raw.evictions

    bench_data["worker#"] = worker
    bench_data["obj-id"] = obj_id
    bench_data[timer.name] = timer.elapsed()
    bench_data["wrkr-num-objs"] = len(obj)
    bench_data["mean-obj-time"] = timer.elapsed() / len(obj)
    bench_data["num-dsets"] = num_dsets
    bench_data["mean-dset-time"] = timer.elapsed() / num_dsets
    bench_data["pb-size"] = h5f_kwargs["page_buf_size"]
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

    if cli.infolder.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        pb_size = [256 * MiB]  # page buffer cache sizes
    else:
        fs = fsspec.filesystem("file")
        pb_size = [0, 256 * MiB]  # page buffer cache sizes
    shot_files = sorted(fs.glob(cli.infolder + "/*.hdf5"))
    lggr.info("Found %d shot files at %s", len(shot_files), cli.infolder)
    if len(shot_files) == 0:
        raise SystemExit(f"No shot files found in {cli.infolder}")
    else:
        lggr.debug("List of shot files: %r", shot_files)

    cpus = cpu_count()
    lggr.debug("%d CPUs reported for the system", cpus)

    # Figure out h5py file open settings...
    if "s3" in fs.protocol:
        h5py_kwargs = {"driver": "ros3", "page_buf_size": 64 * MiB}
        h5py_kwargs.update(get_s3_params(need_region=True))

        # fs.glob() output currently does not have the s3 schema part so add it here...
        shot_files = ["s3://" + _ for _ in shot_files]
    else:
        h5py_kwargs = dict()

    # Gather info about the content in the files...
    lggr.info(f"Gathering file content info from {len(shot_files)}...")
    shots = dict()
    with Timer("gather-info") as gather:
        for _ in shot_files:
            lggr.debug("Gathering content info from %s", _)
            with h5py.File(_, mode="r", **h5py_kwargs) as f:
                objs = gather_dset_info(f)
            shots.update(objs)
    lggr.info("Gathering file content time = %.4f seconds", gather.elapsed())

    # Re-arrange per-shot info into per-signal...
    signals = defaultdict(list)
    # for s in chain.from_iterable(all_shots_info.values()):
    for _ in shots.values():
        fname = _["fname"]
        for s in _["h5path"]:
            name_parts = Path(s).parts
            if "signals" in name_parts:  # "signals" must be in the HDF5 path
                signals[name_parts[-1]].append({"h5path": s, "fname": fname})

    # Run the benchmarks with different parameters...
    data = list()
    for rp in bench_params(
        pb_size=pb_size,  # libhdf5 page buffer size
        num_workers=[1, 2, 4, 8, 16, 24, 32, 48, 64],  # number of Dask workers
        shots=[None, 0],  # number of shots to read (0 means all)
        signals=[None, 0],  # number of signals to read (0 means all)
    ):
        # Keep only cases of interest...
        if (rp.shots is None and rp.signals is None) or (
            rp.shots is not None and rp.signals is not None
        ):
            continue
        lggr.info("Benchmark run parameters: %s", rp)
        if rp.num_workers > cpus:
            lggr.warning(
                "Number of workers %d greater than reported CPUs %d",
                rp.num_workers,
                cpus,
            )

        h5py_kwargs.update({"rdcc_nbytes": 8 * MiB, "page_buf_size": rp.pb_size})
        lggr.debug("h5py.File kwargs: %s", h5py_kwargs)

        if rp.shots is None and rp.signals is not None:
            objs = signals
            obj_type = "signals"
            lggr.debug("Will read shot files by signal")
            if rp.num_workers < 8:
                lggr.info(
                    "Skipping this benchmark, too little workers: %d", rp.num_workers
                )
                continue
        else:
            objs = shots
            obj_type = "shots"
            lggr.debug("Will read shot files by shot")

        # Randomize and select the shots/signals to read...
        use_objs = list(objs.keys())
        np.random.shuffle(use_objs)

        dask_client = Client(
            processes=True, n_workers=rp.num_workers, threads_per_worker=1
        )
        bench_futures = list()
        with Timer("total-runtime") as timer:
            if obj_type == "shots":
                what = objs[
                    use_objs[0]
                ]  # only use the first shot from the randomized list
                lggr.info(
                    "Reading shot #%d data with %d signals and their scales with %d worker(s)",
                    use_objs[0],
                    len(what["h5path"]),
                    rp.num_workers,
                )
                worker = 0
                for batch in batched(
                    what["h5path"], (len(what["h5path"]) // rp.num_workers) + 1
                ):
                    worker += 1
                    bf = dask_client.submit(
                        reader,
                        str(use_objs[0]),
                        {what["fname"]: batch},
                        worker,
                        **h5py_kwargs,
                    )
                    bench_futures.append(bf)

            elif obj_type == "signals":
                lggr.info(
                    "Reading %d signals and their scales from %d files with %d worker(s)",
                    len(use_objs),
                    len(shot_files),
                    rp.num_workers,
                )
                for sig, what in objs.items():
                    lggr.debug("Reading signal %s data", sig)
                    worker = 0
                    for batch in batched(what, (len(what) // rp.num_workers) + 1):
                        worker += 1
                        bf = dask_client.submit(
                            reader,
                            sig,
                            dict((_["fname"], [_["h5path"]]) for _ in batch),
                            worker,
                            **h5py_kwargs,
                        )
                        bench_futures.append(bf)

            bench_data = dask_client.gather(bench_futures)

        dask_client.close()
        lggr.info("Benchmark case runtime = %.4f seconds", timer.elapsed())
        lggr.debug("Worker benchmark results: %s", bench_data)
        for wrkr, _ in enumerate(bench_data):
            _.update(
                {
                    "num-workers": rp.num_workers,
                    "obj-type": obj_type,
                    "tot-num-obj": len(use_objs),
                    timer.name: timer.elapsed(),
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
