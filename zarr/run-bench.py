import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from time import sleep
from typing import Generator, Union

import dask.config
import numpy as np
import obstore as obs
import pandas as pd
from dask.distributed import Client, as_completed
from obstore.store import LocalStore, S3Store
from s3creds import get_s3_config
from zarr.storage import ObjectStore
import zarr

# Silence the warning globally for this script execution
warnings.filterwarnings(
    "ignore", ".*Numcodecs codecs are not in the Zarr version 3 specification.*"
)

# Increase timeout defaults to avoid scheduler-worker comms issues...
dask.config.set(
    {
        "distributed.comm.timeouts.connect": "120s",
        "distributed.comm.timeouts.tcp": "120s",
        "distributed.scheduler.worker-ttl": "120s",
        "distributed.comm.retry.count": 15,
    }
)

lggr = logging.getLogger("mdsplusml-bench-zarr")


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


def parse_cli() -> argparse.Namespace:
    """Process command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MDSplusML benchmark script for single-shot Zarr stores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infolder",
        help="A folder with single-shot Zarr stores.",
        type=str,
        metavar="FOLDER",
    )
    parser.add_argument(
        "--read",
        help="What data to read from the stores",
        type=str,
        nargs="+",
        choices=["shots", "signals"],
        default=["shots", "signals"],
    )
    parser.add_argument(
        "--min-workers",
        help="Minimal number of Dask workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-workers",
        help="Maximum number of Dask workers",
        type=int,
        default=180,
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


def batch_tasks(num_batches: int, lst: list[dict], mode: str = "equal_effort"):
    """Divvy up items in batches based on the mode. Two modes possible:
    `equal_effort` and `round_robin`."""
    if num_batches <= 0:
        raise ValueError(f"Invalid value for number of batches: {num_batches}")
    actual_batches = min(num_batches, len(lst))
    if mode == "round_robin":
        return [lst[i::actual_batches] for i in range(actual_batches)]
    elif mode == "equal_effort":
        chunk_size = (len(lst) + actual_batches - 1) // actual_batches
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    raise ValueError(f'Unknown mode: "{mode}"')


def gather_dset_info(
    zroot: zarr.Group, fname: str
) -> dict[int, dict[str, Union[str, list[str]]]]:
    """Discover all signal datasets in the Zarr store starting from its root group.

    Every array with a `shot` attribute is assumed to be holding signal data.
    """
    shots = defaultdict(list)

    for name, zobj in zroot.members(max_depth=None):
        if isinstance(zobj, zarr.Array) and ("shot" in zobj.attrs):
            shots[zobj.attrs["shot"]].append(name)

    keys = list(shots.keys())
    if len(keys) != 1:
        raise ValueError(f"Store {fname} holds more than one shot (or no shots)")
    return {keys[0]: {"zarrpath": next(iter(shots.values())), "fname": fname}}


def reader(obj_id: str, obj: dict, worker: int, **zarr_kwargs) -> dict[str, float]:
    """Read data for supplied selection of shots/signals in the given Zarr store(s)."""
    zarr_kwargs.pop("rdcc_nbytes", None)

    bench_data = dict()
    open_times = list()
    read_times = 0
    num_files = num_dsets = 0
    for fname, signals in obj.items():
        with Timer("open-file-time") as timer:
            if fname.startswith("s3://"):
                bucket = fname.split("/")[2]
                path = "/".join(fname.split("/")[3:])
                store = ObjectStore(S3Store(bucket, config=get_s3_config()))
                f = zarr.open_group(store=store, path=path, mode="r", **zarr_kwargs)
            else:
                store = ObjectStore(LocalStore(fname))
                f = zarr.open_group(store=store, mode="r", **zarr_kwargs)
            num_files += 1
        open_times.append(timer.elapsed())
        with Timer("read-data-time") as timer:
            for s in signals:
                sig_dset = f[s]
                sig_dset[...]
                num_dsets += 1

                # Read all Zarr arrays listed in `_ARRAY_DIMENSIONS` (the
                # equivalent of HDF5 dimension scales for a signal dataset). The
                # dim coordinate arrays live in the signal's enclosing shot
                # group -- `shots/<SHOT_ID>/<dim_name>`.
                parts = s.split("/")
                if "signals" in parts:
                    shot_group = f["/".join(parts[: parts.index("signals")])]
                    for dim_name in sig_dset.attrs.get("_ARRAY_DIMENSIONS", []):
                        if dim_name in shot_group:
                            shot_group[dim_name][...]
                            num_dsets += 1
        read_times += timer.elapsed()

    bench_data["median-open-file-time"] = np.median(open_times)
    bench_data["num-open-files"] = num_files
    bench_data["worker#"] = worker
    bench_data["obj-id"] = obj_id
    bench_data["read-data-time"] = read_times
    bench_data["num-objs"] = len(obj)
    bench_data["num-dsets"] = num_dsets

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
        bucket = cli.infolder.split("/")[2]
        obs_store = S3Store(bucket, config=get_s3_config())
        prefix = "/".join(cli.infolder.split("/")[3:])
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        res = obs.list_with_delimiter(obs_store, prefix=prefix)
        prefixes = res.get("common_prefixes", [])
        shot_files = sorted(
            [
                f"s3://{bucket}/{p.rstrip('/')}"
                for p in prefixes
                if p.rstrip("/").endswith(".zarr")
            ]
        )
    else:
        shot_files = sorted(
            [str(p.resolve()) for p in Path(cli.infolder).glob("*.zarr")]
        )

    lggr.info("Found %d shot stores at %s", len(shot_files), cli.infolder)
    if len(shot_files) == 0:
        raise SystemExit(f"No shot stores found in {cli.infolder}")
    else:
        lggr.debug("List of shot stores: %r", shot_files)

    cpus = cpu_count()
    lggr.debug("The number of CPUs reported: %d", cpus)

    # Specific zarr config on open...
    zarr_kwargs = {"use_consolidated": True}

    # Gather info about the content in the stores...
    lggr.info(f"Gathering store content info from {len(shot_files)}...")
    shots = dict()
    with Timer("gather-info") as gather:
        for _ in shot_files:
            lggr.debug("Gathering content info from %s", _)
            if _.startswith("s3://"):
                bucket = _.split("/")[2]
                path = "/".join(_.split("/")[3:])
                store = ObjectStore(S3Store(bucket, config=get_s3_config()))
                root = zarr.open_group(store=store, path=path, mode="r", **zarr_kwargs)
            else:
                store = ObjectStore(LocalStore(_))
                root = zarr.open_group(store=store, mode="r", **zarr_kwargs)

            objs = gather_dset_info(root, _)
            shots.update(objs)
    lggr.info("Gathering store content time = %.4f seconds", gather.elapsed())

    # Re-arrange per-shot info into per-signal...
    signals = defaultdict(list)
    for _ in shots.values():
        fname = _["fname"]
        for s in _["zarrpath"]:
            name_parts = Path(s).parts
            try:
                # "signals" must be in the Zarr path
                signals_index = name_parts.index("signals")
            except ValueError:
                continue

            # For signal identifier across stores use only its latter part of the
            # Zarr array's path without the shot number.
            signals["/".join(name_parts[slice(signals_index + 1, None)])].append(
                {"zarrpath": s, "fname": fname}
            )

    # Run the benchmarks with different parameters...
    bench_data = list()
    prev_num_workers = {"shots": -1, "signals": -1}
    for rp in bench_params(
        num_workers=[  # number of Dask workers
            1,
            2,
            4,
            6,
            8,
            12,
            16,
            24,
            32,
            40,
            48,
            64,
            80,
            96,
            112,
            128,
            140,
            160,
            180,
        ],
        shots=[None, 0],  # number of shots to read (0 means all)
        signals=[None, 0],  # number of signals to read (0 means all)
    ):
        # Keep only cases of interest...
        if (rp.shots is None and rp.signals is None) or (
            rp.shots is not None and rp.signals is not None
        ):
            continue
        lggr.info("Benchmark run parameters: %s", rp)
        if not (cli.min_workers <= rp.num_workers <= cli.max_workers):
            lggr.info(
                "Skipping this benchmark, number of workers %d outside specified range [%d, %d]",
                rp.num_workers,
                cli.min_workers,
                cli.max_workers,
            )
            continue
        if rp.num_workers > cpus:
            lggr.warning(
                "Number of workers %d greater than reported CPUs %d",
                rp.num_workers,
                cpus,
            )

        if rp.shots is None and rp.signals is not None:
            objs = signals
            obj_type = "signals"
        else:
            objs = shots
            obj_type = "shots"

        if obj_type not in cli.read:
            lggr.info("Skipping since reading of %s not requested", obj_type)
            continue

        # Randomize and select the shots/signals to read...
        use_objs = list(objs.keys())
        np.random.shuffle(use_objs)

        # Run the benchmark cases...
        if obj_type == "shots":
            # Given the max num. workers and all the shot stores, break down the
            # work in tasks and determine actual num. workers based on the
            # batching mode.
            tasks = batch_tasks(rp.num_workers, use_objs, mode="equal_effort")
            num_workers = len(tasks)
            if num_workers == prev_num_workers[obj_type]:
                lggr.warning(
                    f"Actual number of workers same as previous case, skipping: {num_workers}"
                )
                continue
            else:
                prev_num_workers[obj_type] = num_workers

            lggr.info(
                "Reading from %d shot stores with %d signals and their scales using %d worker(s)",
                len(use_objs),
                len(signals),
                num_workers,
            )

            # Start Dask client and LocalCluster, wait 2 secs...
            dask_client = Client(
                processes=True, n_workers=num_workers, threads_per_worker=1
            )
            sleep(2)

            worker = 1
            with Timer("total-runtime") as timer:
                bench_futures = list()
                for shot_ids in tasks:
                    bf = dask_client.submit(
                        reader,
                        "shot files",
                        dict((objs[_]["fname"], objs[_]["zarrpath"]) for _ in shot_ids),
                        worker,
                        **zarr_kwargs,
                    )
                    bench_futures.append(bf)
                    worker += 1

                # Collect benchmark results from the Dask workers...
                futures_results = [_.result() for _ in as_completed(bench_futures)]

        elif obj_type == "signals":
            # Given the max num. workers and all the signals, break down the
            # work in tasks per signal, and determine actual num. workers based
            # on the batching mode.
            tasks = dict()
            num_workers = 1
            for sig in use_objs:
                tasks[sig] = batch_tasks(rp.num_workers, objs[sig], mode="equal_effort")
                num_workers = max(num_workers, len(tasks[sig]))
            if num_workers == prev_num_workers[obj_type]:
                lggr.warning(
                    f"Actual number of workers same as previous case, skipping: {num_workers}"
                )
                continue
            else:
                prev_num_workers[obj_type] = num_workers

            lggr.info(
                "Reading %d signals and their scales from %d stores using %d worker(s)",
                len(use_objs),
                len(shot_files),
                num_workers,
            )

            # Start Dask client and LocalCluster, wait 2 secs...
            dask_client = Client(
                processes=True, n_workers=num_workers, threads_per_worker=1
            )
            sleep(2)

            with Timer("total-runtime") as timer:
                futures_results = list()
                for sig in tasks.keys():
                    lggr.debug(
                        "Reading signal %s data using %d workers", sig, num_workers
                    )
                    bench_futures = list()
                    worker = 1
                    for batch in tasks[sig]:
                        bf = dask_client.submit(
                            reader,
                            sig,
                            dict((_["fname"], [_["zarrpath"]]) for _ in batch),
                            worker,
                            **zarr_kwargs,
                        )
                        bench_futures.append(bf)
                        worker += 1

                    # Collect benchmark results from the Dask workers...
                    futures_results.extend(
                        [_.result() for _ in as_completed(bench_futures)]
                    )

        sleep(2)
        dask_client.close()

        lggr.info("Benchmark case runtime = %.4f seconds", timer.elapsed())
        lggr.debug("Sample worker benchmark results: %s", futures_results[-1])
        for _ in futures_results:
            _.update(
                {
                    "num-workers": num_workers,
                    "obj-type": obj_type,
                    "tot-num-obj": len(use_objs),
                    timer.name: timer.elapsed(),
                }
            )
        bench_data.extend(futures_results)
        if cli.to_csv:
            lggr.debug("Checkpoint benchmark data so far...")
            Path(cli.to_csv).with_suffix(".checkpoint.json").write_text(
                json.dumps(bench_data, indent=None)
            )

    df = pd.DataFrame.from_records(bench_data)
    if cli.to_csv:
        lggr.info("Benchmark results saved to file: %s", cli.to_csv)
        df.to_csv(cli.to_csv, index=False)
        Path(cli.to_csv).with_suffix(".checkpoint.json").unlink(missing_ok=True)
    else:
        pd.set_option("display.max_columns", 16)
        pd.set_option("display.max_colwidth", 200)
        print(df)
        lggr.info("Benchmark results not saved.")
