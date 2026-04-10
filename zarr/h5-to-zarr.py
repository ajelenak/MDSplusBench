import argparse
from configparser import ConfigParser
import logging
import os
from pathlib import Path
import shutil
import sys
import tomllib
from urllib.parse import urlparse

from dask.distributed import Client, LocalCluster, as_completed
import fsspec
import h5py
import numcodecs
import numpy as np
import obstore
from obstore.store import LocalStore, S3Store
import zarr
from zarr.codecs import GZip
from zarr.storage import ObjectStore


if h5py.version.hdf5_version_tuple < (2, 0, 0):
    raise RuntimeError("Must use libhdf5 2.0.0 or later")
if not h5py.h5.get_config().ros3:
    raise RuntimeError("Must use libhdf5 with ros3 driver enabled")

# Configure logging...
logging.basicConfig(
    format="%(levelname)-8s | %(name)-12.12s | %(message)s", level=logging.INFO
)
lggr = logging.getLogger(Path(__file__).stem)


def parse_cli() -> argparse.Namespace:
    """Deal with command-line arguments. Defaults are suppressed to detect actual user input."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 files to Zarr using a strict TOML config with allowed CLI overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the TOML configuration file.",
        default="./h5-to-zarr.toml",
    )

    # CLI-exclusive flags...
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run to test configurations and file discovery without converting",
    )

    # Dask worker argument. Default is 90% of reported CPUs (minimum 1).
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int((os.cpu_count() or 1) * 0.9)),
        help="Number of Dask workers to spin up. One process per worker, no threads.",
    )

    # Suppressed overrides...
    parser.add_argument(
        "--input-uri", type=str, default=argparse.SUPPRESS, help="Override input URI"
    )
    parser.add_argument(
        "--output-uri", type=str, default=argparse.SUPPRESS, help="Override output URI"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--zarr-version", type=int, choices=[2, 3], default=argparse.SUPPRESS
    )

    # Boolean overrides (Generates both --flag and --no-flag)...
    parser.add_argument(
        "--use-shards", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--consolidate-metadata",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    return parser.parse_args()


def load_and_merge_config(cli_args: argparse.Namespace) -> argparse.Namespace:
    """Reads TOML, applies permitted CLI overrides, and returns a unified config namespace."""
    lggr.info(f"Reading configuration from '{cli_args.config}'")
    with open(cli_args.config, "rb") as f:
        toml_data = tomllib.load(f)

    # Map the CLI argument names to their [section] and [key] in the TOML file...
    param_map = {
        "input_uri": ("paths", "input_uri"),
        "output_uri": ("paths", "output_uri"),
        "zarr_version": ("zarr", "version"),
        "use_shards": ("zarr", "use_shards"),
        "consolidate_metadata": ("zarr", "consolidate_metadata"),
        "loglevel": ("logging", "level"),
    }

    final_config = argparse.Namespace()

    for cli_key, (toml_sec, toml_key) in param_map.items():
        # Fetch the entry; default to an empty dict if malformed...
        entry = toml_data.get(toml_sec, {}).get(toml_key)
        if entry is None or "value" not in entry:
            raise ValueError(
                f"Invalid TOML config: Missing '{toml_sec}.{toml_key}.value'"
            )

        val = entry["value"]
        allow_override = entry.get("allow_override", False)

        # 3-Way Check: Did the user type this flag?
        if hasattr(cli_args, cli_key):
            cli_val = getattr(cli_args, cli_key)
            if not allow_override:
                raise PermissionError(
                    f"Cannot override '{toml_key}'. Check your TOML config."
                )

            lggr.info(f"Overriding config '{toml_key}' -> '{cli_val}'")
            val = cli_val

        # Set the resolved value on our final config namespace...
        setattr(final_config, cli_key, val)

    # Pass through CLI-exclusive flags...
    setattr(final_config, "dry_run", getattr(cli_args, "dry_run", False))
    setattr(final_config, "workers", getattr(cli_args, "workers"))

    return final_config


def get_file_list(uri: str) -> list[str]:
    """
    Returns a list of HDF5 files to process using fsspec.
    Handles both local paths and s3:// URIs transparently.
    """
    lggr.info(f"Looking for files in: '{uri}'")

    # fsspec.url_to_fs determines the protocol (file, s3, etc.) and returns the
    # filesystem object...
    fs, path = fsspec.core.url_to_fs(uri)

    # fs.find is a powerful recursive search (like 'find' in bash)
    # detail=False returns just the list of files...
    all_files = fs.find(path, detail=False, maxdepth=1)

    target_files = []
    for f in all_files:
        if f.endswith((".h5", ".hdf5")):
            # fsspec usually returns path relative to the fs root (e.g., 'bucket/key').
            # We need to reconstruct the protocol for downstream usage if it's S3...
            if "file" in fs.protocol:
                target_files.append(f)
            elif "s3" in fs.protocol:
                target_files.append(f"s3://{f}")
            else:
                raise ValueError(f"Unexpected fsspec protocol: {fs.protocol}")

    lggr.info(f"Found {len(target_files)} HDF5 files in '{uri}' to convert.")
    return target_files


def get_h5_handle(uri: str) -> h5py.File:
    """
    Opens HDF5 file. Uses ros3 driver if path is S3 URI.
    """
    if uri.startswith("s3://"):
        lggr.debug(f"Opening S3 HDF5 via ros3: '{uri}'")
        try:
            return h5py.File(uri, mode="r", driver="ros3")
        except Exception as e:
            lggr.error("Failed to open S3 file with ros3 driver. ")
            raise e
    else:
        lggr.debug(f"Opening local HDF5: '{uri}'")
        return h5py.File(uri, mode="r")


def get_s3_config() -> dict[str, str]:
    """Provide S3 connection parameters.

    `obstore` is very picky about the way how AWS credentials are procured.
    """
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
    s3p["access_key_id"] = os.getenv(
        "AWS_ACCESS_KEY_ID", creds.get(profile, "aws_access_key_id", fallback="")
    )
    s3p["secret_access_key"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY",
        creds.get(profile, "aws_secret_access_key", fallback=""),
    )
    s3p["region"] = os.getenv("AWS_REGION", config.get(profile, "region"))

    return s3p


def get_io_store(uri: str) -> S3Store | LocalStore:
    """
    Return appropriate `obstore.store` object for the given URI (local or S3).
    """
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return S3Store(bucket, prefix=prefix, config=get_s3_config())
    else:
        return LocalStore(uri)


def del_s3_zstore(io_store: S3Store) -> int:
    """Remove S3 objects belonging to a Zarr store.

    Return number of deleted S3 objects.
    """
    del_items = [_["path"] for _ in obstore.list(io_store, prefix=None).collect()]
    obstore.delete(io_store, del_items)
    return len(del_items)


def determine_dims(h5_ds: h5py.Dataset) -> list[str]:
    """
    Extracts HDF5 dimension scale names for xarray compatibility via
    `_ARRAY_DIMENSIONS` of Zarr arrays.
    """
    dims = []
    for i in range(h5_ds.ndim):
        dim_obj = h5_ds.dims[i]
        label = dim_obj.label

        if len(dim_obj) > 0:
            scale_name = Path(dim_obj[0].name).name
            dims.append(scale_name)
        elif label:
            dims.append(label)
        else:
            dims.append(f"dim_{i}")
    return dims


def copy_attributes(
    source: h5py.Group | h5py.Dataset, dest: zarr.Group | zarr.Array
) -> None:
    """Copies HDF5 attributes to Zarr attributes."""
    # Reserved (special meaning) attributes...
    HIDDEN_ATTRS = {
        "REFERENCE_LIST",
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_nc3_strict",
        "_NCProperties",
        "_FillValue",
    }
    lggr.debug(f"Converting attributes of '{source.name}'")
    upd = dict()
    for n, v in source.attrs.items():
        if n in HIDDEN_ATTRS:
            continue

        # Fix some attribute values to avoid JSON encoding exceptions...
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="strict")
        elif isinstance(v, (np.ndarray, np.number, np.bool_)):
            if v.dtype.kind == "S":
                v = v.astype(str)
            elif v.size == 1:
                v = v.flatten()[0]
                if isinstance(v, (np.ndarray, np.number, np.bool_)):
                    v = v.tolist()
            else:
                v = v.tolist()
        elif isinstance(v, h5py._hl.base.Empty):
            v = ""
        if v == "DIMENSION_SCALE":
            continue

        try:
            if isinstance(v, (str, int, float)):
                upd[n] = v
            elif isinstance(v, (tuple, set, list)) and all(
                isinstance(_, (str, int, float)) for _ in v
            ):
                upd[n] = list(v)
            else:
                upd[n] = str(v)
        except TypeError:
            raise TypeError(
                f"Error transferring attr: '{n}@{source.name}' = '{v}' ({type(v)})"
            )
        dest.attrs.update(upd)


def process_dataset(
    name: str, h5dset: h5py.Dataset, zarr_group: zarr.Group, config: argparse.Namespace
) -> None:
    """
    Reads HDF5 dataset and creates equivalent Zarr array.
    """
    lggr.debug(f"Processing HDF5 dataset: '{h5dset.name}'")
    comp = h5dset.compression
    if comp is not None and comp not in ["gzip", "deflate"]:
        raise ValueError(
            f"Unsupported compression '{comp}' in dataset '{h5dset.name}'. Only 'gzip' is allowed."
        )

    # Determine chunks, shards, and compressor based on target Zarr version...
    shards = None
    filters = None

    if h5dset.chunks is None:
        chunks = h5dset.shape
        compressor = None
    else:
        chunks = h5dset.chunks
        level = h5dset.compression_opts if h5dset.compression_opts is not None else 1

        if config.zarr_version == 3:
            compressor = GZip(level=level)
            if config.use_shards:
                num_chunks = h5dset.id.get_num_chunks()
                if num_chunks > 1:
                    expanse = (num_chunks,) * len(chunks)
                    shards = tuple(_[0] * _[1] for _ in zip(chunks, expanse))
        else:
            # Zarr v2 uses the standard numcodecs implementation
            compressor = numcodecs.GZip(level=level)

    # Create Zarr array...
    h5dtype = h5dset.dtype
    if h5dtype.kind == "O" and h5py.check_vlen_dtype(h5dtype):
        # Variable-length strings case...
        zdtype = str
        if config.zarr_version == 2:
            filters = [numcodecs.VLenUTF8()]
    else:
        zdtype = h5dtype

    # Structure keyword arguments specifically tailored for the target Zarr format API
    create_kwargs = {
        "name": name,
        "shape": h5dset.shape,
        "dtype": zdtype,
        "chunks": chunks,
        "overwrite": True,
    }

    if config.zarr_version == 3:
        create_kwargs["compressors"] = compressor
        create_kwargs["shards"] = shards
    else:
        create_kwargs["compressor"] = compressor
        if filters is not None:
            create_kwargs["filters"] = filters

    zarr_arr = zarr_group.create_array(**create_kwargs)

    # Transfer data...
    if h5dset.shape == ():
        zarr_arr[()] = h5dset[()]
    else:
        zarr_arr[:] = h5dset[:]

    # Attributes & dimension scales (xarray compatibility)...
    copy_attributes(h5dset, zarr_arr)
    if h5dset.ndim > 0:
        dim_names = determine_dims(h5dset)
        zarr_arr.attrs["_ARRAY_DIMENSIONS"] = dim_names


def process_group(
    h5_group: h5py.Group, zarr_group: zarr.Group, config: argparse.Namespace
) -> None:
    """
    Recursively traverses HDF5 group and populates Zarr group.
    """
    copy_attributes(h5_group, zarr_group)
    for name, obj in h5_group.items():
        if isinstance(obj, h5py.Dataset):
            process_dataset(name, obj, zarr_group, config)
        elif isinstance(obj, h5py.Group):
            lggr.debug(f"Processing HDF5 group: '{h5_group.name}'")
            sub_group = zarr_group.create_group(name)
            process_group(obj, sub_group, config)


def convert_h5_to_zarr(h5_uri: str, config: argparse.Namespace) -> tuple[str, bool]:
    """
    Modified to explicitly return the source URI and a success boolean to inform the Dask Scheduler.
    """
    filename = Path(h5_uri).name
    store_name = Path(filename).with_suffix(".zarr").name

    if config.output_uri.startswith("s3://"):
        zarr_store_uri = f"{config.output_uri.rstrip('/')}/{store_name}"
        iostore = get_io_store(zarr_store_uri)
        del_s3_zstore(iostore)
    else:
        zarr_store_uri = Path(config.output_uri).resolve() / store_name
        if zarr_store_uri.exists():
            if zarr_store_uri.is_file():
                lggr.warning(f"File '{zarr_store_uri}' exists, removing")
                zarr_store_uri.unlink()
            elif zarr_store_uri.is_dir():
                lggr.warning(f"Folder '{zarr_store_uri}' exists, removing")
                shutil.rmtree(zarr_store_uri)
        zarr_store_uri.mkdir(parents=True, exist_ok=True)
        zarr_store_uri = str(zarr_store_uri)
        iostore = get_io_store(zarr_store_uri)

    lggr.info(
        f"Worker converting '{h5_uri}' to '{zarr_store_uri}' (Zarr v{config.zarr_version})"
    )
    try:
        with get_h5_handle(h5_uri) as h5_file:
            zstore = ObjectStore(iostore)
            root = zarr.open_group(
                store=zstore, mode="w", zarr_format=config.zarr_version
            )
            process_group(h5_file, root, config)

        if config.consolidate_metadata:
            lggr.info(f"Consolidating metadata for '{zarr_store_uri}'")
            zarr.consolidate_metadata(store=zstore)

        lggr.info(f"Worker finished translating: '{zarr_store_uri}'")
        return h5_uri, True  # send strict confirmation to the Dask coordinator

    except Exception:
        lggr.exception(f"Worker failed to convert '{h5_uri}'")
        if zarr_store_uri.startswith("s3://"):
            del_s3_zstore(iostore)
        else:
            shutil.rmtree(zarr_store_uri)
        return h5_uri, False  # inform coordinator that task was not accomplished


def main():
    # Parse raw CLI arguments...
    raw_cli = parse_cli()
    if raw_cli.workers > os.cpu_count():
        raise ValueError(
            f"Too many Dask workers requested for this computer, max = {os.cpu_count()}"
        )

    # Merge with TOML and enforce override rules...
    try:
        config = load_and_merge_config(raw_cli)
    except Exception as e:
        lggr.critical(f"Configuration Error: {e}")
        sys.exit(1)

    lggr.setLevel(getattr(logging, config.loglevel.upper()))

    try:
        files = get_file_list(config.input_uri)
    except Exception as e:
        lggr.critical(f"Error listing files with fsspec: {e}")
        sys.exit(1)

    if not files:
        lggr.warning(f"No HDF5 files found in '{config.input_uri}'.")
        sys.exit(0)

    # Handle dry run and exit...
    if config.dry_run:
        lggr.info("DRY RUN ENABLED. No files will be modified or converted.")
        for f in files:
            filename = Path(f).resolve()
            store_name = filename.with_suffix(".zarr").name
            if config.output_uri.startswith("s3://"):
                zarr_store_uri = f"{config.output_uri.rstrip('/')}/{store_name}"
            else:
                zarr_store_uri = str(Path(config.output_uri).resolve() / store_name)
            lggr.info(
                f"[DRY RUN] Would convert '{str(filename)}' to '{zarr_store_uri}' (Zarr v{config.zarr_version})"
            )
        lggr.info("Dry run complete. Exiting before Dask cluster initialization.")
        sys.exit(0)

    lggr.info(
        f"Initializing Dask cluster with {config.workers} distinct workers (1 process per worker, NO threads)..."
    )

    MAX_RETRIES = 3
    retry_counts = {f: 0 for f in files}
    failed_files = []

    with LocalCluster(
        n_workers=config.workers, threads_per_worker=1, processes=True
    ) as cluster:
        with Client(cluster) as client:
            # Seed the cluster with the initial batch of file tasks
            future_to_file = {
                client.submit(convert_h5_to_zarr, f, config): f for f in files
            }

            # Create a dynamic queue iterator that listens for completed worker responses
            seq = as_completed(future_to_file.keys())

            for future in seq:
                f = future_to_file[future]

                try:
                    # Capture the strict confirmation tuple from the worker
                    h5_uri, success = future.result()

                    if success:
                        lggr.info(
                            f"[SUCCESS CONFIRMED] '{h5_uri}' has been safely converted."
                        )
                    else:
                        retry_counts[f] += 1
                        if retry_counts[f] <= MAX_RETRIES:
                            lggr.warning(
                                f"[WORKER FAILED] Task failed internally for '{f}'. Retry {retry_counts[f]}/{MAX_RETRIES}. Resubmitting..."
                            )
                            new_future = client.submit(convert_h5_to_zarr, f, config)
                            future_to_file[new_future] = f
                            seq.add(new_future)
                        else:
                            lggr.error(
                                f"[PERMANENT FAILURE] '{f}' failed {MAX_RETRIES} times. Skipping file."
                            )
                            failed_files.append(f)

                except Exception as e:
                    # In case the worker crashes before even returning the False flag
                    retry_counts[f] += 1
                    if retry_counts[f] <= MAX_RETRIES:
                        lggr.error(
                            f"[WORKER CRASH] Unhandled exception processing '{f}': {e}. Retry {retry_counts[f]}/{MAX_RETRIES}. Resubmitting..."
                        )
                        new_future = client.submit(convert_h5_to_zarr, f, config)
                        future_to_file[new_future] = f
                        seq.add(new_future)
                    else:
                        lggr.error(
                            f"[PERMANENT FAILURE] '{f}' crashed {MAX_RETRIES} times. Skipping file."
                        )
                        failed_files.append(f)

    # Final summary readout
    lggr.info("All pipeline tasks complete.")
    if failed_files:
        lggr.critical(
            f"WARNING: {len(failed_files)} files permanently failed to convert after {MAX_RETRIES} retries:"
        )
        for failed_f in failed_files:
            lggr.critical(f"  - {failed_f}")
    else:
        lggr.info("All files converted successfully.")


if __name__ == "__main__":
    main()
