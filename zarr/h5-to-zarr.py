import argparse
from configparser import ConfigParser
import logging
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlparse

import fsspec
import h5py
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
    """Deal with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 files to Zarr v3 stores using fsspec and obstore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_uri",
        help="Folder or S3 bucket URI containing source HDF5 files (e.g., s3://bucket/data or ./data)",
        type=str,
    )
    parser.add_argument(
        "output_uri", help="Folder or S3 bucket URI for output Zarr stores", type=str
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level",
    )
    parser.add_argument(
        "-M",
        "--consolidate-metadata",
        action="store_true",
        help="Consolidate Zarr store metadata",
    )
    parser.add_argument(
        "-S",
        "--use-shards",
        action="store_true",
        help="Enable storing Zarr chunks into shards",
    )
    return parser.parse_args()


def get_file_list(uri: str) -> list[str]:
    """
    Returns a list of HDF5 files to process using fsspec.
    Handles both local paths and s3:// URIs transparently.
    """
    lggr.info(f"Listing files in: '{uri}'")

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
    # s3p["session_token"] = os.getenv(
    #     "AWS_SESSION_TOKEN",
    #     creds.get(profile, "aws_session_token", fallback=""),
    # )
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
    name: str, h5dset: h5py.Dataset, zarr_group: zarr.Group, **kwargs
) -> None:
    """
    Reads HDF5 dataset and creates equivalent Zarr array.
    """
    lggr.debug(f"Processing HDF5 dataset: '{h5dset.name}'")
    comp = h5dset.compression
    if comp is not None and comp not in ["gzip", "deflate"]:
        raise ValueError(
            f"Unsupported compression '{comp}' in dataset '{h5dset.name}'. "
            "Only 'gzip' (DEFLATE) is allowed."
        )

    # Determine chunks and compressor...
    shards = None
    if h5dset.chunks is None:
        chunks = h5dset.shape
        compressor = None
        lggr.debug(
            f"Dataset '{h5dset.name}' is contiguous. Using single chunk {chunks} and no compression."
        )
    else:
        chunks = h5dset.chunks
        level = h5dset.compression_opts if h5dset.compression_opts is not None else 1
        compressor = GZip(level=level)

        if kwargs.get("use_shards", False):
            num_chunks = h5dset.id.get_num_chunks()
            if num_chunks > 1:
                expanse = (num_chunks,) * len(chunks)  # placeholder for future
                shards = tuple(_[0] * _[1] for _ in zip(chunks, expanse))

    # Create Zarr array...
    h5dtype = h5dset.dtype
    if h5dtype.kind == "O" and h5py.check_vlen_dtype(h5dtype):
        # Variable-length strings case...
        zdtype = str
    else:
        zdtype = h5dtype
    zarr_arr = zarr_group.create_array(
        name=name,
        shape=h5dset.shape,
        dtype=zdtype,
        chunks=chunks,
        shards=shards,
        compressors=compressor,
        overwrite=True,
    )

    # Transfer data...
    if h5dset.shape == ():
        zarr_arr[()] = h5dset[()]  # scalar HDF5 dataset
    else:
        zarr_arr[:] = h5dset[:]

    # Attributes & dimension scales (xarray compatibility)...
    copy_attributes(h5dset, zarr_arr)
    if h5dset.ndim > 0:
        dim_names = determine_dims(h5dset)
        zarr_arr.attrs["_ARRAY_DIMENSIONS"] = dim_names


def process_group(h5_group: h5py.Group, zarr_group: zarr.Group, **kwargs) -> None:
    """
    Recursively traverses HDF5 group and populates Zarr group.
    """
    copy_attributes(h5_group, zarr_group)

    for name, obj in h5_group.items():
        if isinstance(obj, h5py.Dataset):
            process_dataset(name, obj, zarr_group, **kwargs)
        elif isinstance(obj, h5py.Group):
            lggr.debug(f"Processing HDF5 group: '{h5_group.name}'")
            sub_group = zarr_group.create_group(name)
            process_group(obj, sub_group, **kwargs)


def convert_h5_to_zarr(h5_uri: str, cli: argparse.Namespace):
    """
    Orchestrates the conversion of a single HDF5 file to a Zarr store.
    """
    # Extract filename depending on protocol...
    filename = Path(h5_uri).name
    store_name = Path(filename).with_suffix(".zarr").name
    if cli.output_uri.startswith("s3://"):
        # Strip any existing trailing slash, then safely add exactly one...
        zarr_store_uri = f"{cli.output_uri.rstrip('/')}/{store_name}"
        iostore = get_io_store(zarr_store_uri)
        del_s3_zstore(iostore)
    else:
        zarr_store_uri = Path(cli.output_uri).resolve() / store_name
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

    lggr.info(f"Converting '{h5_uri}' to '{zarr_store_uri}'")
    try:
        with get_h5_handle(h5_uri) as h5_file:
            zstore = ObjectStore(iostore)
            root = zarr.open_group(store=zstore, mode="w", zarr_format=3)
            process_group(h5_file, root, use_shards=cli.use_shards)

        lggr.info(f"Successfully converted: '{zarr_store_uri}'")

        if cli.consolidate_metadata:
            lggr.info(f"Consolidating metadata for '{zarr_store_uri}")
            zarr.consolidate_metadata(store=zstore)
    except Exception:
        lggr.exception(f"Failed to convert '{h5_uri}'")
        if zarr_store_uri.startswith("s3://"):
            lggr.warning(
                f"Due to above error, removing S3 objects related to Zarr store: '{zarr_store_uri}'"
            )
            del_s3_zstore(iostore)
        else:
            lggr.warning(f"Due to above error, removing '{zarr_store_uri}'")
            shutil.rmtree(zarr_store_uri)


def main():
    cli = parse_cli()

    lggr.setLevel(getattr(logging, cli.loglevel.upper()))

    try:
        files = get_file_list(cli.input_uri)
    except Exception as e:
        lggr.critical(f"Error listing files with fsspec: {e}")
        sys.exit(1)

    if not files:
        lggr.warning(f"No HDF5 files found in '{cli.input_uri}'.")
        sys.exit(0)

    for f in files:
        convert_h5_to_zarr(f, cli)

    lggr.info("All processing complete.")


if __name__ == "__main__":
    main()
