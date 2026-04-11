import os
from configparser import ConfigParser
from pathlib import Path


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
