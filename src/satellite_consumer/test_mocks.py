import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import patch

from botocore.session import Session
from moto.server import ThreadedMotoServer

if TYPE_CHECKING:
    from botocore.client import BaseClient as BotocoreClient


@contextlib.contextmanager
def mocks3() -> Generator[str]:
    server = ThreadedMotoServer()
    server.start()
    with patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_SECURITY_TOKEN": "test",
            "AWS_SESSION_TOKEN": "test",
            "AWS_ENDPOINT_URL": "http://localhost:5000",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        clear=True,
    ):
        s3_client: BotocoreClient = Session().create_client(
            service_name="s3",
            region_name="us-east-1",
        )
        s3_client.create_bucket(Bucket="test-bucket")
        try:
            yield "s3://test-bucket/"
        finally:
            s3_client.close()
            server.stop()
