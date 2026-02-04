"""Patch for eumdac's _request function to make arguments configurable.

This patch is required as the eumdac API (as of v3.1.1) does not allow the request call to be
configured by the user. This works around the limitation.
"""

import logging
from collections.abc import Callable
from typing import Literal

import requests
from eumdac.logging import logger
from eumdac.request import RequestError, _pretty_print, _should_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Silence retry warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def construct_patched_request_function(
    max_retries: int,
    backoff_factor: float,
    timeout: int,
) -> Callable[..., requests.Response]:
    """Generates a configurable patch for eumdac's internal request handler."""

    def _request(
        method: Literal["get", "post", "patch", "put", "delete"],
        url: str,
        max_retries: int = max_retries,
        backoff_factor: float = backoff_factor,
        **kwargs: object,
    ) -> requests.Response:
        """Patch the `eumdac.request._request()` function to add timeout and remove verbose logs.

        This was forked from eumdac v3.1.1.

        Args:
            method: HTTP request method to use in the request.
            url: *str*
            URL to make the request to.
            max_retries: Max number of retries to perform if the request fails.
            backoff_factor: Backoff factor to apply between attempts.
            **kwargs: Extra arguments to pass to the request, refer to the requests library
                documentation for a list of possible arguments.
        """
        # ------------------------------
        # This block is modified from the original function

        # Set a default timeout
        kwargs.setdefault("timeout", timeout)

        # Replace eumdac's RetryAndLog class to get rid of verbose logs
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "PATCH"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)

        # ------------------------------
        # The rest of the code below is unchanged from `eumdac.request._request()``

        session = requests.Session()

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        response = requests.Response()
        try:
            while True:
                if hasattr(session, method):
                    logger.debug(_pretty_print(method, url, kwargs))
                    response = getattr(session, method.lower())(url, **kwargs)
                    if _should_retry(response):
                        continue
                else:
                    raise RequestError(f"Operation not supported: {method}")
                break
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Received unexpected response: {e}")
        except requests.exceptions.RetryError:
            raise RequestError(  # noqa: B904
                f"Maximum retries ({max_retries}) reached for {method.capitalize()} {url}"
            )

        return response

    return _request
