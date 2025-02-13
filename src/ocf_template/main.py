"""Entrypoint to the application."""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger as log

try:
    __version__ = version("ocf-template")
except PackageNotFoundError:
    __version__ = "v?"

def main() -> None:
    """Entrypoint to the application.

    Thanks to the `script` section in `pyproject.toml`, this function is
    automatically executed when the package is run as a script via
    `ocf-template-cli`. Change name in pyproject to something more suitable
    for your package!
    """
    log.info("Hello, world!")
    log.info("Adding extra context to this log to help with debugging: ", version=__version__)

    log.info("You can also contextualize logs in bulk to save having to provide it repeatedly!")
    filenames: list[str] = [f"file_{i}.txt" for i in range(5)]
    log.debug("Might be some spooky numbers in these files!", num_files=len(filenames))
    for filename in filenames:
        with log.contextualize(filename=filename):
            log.debug("Trying to find spooky numbers in file {filename}")
            if "4" in filename:
                log.warning("Spooky number found!")
            else:
                log.debug("No spooky numbers found")

    log.info("Different log levels are also nicely formatted")
    for i, level in enumerate(["DEBUG", "INFO", "WARNING", "ERROR"]):
        log.log(level, f"This is a {level} log message", index=i)

    log.info("Consider wrapping your main in a try/except block to handle exceptions gracefully")
    try:
        y: int = int(3 / 0)
        log.warning("This will never be logged", y=y)
    except ZeroDivisionError as e:
        log.opt(exception=e).error("Caught exception")

    log.info("And if you run this in a container, you'll see structured logs instead!")
    log.info("Goodbye, world!")

