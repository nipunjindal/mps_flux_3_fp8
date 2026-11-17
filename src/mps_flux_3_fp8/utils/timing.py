import time
from collections.abc import Generator
from contextlib import contextmanager

from mps_flux_3_fp8.utils.log import logger


@contextmanager
def segment_timer(description: str = "", verbose: bool = True) -> Generator[None, None, None]:
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if verbose:
            desc = f"{description} " if description else ""
            logger.info(f"[Timer] {desc}took {elapsed:.2f} seconds")
