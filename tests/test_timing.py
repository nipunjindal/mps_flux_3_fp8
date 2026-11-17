import time

from mps_flux_3_fp8.utils.timing import segment_timer


def test_segment_timer():
    with segment_timer("Test segment"):
        time.sleep(0.01)
