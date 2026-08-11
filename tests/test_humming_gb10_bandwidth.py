from types import SimpleNamespace
from unittest.mock import patch

import pynvml
import pytest

from humming.utils.device import calculate_gpu_bandwidth


def _not_supported(*_args):
    raise pynvml.NVMLError_NotSupported


def test_gb10_bandwidth_falls_back_to_cuda_device_properties() -> None:
    properties = SimpleNamespace(
        memory_clock_rate=8_533_000,
        memory_bus_width=256,
    )
    with (
        patch.object(pynvml, "nvmlInit"),
        patch.object(pynvml, "nvmlShutdown"),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=object()),
        patch.object(pynvml, "nvmlDeviceGetName", return_value="NVIDIA GB10"),
        patch.object(pynvml, "nvmlDeviceGetMemoryBusWidth", return_value=0),
        patch.object(pynvml, "nvmlDeviceGetMaxClockInfo", side_effect=_not_supported),
        patch("torch.cuda.get_device_properties", return_value=properties),
    ):
        assert calculate_gpu_bandwidth() == pytest.approx(273.056)


def test_unknown_device_does_not_guess_bandwidth() -> None:
    with (
        patch.object(pynvml, "nvmlInit"),
        patch.object(pynvml, "nvmlShutdown"),
        patch.object(pynvml, "nvmlDeviceGetHandleByIndex", return_value=object()),
        patch.object(pynvml, "nvmlDeviceGetName", return_value="Unknown GPU"),
        patch.object(pynvml, "nvmlDeviceGetMemoryBusWidth", return_value=0),
        patch.object(pynvml, "nvmlDeviceGetMaxClockInfo", side_effect=_not_supported),
    ):
        with pytest.raises(pynvml.NVMLError_NotSupported):
            calculate_gpu_bandwidth()
