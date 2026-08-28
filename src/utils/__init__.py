"""Utility modules for physics validation, visualization, and runtime gates."""


def assert_tensor_on_gpu(tensor, label: str, *, require_gpu: bool) -> str:
    """Fail closed when a formal-training tensor is not placed on a GPU."""
    device = str(getattr(tensor, "device", ""))
    if require_gpu and "/DEVICE:GPU:" not in device.upper():
        raise RuntimeError(
            f"CPU fallback detected for required GPU tensor {label!r}: {device or 'unknown device'}"
        )
    return device
