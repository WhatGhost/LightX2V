from lightx2v_platform.base import global_var
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def get_fp8_gemm_triton_configs(device):
    """Platform-tuned launch configs for the fp8-triton GEMM kernels on ``device``, or None to keep their autotuning."""
    get_configs = getattr(PLATFORM_DEVICE_REGISTER[global_var.PLATFORM], "fp8_gemm_triton_configs", None)
    return None if get_configs is None else get_configs(device)
