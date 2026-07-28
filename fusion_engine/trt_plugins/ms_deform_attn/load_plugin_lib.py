import ctypes
import os

PLUGIN_LIBRARY_NAME = "libpose_ms_deform_attn_plugin.so"
PLUGIN_LIBRARY = os.path.join(os.path.dirname(os.path.realpath(__file__)), PLUGIN_LIBRARY_NAME)


def load_plugin_lib() -> str:
    if not os.path.isfile(PLUGIN_LIBRARY):
        raise IOError(
            f"Failed to load {PLUGIN_LIBRARY}. Build it with: make -C trt_plugins/ms_deform_attn"
        )
    try:
        ctypes.CDLL(PLUGIN_LIBRARY, winmode=0)
    except TypeError:
        ctypes.CDLL(PLUGIN_LIBRARY)
    return PLUGIN_LIBRARY
