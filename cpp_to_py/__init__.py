from importlib import import_module
from pathlib import Path
import sys

import torch

__all__ = [
    "dct_cuda",
    "flute_cpp",
    "hpwl_cuda",
    "io_parser",
    "density_map_cuda",
    "draw_placement",
    "wa_wirelength_hpwl_cuda",
    "gpugr",
    "gpudp",
    "routedp",
    "gputimer",
    "wirelength_timing_cuda",
]


def _load_extension(module_name):
    try:
        return import_module(f".cpybin.{module_name}", __name__)
    except ModuleNotFoundError:
        package_dir = Path(__file__).resolve().parent
        candidate_paths = [
            package_dir / "cpybin",
            package_dir.parent / "build" / "cpp_to_py" / module_name,
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                candidate_str = str(candidate_path)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return import_module(module_name)
        raise


dct_cuda = _load_extension("dct_cuda")
flute_cpp = _load_extension("flute_cpp")
hpwl_cuda = _load_extension("hpwl_cuda")
io_parser = _load_extension("io_parser")
density_map_cuda = _load_extension("density_map_cuda")
draw_placement = _load_extension("draw_placement")
wa_wirelength_hpwl_cuda = _load_extension("wa_wirelength_hpwl_cuda")
gpugr = _load_extension("gpugr")
gpudp = _load_extension("gpudp")
routedp = _load_extension("routedp")
gputimer = _load_extension("gputimer")
wirelength_timing_cuda = _load_extension("wirelength_timing_cuda")

