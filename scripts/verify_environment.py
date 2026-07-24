from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit(f"Python 3.11+ is required; found {platform.python_version()}")

    import torch

    import poker_arena

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the tensorized prefix-branch trainer, but it is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected exactly one CUDA GPU on this workstation; found {torch.cuda.device_count()}")

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )

    if devices[0]["compute_capability"][0] < 12:
        raise SystemExit(
            f"Expected an RTX 50-series-capable CUDA runtime; compute capability is {devices[0]['compute_capability']}"
        )

    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "executable": sys.executable,
                "poker_arena": poker_arena.__file__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "devices": devices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
