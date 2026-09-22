import json
import platform
import shutil
import torch


def main():
    info = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,
        free_disk_gb=round(shutil.disk_usage(".").free / 1024**3, 1),
    )
    if torch.cuda.is_available():
        info.update(
            gpu=torch.cuda.get_device_name(),
            vram_gb=round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            ),
            bf16=torch.cuda.is_bf16_supported(),
        )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
