import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis


def main():
    parser = argparse.ArgumentParser(description="Profile the bi-temporal PUGNet model.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bcd_root = Path(__file__).resolve().parent / "building-change-detection"
    sys.path.insert(0, str(bcd_root))
    os.chdir(bcd_root)

    import network.PUGNet as pugnet_module

    # PVT checkpoint values do not affect parameter/FLOP counts; preserve PVT-v2-B4 structure.
    original_torch_load = torch.load

    def skip_pvt_weight(path, *load_args, **load_kwargs):
        if str(path).replace("\\", "/").endswith("pretrained_model/pvt_v2_b4.pth"):
            return {}
        return original_torch_load(path, *load_args, **load_kwargs)

    torch.load = skip_pvt_weight
    try:
        model = pugnet_module.PUGNet(backbone_name="PVT-v2", channel=32)
    finally:
        torch.load = original_torch_load

    device = torch.device(args.device)
    model = model.eval().to(device)
    inputs = tuple(torch.randn(1, 3, args.height, args.width, device=device) for _ in range(2))

    flop_error = None
    unsupported = {}
    try:
        analysis = FlopCountAnalysis(model, inputs)
        flops = analysis.total()
        unsupported = dict(Counter(analysis.unsupported_ops()))
    except Exception as exc:
        flops = None
        flop_error = f"{type(exc).__name__}: {exc}"

    params = sum(p.numel() for p in model.parameters())
    result = {
        "model": "PU_RS-PUGNet-PVT-v2-B4",
        "input": "two RGB images, batch=1",
        "image_size": [args.height, args.width],
        "params_m": params / 1_000_000,
        "parameters_total": params,
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "flops_g": flops / 1_000_000_000 if flops is not None else None,
        "fvcore_flops_per_pair": flops,
        "fvcore_convention": "one multiply-add is counted as one FLOP",
        "unsupported_ops": unsupported,
        "fvcore_error": flop_error,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)


if __name__ == "__main__":
    main()
