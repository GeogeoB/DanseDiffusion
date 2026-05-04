from pathlib import Path

import torch
import numpy as np
import re

from dataset import MHRSequenceViewer, interpolate_1d

def extract_number(p):
    # récupère le nombre dans "output_728"
    match = re.search(r"(\d+)", p.stem)
    return int(match.group(1)) if match else -1

def parse(p):
    n, i = map(int, re.findall(r"\d+", p.stem))
    return n, i

if __name__ == "__main__":
    paths = list(Path("output").glob("*.npz"))
    max_n = max(parse(p)[0] for p in paths)
    paths = sorted(
        [p for p in paths if parse(p)[0] == max_n],
        key=lambda p: parse(p)[1]
    )[::2]
    # # paths = sorted(paths, key=extract_number)[-1:]

    # paths = [
    #     "output/output_728_5.npz",
    #     "output/output_728_6.npz",
    # ]

    print("paths", paths)

    print(f"{len(paths) = }")

    shapes = []
    mhrs = []
    exprs = []

    for path in paths:
        data = np.load(path)

        print(torch.from_numpy(data["sequence_shape"]).size())
        print(torch.from_numpy(data["sequence_mhr_model_params_latent"]).size())
        print(torch.from_numpy(data["sequence_expr_params"]).size())

        shapes.append(torch.from_numpy(data["sequence_shape"]))
        mhrs.append(torch.from_numpy(data["sequence_mhr_model_params_latent"]))
        exprs.append(torch.from_numpy(data["sequence_expr_params"]))

    # data = np.load("dataset/latent/gMH_sBM_c08_d24_mMH3_ch03.npz")

    # for k in data.keys():
    #     print(k)

    # shape = torch.from_numpy(data["sequence_shape"])
    # mhr = torch.from_numpy(data["sequence_mhr_model_params_latent"])
    # expr = torch.from_numpy(data["sequence_expr_params"])

    viewer = MHRSequenceViewer(fps=10)

    viewer.display_multiple_sequences(
        shapes, mhrs, exprs, loop=True
    )

    viewer.close()