import numpy as np

a = np.load("dataset/data/gHO_sFM_c01_d20_mHO3_ch11/frame_0008.npz", allow_pickle=True)["arr_0"].item()
for k, i in a.items():
    print(f"{k}:")
    print(i)
    if type(i) is np.ndarray:
        print("shape", i.shape)
    print()
