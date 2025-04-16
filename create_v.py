import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

# v = torch.from_file('./marmousi_vp.bin', size=64*64).reshape(1, 64, 64)
# v = torch.log(v/1000)

v = np.load('./test_data/CurveVel-A/0.npy').reshape(1,64,64)
vv = (v-3000)/1500

np.save('./data/function/v.npy', vv)