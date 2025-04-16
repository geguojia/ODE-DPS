import os
import numpy as np
for i in range(100):
    v = np.load('./test_data/CurveVel-A/{}.npy'.format(i)).reshape(1,64,64)
    vv = (v-3000)/1500

    np.save('./data/function/v.npy', vv)
    
    os.system('python3 sample_condition.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/diffusion_config.yaml \
        --task_config=configs/forward_process.yaml \
        --gpu=0 \
        --save_dir=myresult;')