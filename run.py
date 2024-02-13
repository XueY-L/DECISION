import os

for i in range(1, 14):
    for j in range(5000//50):
        cmd = f'python adapt_multi_imagenetc.py --dset imagenetc --t {i} --max_epoch 30 --gpu_id 2 --output_src ckps/source/ --output ckps/adapt --batch_idx {j} --batch_size 17'
        print(cmd)
        os.system(cmd)