import os

# for imagenetc
# for i in range(7, 8):
#     for j in range(5000//50):
#         cmd = f'python adapt_multi_imagenetc.py --dset imagenetc --t {i} --max_epoch 30 --gpu_id 0 --output_src ckps/source/ --output ckps/adapt --batch_idx {j} --batch_size 17'
#         print(cmd)
#         os.system(cmd)

# for cifar100c
for i in range(10, 15):
    for j in range(10000//200):
        cmd = f'python adapt_multi_cifar100c.py --dset cifar100c --t {i} --max_epoch 10 --gpu_id 2 --output_src ckps/source/ --output ckps/adapt --batch_idx {j}'
        print(cmd)
        os.system(cmd)