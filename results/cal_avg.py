path = '/home/yxue/DECISION/results/domainnet126_bs50_原来的aug训的源模型_15epochs/DomainNet126_[\'clipart\', \'painting\']_target-sketch_bs50.txt'

f = open(path, 'r')

lines = f.readlines()

sum_ = 0
for l in lines:
    sum_ += float(l)
print(sum_ / len(lines))