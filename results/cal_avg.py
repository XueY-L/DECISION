path = '/home/yxue/DECISION/results/domainnet126_bs50/DomainNet126_[\'real\', \'sketch\']_target-painting_bs50.txt'

f = open(path, 'r')

lines = f.readlines()

sum_ = 0
for l in lines:
    sum_ += float(l)
print(sum_ / len(lines))