import numpy as np

predicted={}
def output2file(predicted, ids, save_path):
    first = True
    for id in ids:
        sale = np.array(predicted[str(id)])
        line = np.insert(sale, 0, id, axis=0)[None, :]
        # line=np.concatenate((np.array(id),sale))
        if first:
            output = line
            first = False
            continue
        output=np.append(output, line, axis=0)
    overall = np.sum(output, axis=0)
    output = np.insert(output, 0, overall, axis=0)
    output[0][0] = 0
    np.savetxt(save_path, output, fmt='%d', delimiter=' ')

ids=[]
sale=[]
with open('product_distribution_training_set.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        ids.append(line.strip().split()[0])

with open('result.txt','r') as f:
    lines=f.readlines()
    for line in lines:
        sale_per = [int(q) for q in line.strip().split(' ')[0:]]
        sale.append(sale_per)

for i in range(len(ids)):
    predicted[str(ids[i])]=sale[i]
output2file(predicted,ids,'final.txt')

