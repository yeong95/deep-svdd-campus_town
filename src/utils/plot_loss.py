import os 
import numpy as np 
import sys

import matplotlib.pyplot as plt 

save_path = '../../log/tofu_test'

f = open('../../log/tofu_test/log.txt', 'r')

# data = f.readlines()[21:323]
# import pdb;pdb.set_trace()
# print(data)

# sys.exit(0)

loss_list = []

while True:
    line = f.readline()
    if not line: break
    # import pdb;pdb.set_trace()
    split_line = line.split(' ')
    if (np.array(split_line) == 'Loss:').sum() > 0 and \
        (np.array(split_line) == 'Epoch').sum() > 0:
        # print(line)
        # print(split_line[-1][:-1])
        loss_list.append(split_line[-1][:-1])

        
pretraining_loss = loss_list[:300]
pretraining_loss = [float(i) for i in pretraining_loss]

training_loss = loss_list[300:]
training_loss = [float(i) for i in training_loss]

plt.figure()
plt.plot(np.arange(len(pretraining_loss)), pretraining_loss, marker='.', c='red')
plt.title('pretraning loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(save_path, 'pretraining_loss.png'))

plt.figure()
plt.plot(np.arange(len(training_loss)), training_loss, marker='.', c='red')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss')
plt.savefig(os.path.join(save_path, 'training_loss.png'))