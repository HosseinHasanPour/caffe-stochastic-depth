stoch_test_acc54 = open('../plotdata/stoch_test_acc.txt', 'w+')
# from dateutil import parser
import datetime
import time

with open("../data/stoch54_gpu0_db1_lr01.txt") as f:
    data = f.read()

data = data.split('\n')
data = [row for row in data if not row.find('Test net output') == -1 and not row.find('Accuracy') == -1]
data = [row.split(' ') for row in data]

for i in range(0, len(data)):
    data[i] = [i*400, data[i][-1]]

stoch_test_acc54.writelines(' '.join(str(j) for j in i) + '\n' for i in data);




reg_test_acc54 = open('../plotdata/reg_test_acc.txt', 'w+')

with open("../data/reg54_gpu1_db3_lr01.txt") as f:
    data = f.read()


data = data.split('\n')
data = [row for row in data if not row.find('Test net output') == -1 and not row.find('Accuracy') == -1]
data = [row.split(' ') for row in data]


for i in range(0, len(data)):
    data[i] = [i*400, data[i][-1]]

reg_test_acc54.writelines(' '.join(str(j) for j in i) + '\n' for i in data);
