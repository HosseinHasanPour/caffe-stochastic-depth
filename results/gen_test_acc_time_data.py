stoch_test_acc_time14 = open('stoch_test_acc_time14.txt', 'w+')
stoch_test_acc_time_diffs14 = open('stoch_test_acc_time_diffs14.txt', 'w+')
# from dateutil import parser
import datetime
import time

with open("stoch14.txt") as f:
    data = f.read()

data = data.split('\n')
data = [row for row in data if not row.find('Test net output') == -1 and not row.find('Accuracy') == -1]
data = [row.split(' ') for row in data]
data = [[row[0] + ' ' + row[1], row[-1]] for row in data]
data = [['6/16: ' + row[0][3:-7], row[1] ] for row in data ]
firsttime = time.mktime(datetime.datetime.strptime(data[0][0], "%m/%y: %d %H:%M:%S" ).timetuple())
data = [[time.mktime(datetime.datetime.strptime(row[0], "%m/%y: %d %H:%M:%S" ).timetuple()) - firsttime, row[1]] for row in data ]

stoch_test_acc_time14.writelines(' '.join(str(j) for j in i) + '\n' for i in data);

for i in range(1,len(data)):
    stoch_test_acc_time_diffs14.write(str((i-1)*100) + " " + str(data[i][0] - data[i-1][0]) + '\n')



reg_test_acc_time14 = open('reg_test_acc_time14.txt', 'w+')
reg_test_acc_time_diffs14 = open('reg_test_acc_time_diffs14.txt', 'w+')

with open("reg14.txt") as f:
    data = f.read()

data = data.split('\n')
data = [row for row in data if not row.find('Test net output') == -1 and not row.find('Accuracy') == -1]
data = [row.split(' ') for row in data]
data = [[row[0] + ' ' + row[1], row[-1]] for row in data]
data = [['6/16: ' + row[0][3:-7], row[1] ] for row in data ]
firsttime = time.mktime(datetime.datetime.strptime(data[0][0], "%m/%y: %d %H:%M:%S" ).timetuple())
data = [[time.mktime(datetime.datetime.strptime(row[0], "%m/%y: %d %H:%M:%S" ).timetuple()) - firsttime, row[1]] for row in data ]

reg_test_acc_time14.writelines(' '.join(str(j) for j in i) + '\n' for i in data);

for i in range(1,len(data)):
    reg_test_acc_time_diffs14.write(str((i-1)*100) + " " + str(data[i][0] - data[i-1][0]) + '\n')
