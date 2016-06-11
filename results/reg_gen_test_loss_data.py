reg_test_loss14 = open('reg_test_loss14.txt', 'w+')

with open("stoch14.txt") as f:
    data = f.read()

data = data.split('\n')
data = [[row.split('=')[0], row.split('=')[1].split('(')[0]] for row in data if len(row.split('=')) >= 2]

data = [row for row in data if not row[0].find('Test') == -1 and not row[0].find('SoftmaxWithLoss1') == -1]

for i in range(0, len(data)):
    data[i][0] = i*100

# for i in range(1, len(data)):
#     print data[i

reg_test_loss14.writelines(' '.join(str(j) for j in i) + '\n' for i in data);


# test_acc = []
# for i in range(0,len(stoch)):
# 	row = stoch[i]
# 	newrow = []
#
# 	if not row[0].find('Test') == -1 and not row[0].find('Accuracy') == -1:
# 		if i == 0:
# 			Iteration = 0
# 		else:
# 			Iteration = min(int(stoch[i-1][0].split(',')[0].split(' ')[-1])+10, 64000)
# 		newrow.append(Iteration)
# 		newrow.append(row[1])
# 		test_acc.append(newrow)
#
# test_accuracy.writelines(' '.join(str(j) for j in i) + '\n' for i in test_acc)
#
#
# #for row in test_acc:
# #    print row
#
#
# with open("reg.txt") as f:
#     reg = f.read()
