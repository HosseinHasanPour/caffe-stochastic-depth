test_accuracy = open('test_acc', 'w+')

with open("stoch.txt") as f:
    stoch = f.read()

stoch = stoch.split('\n')
stoch = [[row.split('=')[0], row.split('=')[1].split('(')[0]] for row in stoch if len(row.split('=')) >= 2] 

test_acc = [row for row in stoch if not row[0].find('Test') == -1 and not row[0].find('Accuracy') == -1]

test_acc = []
for i in range(0,len(stoch)):
	row = stoch[i]
	newrow = []
	
	if not row[0].find('Test') == -1 and not row[0].find('Accuracy') == -1:
		if i == 0:
			Iteration = 0
		else:
			Iteration = min(int(stoch[i-1][0].split(',')[0].split(' ')[-1])+10, 64000)
		newrow.append(Iteration)
		newrow.append(row[1])
		test_acc.append(newrow)

test_accuracy.writelines(' '.join(str(j) for j in i) + '\n' for i in test_acc)


#for row in test_acc:
#    print row


with open("reg.txt") as f:
    reg = f.read()

