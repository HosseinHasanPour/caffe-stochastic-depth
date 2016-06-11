stoch_train_loss14 = open('stoch_train_loss14.txt', 'w+')

with open("stoch14.txt") as f:
    data = f.read()

data = data.split('\n')
data = [[row.split('=')[0], row.split('=')[1].split('(')[0]] for row in data if len(row.split('=')) >= 2]

data = [row for row in data if not row[0].find('Train') == -1 and not row[0].find('SoftmaxWithLoss1') == -1]

for i in range(0, len(data)):
    if i % 10 == 0:
        data[i][0] = i*10

# for i in range(1, len(data)):
#     print data[i

stoch_train_loss14.writelines(' '.join(str(j) for j in i) + '\n' for i in data);




reg_train_loss14 = open('reg_train_loss14.txt', 'w+')

with open("reg14.txt") as f:
    data = f.read()

data = data.split('\n')
data = [[row.split('=')[0], row.split('=')[1].split('(')[0]] for row in data if len(row.split('=')) >= 2]

data = [row for row in data if not row[0].find('Train') == -1 and not row[0].find('SoftmaxWithLoss1') == -1]

for i in range(0, len(data)):
    if i % 10 == 0:
        data[i][0] = i*10

# for i in range(1, len(data)):
#     print data[i

reg_train_loss14.writelines(' '.join(str(j) for j in i) + '\n' for i in data);
