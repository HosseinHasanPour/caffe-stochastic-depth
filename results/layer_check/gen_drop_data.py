drop_data = open('drop_data.txt', 'w+')
# from dateutil import parser
import datetime
import time

with open("./layers_dropped_test") as f:
    data = f.read()

data = data.split('\n')
data = [row for row in data if not row.find('skipping block') == -1]
data = [row.split(':') for row in data]

data_dict  = {}
for row in data:
    key = row[1]
    if data_dict.has_key(key):
        data_dict[key] += 1
    else:
        data_dict[key] = 1

keys = data_dict.keys()
keys = [int(key) for key in keys]
keys.sort()

new_dict = [[1,1]]
for i in range(0, len(keys)):
    oldkey = ' ' + str(keys[i])
    val = data_dict[oldkey]
    new_dict+= [[i+2, 1 - float(val)/400.0]]

print new_dict


# for i in range(0, len(data)):
#     print data[i]
    # data[i] = [i*400, data[i][-2]]

drop_data.writelines(' '.join(str(j) for j in i) + '\n' for i in new_dict);
