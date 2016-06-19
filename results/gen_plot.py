import pylab
import numpy as np

list_of_files = [['layer_check/drop_data.txt', 'layers']]

datalist = [ ( pylab.loadtxt(filename), label ) for filename, label in list_of_files ]

for data, label in datalist:
    pylab.plot( data[:,0], data[:,1], label=label )

pylab.legend()
pylab.title("Resblock survival rate")
pylab.xlabel("survival rate")
pylab.ylabel("resblock num")
pylab.xticks(np.arange(1, 55, 1))
pylab.show()


# with open("stoch.txt") as f:
#     stoch = f.read()
#
# stoch = stoch.split(' ')
#
# for row in stoch:
#     print row
#
#
# with open("reg.txt") as f:
#     reg = f.read()
