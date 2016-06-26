import pylab
import numpy as np

list_of_files = [['../plotdata/reg_train_loss.txt', 'regular depth'],['../plotdata/stoch_train_loss.txt', 'stochastic depth']]

datalist = [ ( pylab.loadtxt(filename), label ) for filename, label in list_of_files ]

for data, label in datalist:
    pylab.plot( data[:,0], data[:,1], label=label )

pylab.legend()
pylab.title("54 Resblock Net Training Loss vs Iterations")
pylab.xlabel("Iterations")
pylab.ylabel("Training Loss (Softmax)")
pylab.xticks(np.arange(0, 250001, 25000))
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
