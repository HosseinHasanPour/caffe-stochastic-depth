import pylab

list_of_files = [['54net/stoch_train_loss.txt', 'Stochastic Depth'], ['54net/reg_train_loss.txt', 'Reguar Depth']]

datalist = [ ( pylab.loadtxt(filename), label ) for filename, label in list_of_files ]

for data, label in datalist:
    pylab.plot( data[:,0], data[:,1], label=label )

pylab.legend()
pylab.title("54 Block Net Training Loss vs Iterations")
pylab.xlabel("Iterations")
pylab.ylabel("Train Loss")
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
