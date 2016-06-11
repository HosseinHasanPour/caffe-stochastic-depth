import pylab

list_of_files = [['stoch_train_loss14.txt', 'Stochastic Depth'], ['reg_train_loss14.txt', 'Reguar Depth']]

datalist = [ ( pylab.loadtxt(filename), label ) for filename, label in list_of_files ]

for data, label in datalist:
    pylab.plot( data[:,0], data[:,1], label=label )

pylab.legend()
pylab.title("14 Block Net")
pylab.xlabel("Iterations")
pylab.ylabel("Training Loss")
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
