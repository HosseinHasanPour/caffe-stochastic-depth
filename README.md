# Deep Networks with Stochastic Depth (README in progress)

This project is an implementation of the Stochastic Depth method for training neural Networks, as specified in the research paper
here: https://arxiv.org/abs/1603.09382. In summary: during training, layers are stochastically dropped from the network, while in testing all layers remain. This has been shown to result in lower test error and shorter training time than equivalent networks that don't use stochastic depth.

This implementation is a work in progress. It currently has a working example of a 54 resblock convolutoinal neural network. This network is identical to the networks specified in the stochastic depth paper. It uses a linear resblock survival rate from 1.0 to 0.5 from resblocks 1 to 54 respectively and runs on the cifar10 dataset.


# Getting Started

Follow the standard caffe installation procedure specified here: http://caffe.berkeleyvision.org/installation.html. 

To run the example, run the command ___ from the caffe root directory. 

# Implementation

In the current implementation, there are two c++ functions that must be replaced in order to train a different network with stochastic depth. These are:
- void Net<Dtype>::ChooseLayers_StochDep()
- void Net<Dtype>::InitTestScalingStochdept()

There functionality is pretty simple, but implementing them can be a bit rough.

ChooseLayers_StochDep() initializes the datastructure `vector<int> layers_chosen`
`vector<int> layers_chosen` contains the indeces of the layers in 
