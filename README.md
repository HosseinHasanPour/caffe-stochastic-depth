# Deep Networks with Stochastic Depth (README in progress)

This project is an implementation of the Stochastic Depth method for training neural Networks, as specified in the research paper
here: https://arxiv.org/abs/1603.09382. In summary: during training, layers are stochastically dropped from the network, while in testing all layers remain. This has been shown to result in lower test error and shorter training time than equivalent networks that don't use stochastic depth.

This implementation is a work in progress. It currently has a working example of a 54 resblock convolutoinal neural network. This network is identical to the networks specified in the stochastic depth paper. It uses a linear resblock survival rate from 1.0 to 0.5 from resblocks 1 to 54 respectively and runs on the cifar10 dataset.


## Getting Started

Follow the standard caffe installation procedure specified here: http://caffe.berkeleyvision.org/installation.html. 

To run the example, run the command ___ from the caffe root directory. 

## Implementation

In the current implementation, there are two c++ functions that must be replaced in order to train a different network with stochastic depth. These are:
- `void Net<Dtype>::ChooseLayers_StochDep()`
- `void Net<Dtype>::InitTestScalingStochdept()`

Their functionality is pretty simple, but implementing them can be a bit rough.

## `void Net<Dtype>::ChooseLayers_StochDep()`

`Net<Dtype>::ChooseLayers_StochDep()` is called by the solver in the function `void Solver<Dtype>::Step(int iters)` during every training iteration. It's job is to initialize a few data structures: 

- `vector<int> layers_chosen`  
- `vector<vector<Blob<Dtype>*> > bottom_vecs_stochdept_;`
- `vector<vector<Blob<Dtype>*> > top_vecs_stochdept_;`

### `vector<int> layers_chosen` 
Internally, caffe stores it's layers in a vector, and it loops through this vector when doing forward and backward passes, calling each layer's individual forward and backward function. This vector is called ` vector<shared_ptr<Layer<Dtype> > > layers_`. `layers_chosen` must contain the indexes in `layers_` of all the layers that are *not* getting dropped in the current iteration. That is, all the surviving layers. The indeces must be ordered from least to greatest.

### `vector<vector<Blob<Dtype>*> > bottom_vecs_stochdept_;`
This vector stores pointers to bottom blobs of the layers specified in  `layers_chosen`. The indexes in `bottom_vecs_stochdept_` correspond to
