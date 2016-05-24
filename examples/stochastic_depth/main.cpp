#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"



using namespace caffe;
using namespace std;

void standardResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob);
void transitionResLayer(int & elts, int & idx, vector<int>* layers_chosen, double ran, double prob);

int main(int argc, char** argv)
{
    Caffe::set_mode(Caffe::GPU);

    string param_file = "examples/stochastic_depth/solver.prototxt";
    SolverParameter param;
    ReadSolverParamsFromTextFileOrDie(param_file, &param);
    Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);
    shared_ptr<Net<float> > net = solver->net();
    vector<shared_ptr<Layer<float> > > layers = net->layers();


    vector<int>* layers_chosen = new vector<int>();
    net->ChooseLayers_StochDep(layers_chosen);

//	for (int i = 0; i < layers_chosen->size(); i++) {
//		cout << (*layers_chosen)[i] << ": " <<layers[(*layers_chosen)[i]]->type() << endl;
//	}
    solver->Solve_StochDep();
}