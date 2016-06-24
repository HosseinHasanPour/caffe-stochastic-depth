//
// Created by perandre on 6/24/16.
//

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <map>
#include <assert.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/sgd_solvers.hpp"



using namespace caffe;
using namespace std;

int main(int argc, char** argv)
{
    Caffe::set_mode(Caffe::GPU);

    string param_file = "examples/stochastic_depth/solver54.prototxt";
    SolverParameter param;
    ReadSolverParamsFromTextFileOrDie(param_file, &param);
    Solver<float>* solver = SolverRegistry<float>::CreateSolver(param);

    // cout << "Stochastic Depth Solver" << endl;

    solver->Solve_StochDep();
}