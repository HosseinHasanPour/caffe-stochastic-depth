#!/usr/bin/env bash

git pull
make examples
./build/examples/stochastic_depth/main54_reg.bin
