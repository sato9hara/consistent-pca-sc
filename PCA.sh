#!/bin/bash

set -e

default_methods="BatchPCA RobustPCA LossGD LossMEG ConstPCA"

if [ "$#" -eq 0 ]; then
  methods_to_run=$default_methods
else
  methods_to_run="$@"
fi

echo "Starting PCA experiments..."
cd ./PCA

echo "Running synthetic experiment..."
./synthetic.sh $methods_to_run
echo "synthetic experiment finished."

echo "Running openml experiment..."
./openml.sh $methods_to_run
echo "openml experiment finished."

echo "Running face experiment..."
./face.sh $methods_to_run
echo "face experiment finished."

cd ../
echo "All experiments finished successfully."
