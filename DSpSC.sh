#!/bin/bash

set -e

echo "Starting Spectral Clustering with Sparsifier experiments..."
cd ./DSpSC

echo "Running SBM experiment..."
./sbm.sh
echo "SBM experiment finished."

echo "Running email-Eu experiment..."
./email-eu.sh
echo "email-Eu experiment finished."

echo "Running facebook experiment..."
./facebook.sh
echo "facebook experiment finished."

echo "All experiments finished successfully."
