#!/bin/bash

set -e

echo "Starting Spectral Clustering experiments..."
cd ./SpectralClustering

echo "Running SBM experiment..."
./sbm_data.sh
./sbm.sh
echo "SBM experiment finished."

echo "Running email-Eu experiment..."
./email-eu_data.sh
./email-eu.sh
echo "email-Eu experiment finished."

echo "Running facebook experiment..."
./facebook_data.sh
./facebook.sh
echo "facebook experiment finished."

echo "All experiments finished successfully."
