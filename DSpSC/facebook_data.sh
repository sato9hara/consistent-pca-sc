#!/bin/bash

wget -nc https://snap.stanford.edu/data/facebook.tar.gz
tar -zxvf facebook.tar.gz
python facebook_data.py 10
rm -f facebook.tar.gz
rm -rf facebook

