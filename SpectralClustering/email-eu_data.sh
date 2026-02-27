#!/bin/bash

wget -nc https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
gzip -df email-Eu-core-temporal.txt.gz
python email-eu_data.py 20
rm -f email-Eu-core-temporal.txt
