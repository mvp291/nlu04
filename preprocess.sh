#!/bin/bash

echo 'Processing datasets...'

# Process train dataset and create vocabulary dict
echo 'Processing train...'
python preprocess.py --createDict=True --inputFile=./data/train.ja
python preprocess.py --createDict=False --inputFile=./data/train.en

echo 'Processing val...'
python preprocess.py --createDict=False --inputFile=./data/dev.ja
python preprocess.py --createDict=False --inputFile=./data/dev.en

echo 'Processing test...'
python preprocess.py --createDict=False --inputFile=./data/test.ja
python preprocess.py --createDict=False --inputFile=./data/test.en
