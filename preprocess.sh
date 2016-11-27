#!/bin/bash

echo 'Processing datasets...'

# Process train dataset and create vocabulary dict
echo 'Processing train...'
python preprocess.py --createDict=True --dictFile=./data/ja_dict.pkl --inputFile=./data/train.ja
python preprocess.py --createDict=True --dictFile=./data/en_dict.pkl --inputFile=./data/train.en

echo 'Processing val...'
python preprocess.py --createDict=False --dictFile=./data/ja_dict.pkl --inputFile=./data/dev.ja
python preprocess.py --createDict=False --dictFile=./data/en_dict.pkl --inputFile=./data/dev.en

echo 'Processing test...'
python preprocess.py --createDict=False --dictFile=./data/ja_dict.pkl --inputFile=./data/test.ja
python preprocess.py --createDict=False --dictFile=./data/en_dict.pkl --inputFile=./data/test.en
