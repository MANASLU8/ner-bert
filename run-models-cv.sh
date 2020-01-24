#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}
NER_COMPARISON_ROOT=/home/nami/ner-comparison
NER_BERT_ROOT=/home/nami/ner-bert

. ./python/bin/activate
LD_LIBRARY_PATH=/usr/local/cuda/lib64

mkdir $NER_COMPARISON_ROOT/cv/ner-bert
for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
do
	python train-bert.py --train rucv/$i/train.txt --test rucv/$i/test.txt --idx rucv/idx2labels4.txt --checkpoint ckpts/$i.ckpt
	python predict.py --tagged --input_file rucv/$i/test.txt --output_file $NER_COMPARISON_ROOT/cv/ner-bert/$i.txt --train rucv/$i/train.txt --test rucv/$i/test.txt --idx rucv/idx2labels4.txt --checkpoint ckpts/$i.ckpt
done

deactivate
