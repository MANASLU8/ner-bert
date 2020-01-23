#!/bin/bash

NUMBER_OF_FOLDS=${1:-10}
NER_BERT_ROOT=/home/dima/ner-bert
FACT_RU_EVAL2STANFORD_ROOT=/home/dima/fact-ru-eval2stanford

cd $NER_BERT_ROOT
deactivate
. ./python/bin/activate

# Copy dataset to acl folder
# mkdir cv
# for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
# do
# 	cp -r $FACT_RU_EVAL2STANFORD_ROOT/conll2003rucv/$i $NER_BERT_ROOT/cv/
# done

sed -i -E "s/(.+) (.+) (.+) (.+)/\1 \4/" cv/*/*.txt
sed -i '1,2d' cv/*/*.txt

mkdir rucv
for i in $(seq -f "%02g" 1 $NUMBER_OF_FOLDS)
do
	mkdir rucv/$i
	python convert.py cv/$i/train.txt rucv/$i/train.txt
	python convert.py cv/$i/test.txt rucv/$i/test.txt
done