#!/bin/bash
echo "Hello CHTC from Job $1 running on `hostname`"

unzip data.zip
# unzip large training dataset
unzip Train-1.zip

rm data.zip
rm Train-1.zip

# run the pytorch model
python main_$1.py $1 # --save-model --epochs 10

# remove the data directory
# rm -r data