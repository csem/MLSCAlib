#!/bin/bash

echo "Starting the experiments"


echo "ASCAD Variable"
# python main.py -a 10000 -t 110000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 1 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt3 --record-trials 1 --record-axis na -i DIM_1_cnn_best 
# python main.py -a 10000 -t 110000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt3 --record-trials 1 --record-axis na -i DIM_0_cnn_best
#python main.py -a 12000 -t 110000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 1 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt2 --record-trials 1 --record-axis na -i DIM_1_cnn_best_SMOTE --imb SMOT

echo Trzing Lucas options

python main.py -a 10000 -t 110000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt4 --record-trials 1 --record-axis epochs -i DIM_0_cnn_best_batch100
python main.py -a 10000 -t 110000 -b 2 --batch-size 400 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt4 --record-trials 1 --record-axis epochs -i DIM_0_cnn_best_batch_400
python main.py -a 10000 -t 110000 -b 2 --batch-size 50 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 100 --lm ID --loss nll --optimizer RMSprop --lr 0.00001 --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_IDt4 --record-trials 1 --record-axis epochs -i DIM_0_cnn_best_batch_50


python main.py -a 2500 -t 47500 -b 3 --batch-size 50 --pd /path/to/datasets/ -f original.h5 --pr /local/user/ldm/Data/CiC/results/testWeight -d 1 -e 50 --lm HW --loss nll --optimizer Adam --lc SBox -m cnn_exp -v 1 --no-regularization 
