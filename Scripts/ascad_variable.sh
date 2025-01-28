#!/bin/bash

echo "Starting the experiments"


echo "ASCAD Variable"
python main.py -a 12000 -t 48000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_HWt2 --record-trials 1 --record-axis na -i DIM_1_cnn_best 
python main.py -a 12000 -t 48000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_HWt2 --record-trials 1 --record-axis na -i DIM_0_cnn_best
python main.py -a 12000 -t 48000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ascad-variable.h5 --pr /path/to/resultsresults/prof_paper -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_variable_HWt2 --record-trials 1 --record-axis na -i DIM_1_cnn_best_SMOTE --imb SMOTE