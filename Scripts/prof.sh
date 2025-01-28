#!/bin/bash

echo "Starting the experiments"
# python main.py -a 20000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --with-regularization
#  python main.py -a 10000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --with-regularization
# echo "1-----------------"
# python main.py -a 20000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --no-regularization python main.py -a 10000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --no-regularization
# echo "2-----?------------"


# Here I am traiing an experiment wiht dim 0
# python main.py -a 20000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week4 -d 0 -e 50 --lm LSB --loss nll --lc SBox -m mlp --cpu --fast -v 1 

# echo "Trying to reporduce benjamin timon"
# python main.py -a 20000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --with-regularization
# python main.py -a 20000 -t 0 -b 2 --batch-size 1000 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/week5 -d 1 -e 50 --lm LSB --loss mse --lc SBox -m mlp --cpu -v 1 --with-regularization


# This is for the MLP BEST
echo "ASCAD MLP Best"
python main.py -a 1000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 1 --no-regularization --record ASCAD_HW_10_final --record-trials 10 --record-axis na -i DIM_1_mlp_best 
python main.py -a 1000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/prof_paper10 -d 0 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 1 --no-regularization --record ASCAD_HW_10_final --record-trials 10 --record-axis na -i DIM_0_mlp_best 
python main.py -a 1000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 1 --no-regularization --record ASCAD_HW_10_final --record-trials 10 --record-axis na -i DIM_1_nll_SMOTE --imb SMOTE


# This is for the CNN BEST
# echo "ASCAD Desync 50 CNN Best"
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync50.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_desyn50_HW_10_r --record-trials 10 --record-axis na -i DIM_1_cnn_best 
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync50.h5 --pr /path/to/resultsresults/prof_paper10 -d 0 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_desync50_HW_10_r --record-trials 10 --record-axis na -i DIM_0_cnn_best
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync50.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_descyn50_HW_10_r --record-trials 10 --record-axis na -i  DIM_1_cnn_best_SMOTE --imb SMOTE


# echo "ASCAD Desync 100 CNN Best"
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync100.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_desync100_HW_10_r --record-trials 10 --record-axis na -i DIM_1_cnn_best 
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync100.h5 --pr /path/to/resultsresults/prof_paper10 -d 0 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_desync100_HW_10_r --record-trials 10 --record-axis na -i DIM_0_cnn_best
# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /path/to/datasets/ -f ASCAD_desync100.h5 --pr /path/to/resultsresults/prof_paper10 -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_best -v 1 --no-regularization --record ASCAD_descyn100_HW_10_r --record-trials 10 --record-axis na -i  DIM_1_cnn_best_SMOTE --imb SMOTE



# python main.py -a 2500 -t 47500 -b 2 --batch-size 100 --pd /path/to/datasets/ -f original.h5 --pr /path/to/resultsresults/prof -d 0 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 1 --with-regularization --record Test21_HW_1_original --record-trials 1 --record-axis na -i DIM_0_nll_n
# python main.py -a 10000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/prof -d 0 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 2 --with-regularization -i DIM_0_nll_n




# python main.py -a 10000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD_desync50.h5 --pr /path/to/resultsresults/prof -d 1 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_exp -v 1 --with-regularization --record Test_desync50_HW_10 --record-trials 10 --record-axis na -i DIM_1_nll_n
# python main.py -a 10000 -t 50000 -b 2 --batch-size 100 --pd /path/to/datasets/ -f ASCAD_desync50.h5 --pr /path/to/resultsresults/prof -d 0 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_exp -v 1 --with-regularization --record Test_desync50_HW_10 --record-trials 10 --record-axis na -i DIM_0_nll_n



# python main.py -a 10000 -t 50000 -b 2 --batch-size 50 --pd /path/to/datasets/ -f ASCAD.h5 --pr /path/to/resultsresults/prof_paper -d 0 -e 200 --lm HW --loss nll --optimizer RMSprop --lc SBox -m mlp_ascad -v 1 --with-regularization --record ASCAD_HW_1_5 --record-trials 1 --record-axis na -i DIM_0_nll_n_50_batches

# python main.py -a 1000 -t 50000 -b 2 --batch-size 200 --pd /local/user/ldm/Data/CiC/datasets/ -f ASCAD.h5 --pr /local/user/ldm/Data/CiC/results/prof_paper -d 1 -e 100 --lm HW --loss nll --optimizer RMSprop --lc SBox -m cnn_zaid0 -v 1 --with-regularization --record ASCAD_2384821 --record-trials 1 --record-axis na -i DIM_1 -r 3