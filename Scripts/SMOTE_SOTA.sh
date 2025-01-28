#!/bin/bash

echo "Starting the experiments"


# echo "ASCAD RD"

# # python main.py -a 2500 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --pd /path/to/datasets/ -f AES_RD.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 10 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,25,50,90,200 -v 1 --no-regularization --record AES_RD_TEST --record-trials 1 --record-axis na -i DIM_0_MLP
# python main.py -a 2500 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 10 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,25,50,90,200 -v 1 --no-regularization --record AES_RD_TEST --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

# echo "ASCAD HD byte 0 "
# # python main.py -a 2500 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 10 --lm HD --loss nll --optimizer Adam --lc SBox -m mpp -v 1 --no-regularization --record AES_HD_TEST --record-trials 1 --record-axis na -i DIM_0_CNN
# python main.py -a 2500 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 10 --lm HD --loss nll --optimizer Adam --lc SBox -m mpp -v 1 --no-regularization --record AES_HD_TEST --record-trials 1 --record-axis na -i DIM_1_CNN --imb SMOTE



echo "ASCAD RD byte 0 "

# python main.py -a 25000 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_RD1 --record-trials 1 --record-axis na -i DIM_0_MLP
# python main.py -a 25000 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_RD1 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_RD_10k_mlp_2 --record-trials 1 --record-axis na -i DIM_0_MLP
python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5  --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_RD_10k_mlp_2 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 25000 -b 0 --batch-size 256 --prep 4  --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5  --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_RD_25k_mlp_1 --record-trials 1 --record-axis na -i DIM_0_MLP
python main.py -a 5000 -t 25000 -b 0 --batch-size 256 --pearson 50 --prep 4  --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5  --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_RD_25k_mlp_2 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5  --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m mpp -v 1 --no-regularization --record AES_RD_10k_cnn_2 --record-trials 1 --record-axis na -i DIM_0_CNN
python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5  --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m mpp -v 1 --no-regularization --record AES_RD_10k_cnn_2 --record-trials 1 --record-axis na -i DIM_1_CNN --imb SMOTE

python main.py -a 5000 -t 25000 -b 0 --batch-size 256 --prep 4  --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_RD_25k_cnn_2 --record-trials 1 --record-axis na -i DIM_0_CNN
python main.py -a 5000 -t 25000 -b 0 --batch-size 256 --pearson 50 --prep 4  --lr 0.0001 --pd /path/to/datasets/ -f AES_RD.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HW --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_RD_25k_cnn_2 --record-trials 1 --record-axis na -i DIM_1_CNN --imb SMOTE




echo "ASCAD HD byte 0 "

# python main.py -a 25000 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_HD1 --record-trials 1 --record-axis na -i DIM_0_MLP
# python main.py -a 25000 -t 1000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_HD1 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_HD_10k_mlp_2 --record-trials 1 --record-axis na -i DIM_0_MLP
python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,relu,50,30,20,50 -v 1 --no-regularization --record AES_HD_10k_mlp_2 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 50000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_HD_25k_mlp_2 --record-trials 1 --record-axis na -i DIM_0_MLP
python main.py -a 5000 -t 50000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mlp_aesrd_imb,tanh,50,30,20,50 -v 1 --no-regularization --record AES_HD_25k_mlp_2 --record-trials 1 --record-axis na -i DIM_1_MLP --imb SMOTE

python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_HD_10k_cnn_2 --record-trials 1 --record-axis na -i DIM_0_CNN
python main.py -a 5000 -t 10000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_HD_10k_cnn_2 --record-trials 1 --record-axis na -i DIM_1_CNN --imb SMOTE

python main.py -a 5000 -t 50000 -b 0 --batch-size 256 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 0 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_HD_50k_cnn_2 --record-trials 1 --record-axis na -i DIM_0_CNN
python main.py -a 5000 -t 50000 -b 0 --batch-size 256 --pearson 50 --prep 4 --lr 0.0001 --pd /path/to/datasets/ -f aes_hd.h5 --pr /path/to/resultsresults/prof_paper_SMOTE -d 1 -e 1000 --lm HD --loss nll --optimizer Adam --lc SBox -m  mpp -v 1 --no-regularization --record AES_HD_50k_cnn_2 --record-trials 1 --record-axis na -i DIM_1_CNN --imb SMOTE