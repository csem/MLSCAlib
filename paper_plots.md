# Reproducing Plots
Some Figures from the "Taking AI-BAsed Side-Channel Attacks to a New Dimension" can be reproduced easily by using the following commands.


## Figure Generation

To generate the Figures shown in this paper, we executed the following commands using our MLSCAlib. Here, "AES_nRF.h5", "aes_hd.h5" and "dpav4.h5" refer to the file names of our custom dataset, the AES_HD dataset and the DPA contest V4.2 dataset respectively. Note that the following commands exclude the arguments for the database folder and the results path folder.

### Figure with dim1 for train and dim0 for val

When using `----dim 3`, the model will use dimension 1 for training and 0 for inference.

`python main.py -m mlp -e 15 -v 1 --fa AES_nRF.h5 --dim <0|1|3> --lm HW --loc SBox -b 3`



### Figures on the logits' behavior depending on dimension / optimizer with HW

Launch the following multiple times by varying the dimension.

`python main.py -m cnn_exp -e 30 -v 1 --fa AES_nRF.h5 --dim <0|1> --lm HW --loc SBox --record-axis logit --record-trials 10 --rec HW_comparison_dim<0|1> -b 3 -o RMSprop`

### Figures on the logits' behavior depending on dimension / optimizer with ID
Again, to be launched with varying dimensions.

`python main.py -m cnn_exp -e 30 -v 1 --fa AES_nRF.h5 --dim <0|1> --lm ID --loc SBox --record-axis logit --record-trials 10 --rec ID_comparison_dim<0|1> -b 3 -o RMSprop`

### Figure in the Experiments, profiled attack on AES HD

Run the following multiple times and vary the model and dimension arguments. The result will be combined in the same graph as the ----rec argument stays constant.

`python main.py -m <cnn_exp|cnn_aeshd> -e 100 -v 1 --fa aes_hd.h5 --lm HW --dim <0|1> --rec HW_aeshd --record-axis epochs --na 2500 --record-trials 50 --loc LastSBox -b 0 --nt 47500`

### Figure in the Experiments, profiled attack on DPAContestV4.2

`python main.py -m mlp -e 25 -v 1 --fa dpav4.h5 --dim <0|1> --lm HW --rec HW_dpa --record-axis epochs --nt 4500 --na 500 --record-trials 50 --loc SBox -b 0`

### ASCAD Results

The .sh scripts in `Scripts` named `SMOTE_SOTA.sh,ascad_experiments.sh,ascad_variable copy.sh,ascad_variable.sh,ascad_variable_id.sh,logits_box.sh,logits_muller.sh,prof.sh,test_models.sh`, contain the different command line calls used to compute the ASCAD results.
### Unprofiled results for AES HD

Here, the ----nt 0 argument specifies that we should use no profiling traces, and hence launch an unprofiled attack.

`python main.py -m cnn_exp -e 15 --fa aes_hd.h5 --dim <0|1> --lm HD -b 0 --nt 0 --loc LastSBox --na 50000 --cpu --with-regularization`
