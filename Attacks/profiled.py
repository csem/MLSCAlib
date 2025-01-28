"""
MLSCAlib
Copyright (C) 2025 CSEM 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#
# brief     : ML profiled attack on AES
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
import heapq
from math import ceil
import os
from time import gmtime, sleep, strftime, time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import traceback
from Data.custom_datasets import NRFDataset
from Data.data_manager import DataManager
import torch 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from Attacks.attack import  DEFAULT_SGD_MOMENTUM, Attack, SensitivityMode, TrainingMethods
from Ciphers.AES_leakage import AESLeakageModel
from error import Error

class Profiled(Attack):
    """Profiled attack class.
    
    Profiled attacks use a profiling set acquired on a clone device on known key/ptx.
    """
    def __init__(self,epochs=15,model_name="mlp", batch_size=100,loss="nlll",\
                    optimizer="Adam",leakage_model=AESLeakageModel("ID","SBox"),\
                    verbose=1,lambdas = None,dk=False,noise=False,seed=None,\
                     data_manager=DataManager(file_name_attack = 'file_name_attack.h5',na = "all",ns = "all",nt = "all",fs=0,\
                        databases_path="/path/to/Databases/"),results_path="/path/to/PlotsResults/",training = TrainingMethods.DEFAULT,\
                            info="",dim=1,threshold = 0.4,LTH=0.8,lr_schedule=None,learning_rate="default"):
        """ Initializes an Attack.

        Parameters
        ----------
        epochs : int, default : 15
                how many epochs to train the NN.
        model_name : str, default : mlp
                Which NN model to use. 
        batch_size : int, default : 100
                Batch size used during training.
        loss : str, default : nlll
                Loss function to use. Can be mse, cross_entropy or nlll.
        optimizer : str, default : Adam
                Can be Adam or SGD.
        leakage_model : Ciphers.LeakageModel, default : ID(SBox)
                A LeakageModel class used for label computations.
        verbose : int, default : 1
                What should be printed out during execution ? 0 : only the result,
                1 : the different metrics (e.g. remaining time, sensitivities, accuracies),
                2 : output debug informations too.
        lambdas: List[Float], default : None
                If not None, specifies the L1 & L2 regularization terms of two layers.
                The first two values (by default 0.03,0.03) are the L1 and L2 regularization 
                for a model's layer whose name is "regularized\_function\_1". The next two values 
                are the L1 & L2 regularization for any layer in the model whose name contain
                "regularized\_function\_".
        dk : bool, default : False
                Whether to use Domain Knowledge neurons. Those add plaintext information
                in the MLP part of the network.
        noise : bool, default : False
                Whether to add gaussian noise to the input of the training data.
        seed : int, default : None
                Which seed to use. If set to None, will use a default standard value, 5437.
        data_manager : Data.DataManager, default : DataManager()
                The datamanager handles anything related to the data.
        results_path : str, default : "/path/to/PlotsResults/"
                Where to store the learning plots.
        training : attacks.attack.TrainingMethods, default : Trainingmethods.DEFAULT
                Which training method to use.
        info : str, default : ""
                A small text to insert in the result file name.
        dim : int, default : 0
                On which dimension/axis to apply the softmax filter in the NN.
        threshold : float, default : 0.4
                When the certainty is higher than this threshold, we assume the guess is correct.
        LTH : float, default : 0.8
                Percentage of the weights to mask in the LTH pruning. Only used if the training
                method is set to TrainingMethods.PRUNING or TrainingMethods.PRUNING_HALF_EPOCHS.
        lr_schedule : List[int,float] | None. Default : None.
                learning rate Scheduler parameter. If not None, after each lr_schedule[0] epochs,
                multiply the learning rate by lr_schedule[1].
        """
        super().__init__(model_name=model_name,batch_size=batch_size,loss=loss,optimizer=optimizer,\
                            leakage_model=leakage_model,verbose=verbose,lambdas=lambdas,dk=dk,noise=noise,\
                                seed=seed,data_manager=data_manager,results_path=results_path,training=training,info=info,dim=dim,\
                                    threshold=threshold,lr_schedule=lr_schedule,learning_rate=learning_rate)
        self.epochs=epochs
        self.sensitivity_mode = SensitivityMode.CLASSIC
        self.LTH = LTH
        self.compute_logits=False  #used by record_attack to record logits
        self._printd(self)

    def get_pruning_percentage(self):
        return self.LTH
    def _prepare_figures(self):
        """Instantiates the plots for the results metrics."""
        fig=plt.figure(figsize=(15,15))
        plt.subplots_adjust(hspace=0.7)
        plot_sensi = plt.subplot2grid((8, 2), (0, 0), rowspan=2,colspan=2,fig=fig)
        plot_sensi.set_title("Attack Sensitivity")
        # plot_sensi.set_xticks(list(range(self.data_manager.get_ns(self.leakage_model))))
        plot_acc = plt.subplot2grid((8, 2), (2, 0), rowspan=2, colspan=2,fig=fig)
        plot_acc.set_title("Validation Accuracy")
        plot_acc_train = plt.subplot2grid((8, 2), (4, 0), rowspan=2, colspan=2,fig=fig)
        plot_acc_train.set_title("Training Accuracy")

        plot_extra=plt.subplot2grid((8,4),(6,0),rowspan=2,colspan=1,fig=fig)
        plot_extra.set_title("Confusion Matrix")
        plot_extra_val=plt.subplot2grid((8,4),(6,1),rowspan=2,colspan=1,fig=fig)
        plot_extra_val.set_title("Confusion Matrix Val")

        plot_GE = plt.subplot2grid((8, 4), (6, 2), rowspan=2, colspan=2,fig=fig)
        plot_GE.set_title("Fast GE")

        return plot_sensi,plot_acc,plot_acc_train,plot_GE,[plot_extra,plot_extra_val]
            


    def _prepare_figures_confusions(self):
        """EXPERIMENTAL: Instaniates 16 small plots, used when verbose = -1, to plot the 
        confusion matrices of different batches. With verbose =7 and LM=ID, plots the training &
        validation confusion matrices splitted by HW."""
        fig=plt.figure(figsize=(15,15))
        plt.subplots_adjust(hspace=0.7)
        if(self.verbose == 3):
            titles = ["Key Confusion","Key Confusion Val"]
        elif(self.verbose == 4):
            titles = ["Training - ID","Traning - HW 1","Training - HW 2","Training - HW 3",\
                "Training - HW 4","Training - HW 5","Training - HW 6","Training - HW 7","Eval - ID",\
                    "Eval - HW 1","Eval - HW 2","Eval - HW 3","Eval - HW 4","Eval - HW 5","Eval - HW 6","Eval - HW 7"]
        plot_extra00=plt.subplot2grid((8,4),(0,0),rowspan=2,colspan=1,fig=fig)
        plot_extra01=plt.subplot2grid((8,4),(0,1),rowspan=2,colspan=1,fig=fig)
        plot_extra02=plt.subplot2grid((8,4),(0,2),rowspan=2,colspan=1,fig=fig)
        plot_extra03=plt.subplot2grid((8,4),(0,3),rowspan=2,colspan=1,fig=fig)


        plot_extra10=plt.subplot2grid((8,4),(2,0),rowspan=2,colspan=1,fig=fig)
        plot_extra11=plt.subplot2grid((8,4),(2,1),rowspan=2,colspan=1,fig=fig)
        plot_extra12=plt.subplot2grid((8,4),(2,2),rowspan=2,colspan=1,fig=fig)
        plot_extra13=plt.subplot2grid((8,4),(2,3),rowspan=2,colspan=1,fig=fig)


        plot_extra20=plt.subplot2grid((8,4),(4,0),rowspan=2,colspan=1,fig=fig)
        plot_extra21=plt.subplot2grid((8,4),(4,1),rowspan=2,colspan=1,fig=fig)
        plot_extra22=plt.subplot2grid((8,4),(4,2),rowspan=2,colspan=1,fig=fig)
        plot_extra23=plt.subplot2grid((8,4),(4,3),rowspan=2,colspan=1,fig=fig)


        plot_extra30=plt.subplot2grid((8,4),(6,0),rowspan=2,colspan=1,fig=fig)
        plot_extra31=plt.subplot2grid((8,4),(6,1),rowspan=2,colspan=1,fig=fig)
        plot_extra32=plt.subplot2grid((8,4),(6,2),rowspan=2,colspan=1,fig=fig)
        plot_extra33=plt.subplot2grid((8,4),(6,3),rowspan=2,colspan=1,fig=fig)

        res = [\
            plot_extra00,plot_extra01,plot_extra02,plot_extra03,
            plot_extra10,plot_extra11,plot_extra12,plot_extra13,
            plot_extra20,plot_extra21,plot_extra22,plot_extra23,
            plot_extra30,plot_extra31,plot_extra32,plot_extra33]

        for i,r in enumerate(res):
            r.set_title(titles[i % len(titles)])
        return res

    def attack_bytes(self,byte_range,get_fast_GE = False):
        """Launches the attack_byte() method on each given byte in the byte_range.

        Parameters
        ----------
        byte_range : List[int] | int
                On which byte(s) to launch a SCA.
        get_fast_GE : bool, default : False
                whether to compute & return the fast Guessing Entropy over each epoch.
        
        Returns
        -------
        key_ranking : List[int]
                The key ordered in decreased order of probability
        key_certainty :  float
                An estimated probability that GE = 1.
        fast_GEs : List[int] | None
                If get_fast_GE is True, the guessing entropy of each epoch. 
        """
        if(isinstance(byte_range,int)):
            byte_range=[byte_range]
        res = []
        for byte in byte_range:
            self._printd(f"Attacking byte {byte}...")
            res.append(self.attack_byte(byte,fast_GE=get_fast_GE))
        return res
    def attack_byte(self,byte,get_output_probas=False,split_rate = 0.95,fast_GE = False,get_fast_ge = False,fast_sensi = False):
        """Attack byte: launches a full attack on a byte. 

        Will print the result of the attack on the terminal.

        Parameters
        ----------
        byte : int
                byte number to attack. Some bytes are harder to attack.
        get_output_probas : bool, default : False
                Whether to return the output probabilities of each 256 keys. If
                the attack is blind, the function will return them anyway.
        split_rate : float, default : 0.95
                In a blind attack scenario: which percentage of the profiling
                traces to use for training. The rest will be used as a validation
                set. In the other scenario, this is useless as the attack traces
                will be used for validation and all the profiling traces for training.
        fast_GE: bool, default : False
                Whether to compute the fast Guessing Entropy at each epoch during 
                training.
        get_fast_ge : bool, default : False
                Whether to return the fast Guessing Entropy. (Note that even when set
                to False, the GE may appear on the result pdf.)
        fast_sensi : bool, default : False
                Whether to compute the value of the maximum peak of the sensitivity at 
                each epoch during training. Only returned if get_fast_ge is True.
        
        Returns
        -------
        res_key_probas | ge,min_ge1,(fast_GEs,fast_sensis)
                If the attack is blind, returns the key probabilities.
                Otherwise, returns the Guessing entropy, the minimum amount of attack
                traces needed to reach GE=1 if possible, the fast GEs and the sensitivity
                peaks if demanded.
        """
        self._printd("In attack byte profiled")
        if(self.verbose ==3 or (self.verbose==4 and self.leakage_model.is_ID())):
            res = self._prepare_figures_confusions()
        elif(self.verbose==1 or self.verbose ==2):
            plot_sensi,plot_acc,plot_acc_train,plot_GE,[plot_extra,plot_extra_val] = self._prepare_figures()
        # x_profiling_res,label_profiling,x_val,label_val,plaintext_val,mask_val,x_attack_res,label_attack,self.plaintext_attack,self.key_attack,self.mask_att
        x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,x_attack,\
            label_attack,plaintext_attack,key_attack,mask_attack = self.data_manager.prepare_profiled_data(byte,leakage_model = self.leakage_model ,\
                                            use_attack_as_val = None,split_rate=split_rate,dk=self.dk)
        self._printdw("NT:",len(x_profiling),", NA:",len(x_attack),", NS:",self.data_manager.get_ns(self.leakage_model))
        model,accuracies,fast_GEs,train_acc=self.train_model(x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,key_attack,byte,None,True,fast_GE,sensi_reg=False,fast_sensi=fast_sensi)
        if(fast_sensi):
            fast_GEs,fast_sensis = fast_GEs
        if(self.verbose==1 or self.verbose == 2):
            #plot_sensi.plot(sensitivit,linewidth=1,alpha=0.4)#,label="Max Random "+str(sum_sensi)+sum_abs_sensi)
            plot_acc.plot(accuracies,'g',linewidth=1,alpha=0.4)#,label="Random "+str(sum_acc))
            plot_acc_train.plot(train_acc,'g',linewidth=1,alpha=0.4)
            plot_GE.plot(fast_GEs,'g',linewidth=1,alpha=0.4)
        # self._printw("WARNING ! PREDICTIONS ARE SYNTHETIC")
        # output_probas = model(x_attack,label_attack)
        output_probas  = self._batch_predict(model,x_attack,detach=True).detach().cpu()
        torch.save(output_probas,"/local/user/ldm/Data/CiC/results/testWeight/output_probas1.pt")
        torch.save(label_attack,"/local/user/ldm/Data/CiC/results/testWeight/label_attack1.pt")

        print("output_probas",output_probas)
        # if(self.model_name !="mesh"):
        #     if(not os.path.exists(os.path.dirname(__file__) + "/"+f"Models/")):
        #         os.mkdir(os.path.dirname(__file__) + "/"+f"Models/")
            # torch.save(model.state_dict(),os.path.dirname(__file__) + "/"+f"Models/state_{self.model_name}_dim{self.dim}_{str(self.leakage_model)}_nrf.pt")
        if(self.verbose ==4 and self.leakage_model.is_ID()):
            #Confusion matrices of ID attack, separated by HW value id_in_hw_visualize
            model.train()
            self._draw_class_predictions(res,model,x_profiling,label_profiling,id_in_hw_visualize=True)
            model.eval()
            self._draw_class_predictions(res[8:],model,x_attack,label_attack,id_in_hw_visualize=True)

        elif(self.verbose == 3):
            model.train()
            self._draw_class_predictions(res[0],model,x_profiling,label_profiling)
            model.eval()
            self._draw_class_predictions(res[1],model,x_attack,label_attack)
            batch_size_drawing = self.batch_size
            for i,r1 in enumerate(res[2:]):
                self._draw_class_predictions(r1,model,x_attack[batch_size_drawing*i:batch_size_drawing*(i+1)],label_attack[batch_size_drawing*i:batch_size_drawing*(i+1)])
        elif(self.verbose==1 or self.verbose == 2):
            model.train()
            self._draw_class_predictions(plot_extra,model,x_profiling,label_profiling)
            model.eval()
            self._draw_class_predictions(plot_extra_val,model,x_attack,label_attack)
        
        ########################## Try random label guesses ? #######
        #output_probas = np.ones_like(output_probas) * np.array([0,0,0,0,1,0,0,0,0])
        #output_probas = np.ones_like(output_probas) * to_categorical(np.random.choice(9,size = (len(output_probas),),p = (1/256,8/256,28/256,56/256,70/256,56/256,28/256,8/256,1/256)))
        #output_probas = np.ones_like(output_probas) * to_categorical(np.random.choice(256,size = (len(output_probas),)))#,p = (0/256,0/256,0/256,79/256,98/256,79/256,0/256,0/256,0/256)),9)
        ########################## End ####################################
        
        output_probas_on_keys = self.turn_output_probas_to_key_proba(output_probas=output_probas,plaintexts=plaintext_attack,byte_position = byte,masks=mask_attack)
        # print("output_probas_on_keys",output_probas_on_keys)
        self._end_record()
        if(not self.data_manager.blind):
            ### PART TO COMPARE THE GE % FOR DIM 0 AND DIM 1
            # probas = np.sum((output_probas_on_keys), axis=0)
            # guesses = heapq.nlargest(256, range(len(probas)), probas.take)
            # for i in range(len(guesses)-1):
            #     print("dim",self.dim,"on",i,"th key has proba:",np.exp(probas[guesses[i]]-probas[guesses[i+1]]))

            ge = Attack.get_GE(output_probas_on_keys,key_attack[0][byte])
            if(self.verbose==1 or self.verbose==2):
                plot_sensi.set_title("Attack Sensitivity")
                minimum = max(1,(len(x_attack)-1) - ((len(x_attack)-1) % self.batch_size))
                x_attack_sensitive,label_attack_sensitive = x_attack[:min(self.batch_size * 3,minimum)],label_attack[:min(self.batch_size * 3,minimum)]  #only use few attack traces to get the sensitivity
                x_attack_sensitive.requires_grad_()
                sensitivity=torch.abs(self.get_sensitivity_tensors(model,x_attack_sensitive,label_attack_sensitive)).detach().cpu().numpy()
                plot_sensi.plot(sensitivity,'r',linewidth=1,alpha=0.4)
            if(self.verbose != 0):
                try:
                    plt.savefig(self._get_res_name(byte,ge))#self.results_path+f'profiled,byte_{byte},{self.leakage_model},batch_{self.batch_size},epochs_{self.epochs},Na_{self.data_manager.na},Ns_{self.data_manager.ns},{self.loss},{self.model_name},'+str(self.lambdas).replace(" ","")+'.pdf')
                except:
                    print("Error while saving the metrics. Could not save ",self._get_res_name(byte,ge))

            print("Guessing entropy = ",ge," for byte",byte)
            if(ge==1):
                min_ge1=Profiled.get_min_trace_for_GE1(output_probas_on_keys,key_attack[0][byte])
                print("Minimum attack traces to reach GE = 1: ",min_ge1," for byte",byte)
            else:
                min_ge1= None
            if(not get_output_probas):
                if(get_fast_ge):
                    if(fast_sensi):
                        return ge,min_ge1,fast_GEs,fast_sensis
                    else:
                        return ge,min_ge1,fast_GEs
                else:
                    return ge,min_ge1
            else:
                if(get_fast_ge):
                    self._printw("Cant't return fastGE")
                return ge,min_ge1,output_probas_on_keys
        else:
            res_key_probas = Attack.get_keys_from_probas(output_proba_on_key=output_probas_on_keys)
            if(self.verbose==1 or self.verbose ==2):
                x_attack.requires_grad_()
                sensitivity=torch.abs(self.get_sensitivity_tensors(model,x_attack,label_attack)).detach().cpu().numpy()
                plot_sensi.set_title(f"Attack Sensitivity - Byte {byte} - Best key guesses {res_key_probas[0][0]},{res_key_probas[0][1]}  & {res_key_probas[0][2]}")
                plot_sensi.plot(sensitivity,'r',linewidth=1,alpha=0.4)
            if(self.verbose !=0):
                plt.savefig(self._get_res_name(byte,ge="blind"))
            print(f"Byte {byte} attack result: ",res_key_probas)
            return res_key_probas

    def _get_res_name(self,byte,ge):
        return super()._get_res_name(f"profiled,byte{byte},GE_{ge}")
    def _get_super_name(self,info=""):
        return super()._get_res_name(info)
    def train_model_adversarial(self):
        """Performs an adversarial learning. Not implemented yet."""
        raise NotImplementedError("")
    def train_model_cross_validation(self):
        """Performs a learning using cross-validation techniques. Not implemented yet."""
        raise NotImplementedError("")

    def train_model_default(self,x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,key_attack,\
                                    byte,model=None,get_metrics=False,fast=False,sensi_reg = False,fast_sensi = False):
        """ Trains the model.

        Parameters
        ----------
        x_profiling : List
                The attack traces used for unprofiled training. Should have the shape (na,1,ns).
        label_profiling : List[int]
                The labels corresponding to x_attack.
        x_validation : List
                Traces to use for metrics computations (e.g. sensitivity computation). Can be
                the same as x_attack.
        label_validation : List[int]
                The labels corresponding to x_attack_val.
        plaintext_val ; List[int]
                The plaintexts corresponding to x_validation
        key_attack : List[List[int]]
                The attack key(s). 
        byte : int
                Target byte index.
        model : nn.Module, default : None
                If not set to None, will re-use the model given instaed of creating a new.
        get_metrics : bool, default : True
                Whether to calculate validation accuracy and training accuracy.
        fast : bool, default : True
                Whether to calculate the fast Guessing Entropy during learning. May induce
                additionnal delays. 
        sensi_reg : bool, default : False
                Whether to apply a sensitivity regularization. This is not implemented yet
                for profiled attacks.
        fast_sensi : bool, default : False
                Whether to return, for each epoch, the value of the highest sensitivity peak
                computed on the vlidation data.

        Returns
        -------
        model : torch.nn.Module
                The trained model.
        """
        ACCURACY_SIZE = None
        if(model == None):
            model = self._get_model()
        if(self.compute_logits):
            self._printdw("WARNING: no fastGE but STD logit instead")
            model.set_dim("no")
        
        if(sensi_reg):
            self._printw("Discarding the sensitivity regularization as it is not implemented in profiled attacks yet.")
        
        loss_fn,label_profiling = self._get_loss_label(label_profiling)
        if(self.model_name == "mesh"):
            start = 0
            for name,_ in list(model.named_parameters()):
                if("first_model" in name):
                    start += 1
        else:
            start=0
        #     optimizer = torch.optim.Adam(list(model.parameters())[start:], lr=learning_rate)
        # elif(self.optimizer=="SGD"):
        #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=DEFAULT_SGD_MOMENTUM) #momentum 0.5 0.9 0.99
        optimizer = self._get_optimizer(model,start = start,learning_rate=self.learning_rate)
        accuracies=list()
        train_acc=list()
        fast_GEs=list()
        fast_sensis=list()
        data_set=NRFDataset(x_profiling,label_profiling,device = self.data_manager.device)
        train_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle = self.seed is None)
        if(self.lr_schedule is not None):
            # scheduler = StepLR(optimizer, step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])
            scheduler = OneCycleLR(optimizer=optimizer,epochs=self.epochs,max_lr= 0.002,total_steps=None,steps_per_epoch=ceil(self.data_manager.nt/self.batch_size)\
                 ,pct_start=0.3,anneal_strategy='linear',cycle_momentum=False,base_momentum=0.85,max_momentum=0.95,div_factor=25,final_div_factor=1e4,three_phase=True\
                    ,last_epoch=-1,verbose=self.verbose==2)#,step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])
        self._start_record()
        for t in range(self.epochs):
            model.train()
            # print("sum_std",sum_std)
            for i,(x,y) in enumerate(train_loader):
                if(i==0 and self.compute_logits):
                    model.eval()
                    model.set_dim("no softmax")
                    logits = model(x)
                    fast_GEs.append(torch.mean(logits,0).detach().cpu().numpy())
                #     # print("Appending",torch.sum(logits,0))
                    model.set_dim(self.dim)
                    model.train()
                y_pred = model(x)
                
                # if(self.compute_logits):
                #     if(i==0):
                #         fast_GEs.append(torch.mean(y_pred,0).detach().cpu().numpy())
                #     y_pred = _log_softmax(self.dim,True)(y_pred)
                if(self.loss=="mse"):
                    y_pred=torch.exp(y_pred)
                # print("y is ",y )
                # print("y pred is ",y_pred)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()

                # Compute a regularization, different for each layer 
                if(self.lambdas!=None):
                    penalty=0.0
                    for name,p in model.named_parameters():
                        if("mask" in name):
                            continue
                        if(name=="regularized_function_1.weight"):
                            penalty+= self.lambdas[0] * sum((w.abs().sum() for w in p )) + self.lambdas[1] * sum((w.pow(2.0).sum()) for w in p) 
                        elif(name=="regularized_function_2.weight"  or ("regularized_function_" in name and "weight" in name)):
                            penalty+= self.lambdas[2] * sum((w.abs().sum()) for w in p) + self.lambdas[3] * sum((w.pow(2.0).sum()) for w in p)
                    loss+=penalty
                loss.backward()
                # for name,w in list(model.named_parameters()):
                #     if("regularized_function_2" in name and "weight" in name):
                #         # print("at epoch",t,"batch",i,name,"grad is "," of shape ",w.shape,w[0].shape,torch.mean(w,1).shape)
                #         fast_GEs.append(torch.mean(w.grad,1))
                #         # print(torch.mean(w,1))
                #         pass
                optimizer.step()
                #print("loss is ",float(loss))
                del loss
            if(self.lr_schedule is not None):
                # In case you use the OneCycleLR, put the scheduler step inside the batch loop.
                scheduler.step()
            model.eval()
            if(self.verbose ==1 or self.verbose ==2):
                #Watch out ! If the validation is in GPU, and is too big, it may be too big for the GPU RAM 
                accuracies.append(self.get_accuracy(self._batch_predict(model,x_validation[:ACCURACY_SIZE]),label_validation[:ACCURACY_SIZE]))
                train_acc.append(self.get_accuracy(self._batch_predict(model,x_profiling[:ACCURACY_SIZE]),label_profiling[:ACCURACY_SIZE]))
            if(fast and not self.data_manager.blind and not self.compute_logits):
                # self._printw("Set to use all of the validation traces !")
                fast_GEs.append(self.get_fast_GE(model=model,traces=x_validation,plaintext_attack=plaintext_val,byte=byte,key_byte=key_attack[0][byte],fast_size=5000000,masks=mask_val))

            if(fast_sensi):
                fast_sensis.append(float(torch.max(self.get_sensitivity_tensors(model,x_validation,label_validation)).cpu()))
            if(t==0):
                self._round_record(self.epochs)

        if(fast_sensi):
            fast_GEs = (fast_GEs,fast_sensis)
        return model,accuracies,fast_GEs,train_acc 

    @staticmethod
    def get_min_trace_for_GE1(output_proba_on_key,key_byte,step=1):
        """Computes the minimum number of traces required to get GE = 1.

        Iteratively considers more and more attack traces until it is possible
        to reach a Guessing Entropy of 1. You should consider taking the mean
        over multiple execution to get a meaningfull result.

        Parameters
        ----------
        output_proba_on_key : List[List[float]]
                List of probabilities sorted on the secret key for each plaintext.
        key_byte : int
                The value of the target key byte.
        step : int, default : 1
                Defines by how much to increase the number of attack traces used until reaching GE=1
                between each trial.
        Returns
        -------
        int
                Minimum amount of attack traces required to reach a Guessing Entropy of 1.
        """
        k=1
        while(np.argmax(np.sum(output_proba_on_key[:k],axis=0)) != key_byte and k<=len(output_proba_on_key)):
            k=k+step
        if(np.argmax(np.sum(output_proba_on_key[:k],axis=0)) != key_byte):
            return -k
        else:
            return k


    def __str__(self):
        return "Profiled SCA Attack on AES with"+ super().__str__()

    def record_attack(self,byte=3,number_of_trials = 10,info_for_filename = "",info_for_plot = "",x_axis = "epoch",\
                        redraw_this = None,legend_location="inside",overwrite_title=None,target_model_substring="",\
                            target_model_without_substring=None,ncols = 1,plot=False,custom_palette=None,logit=False,\
                                replace_legend=(lambda x:x),font_size = 10,font_weight='normal',file_extension = "pdf"):
        """Creates a graph that allows to compare the performance of different models.

        The graph created will plot, for each model, the mean value of the Key Rank along with
        a 95% confidence interval. Each time this function is being executed, it will store the result in the 
        Results/ folder (inside the directory of this file). The naming of the temporary results (stored in a *.npy*) is 
        *{info_for_filename}-AXIS-{x_axis'}-FAST_GE_epochs-{self.epochs},byte-{byte},{self.data_manager.get_res_name()}.npy*.
        Hence, when the user calls this function twice with the same *info_for_filename*, *x_axis,epochs*, *byte* and the same
        file (with na/ns/nt), it will combine the results into a same file. The graph will have a different line
        for each model, where info_for_plot allows an additional model information addition. (i.e. using MLP with 
        "label : HW" will have a different line on the graph than MLP with "label : ID" as info_for_plot). 
        Warning: adding noise with CustomDataManager and using *number_of_trials = 0* will draw a graph with incorrect title.
        Warning: Currently, the byte number should always stay the same between records of the same plot.

        Parameters
        ----------
        byte : int, default :3
                Target byte number.
        number_of_trials : int, default : 10
                How many attack_byte() runs to do. The graph will plot the mean + 95% confidence interval.
                If set to 0, it will generate the graph again, provided that the corresponding .npy file
                is available in Results/ .
        info_for_filename : str, default : ""
                Determines a specific result-plot identifier. Each plot has a unique identifier. This means, 
                running this function multiple times with arguments leading to the same identifier will
                result in a single plot (and different lines iff the model name / info_for_plot is different).
        info_for_plot : str, default : ""
                When running this function, the Profiled class has a model name stored. This argument will add
                a specific information in the legend of the graph alongside the model name. This for example
                allows to have different lines for the same model but varying parameters.
        x_axis : str, default : "epoch"
                Along which x-axis to plot the graph. Can be either "na", "nt", "epoch" or "epoch_sensi". "na"-axis
                will have thex-axis be the number of attack traces (hence the computation will incrementally change 
                the number of attack traces considered when doing the attack. na < 11, it will mean 100 trials. For 
                10<na<2400, it will mean 10 trials and otherwise just 1 trial. This is as such because to avoid pure 
                luck: using 1 (or few) attack trace may give the correct/wrong key just by luck.) Using "nt"-axis, the
                x-axis will be along the number of profiling traces. For each index, the a whole computation is done with
                the corresponding number of profiling traces (takes time). When using the "epoch"-axis, the x-axis 
                represents the number of epoch. This can be seen as a graph of fast-Guessing Entropy which uses
                all of the validation traces instead of only a subset. When using "epoch_sensi", the function will produce
                two pdf plots, each of them with epoch as x-axis. The first will have the key-rank as y-axis and the second
                will have the highest sensitivity value for the given epoch.
        redraw_this : str, default : None
                Path to a .npy file containing the result of a previous recording that you wish to draw again without having
                to give all the exact same parameters. The function will automatically infer the x-axis used.
        legend_location : str, default : "inside"
                Where to put the legend. Can be any of inside, bottom, upper left or absent.
        overwrite_title : str, default : None
                By default (None), will infer the title from the current Profiled class details. If set to empty string,
                will not put the title. You can also specify any title you want.
        target_model_substring : str, default : ""
                Only plots the models containing target_model_substring in their name.
        target_model_without_substring : str, default : None
                If not None (or ""), will only plot the models which do not contain target_model_without_substring in
                their name.
        ncols : int, default : 1
                On how many columns to spread the legend.
        plot: bool, default: False
                Whether to plot the figure in addition to saving it in a pdf.
        custom_palette: List[List[float]], default : None
                When given, will plot the epoch graph according to the given palette. It should be of size Mx3, where M
                is the number of models remaining after the selection via target_model_substring and 
                target_model_without_substring. Each have a color specified in RGB with 3 floats.
        logit: bool, default: False
                Whether to plot the logit instead of the fast Guessing Entropy. The x-axis should be set to epoch.
        replace_legend: str -> str, default: lambda x:x
                Function to change the legend element wise. For example, if the title of the plot is "MLP model with ID
                labelling for different optimizers", removing the "ID" or "MLP" mention in the labels may be useful.
        font_size: int, default: 10
                The font size to use in the legend and axis.
        font_weight: str, default: normal
                Can be normal, bold, light.
        file_extension: str, default: "pdf"
                The resulting file extension of the recording. Can be for example 'png', 'pdf', 'svg', 'jpeg', ...
        Raises
        ------
        Error
                When number_of_trials is set to 0 and no previous trials have been done with the same arguments.
        Error
                When the target_model_(without)_substring specification(s) disable(s) all the models. In that case, the names
                of all the available models will be printed in the console.
        
        """
        if(target_model_without_substring == ""):
            target_model_without_substring = None
        reuse_old_result = False  #only for previous compat. set this to True to re-plot old plots with old results
        assert not(logit and (not "epoch" in x_axis))
        assert not (reuse_old_result and (number_of_trials != 0 or "sensi" in x_axis))
        assert not (redraw_this is not None and number_of_trials != 0)
        if(redraw_this is not None and x_axis == "epoch"):
            if("AXIS-na" in redraw_this):
                self._print("Setting the record x-axis to na")
                x_axis = "na"
            elif("AXIS-nt" in redraw_this):
                self._print("Setting the record x-axis to nt")
                x_axis = "nt"
            elif(not reuse_old_result):
                self._print("Setting the record x-axis to sensi.")
                x_axis = "epoch_sensi"
        model_name = self.model_name
        results_path = self.results_path #"/path/to/PlotsResults/"
        path = os.path.dirname(__file__) + "/" +"Results/"
        if("epoch" in x_axis):
            self._printdw("Using epoch axis")
            number_x_axis = list(range(self.epochs))#* ceil(47500 / self.batch_size)))
            if(logit):
                self.compute_logits=True
        elif("na" in x_axis):
            self._printdw("Using NA axis")
            if(self.training == TrainingMethods.DEFAULT):
                self.train_model = self._train_model_record
            else:
                self.train_model_default = self._train_model_record
            number_x_axis = list(range(1,self.data_manager.na+1)) #list([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1500,2000,2500])
        elif("nt" in x_axis):
            self._printdw("Using NT axis")
            number_x_axis = list([100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4500,5500,6500,7500,12500,17500,22500,27500,37500,47500])
            self.attack_byte = self._attack_byte_record
        for i in range(number_of_trials):
            time0 = time()
            compute_fast_GE = True

            if("sensi" in x_axis):
                ge,min,fast_GE,fast_sensis = self.attack_byte(byte,fast_GE=compute_fast_GE,get_fast_ge=True,fast_sensi=True)
            else:
                ge,min,fast_GE = self.attack_byte(byte,fast_GE=compute_fast_GE,get_fast_ge=True,fast_sensi=False)
                fast_sensis = np.zeros_like(fast_GE)
            if(self.verbose!=0):
                plt.close()
            time1 = time()
            if(i==0):
                info="Will finish all the trials at max: "+strftime('%d-%m-%Y  %H:%M:%S', gmtime( 60*60 * 2 +time() + (number_of_trials-1)*(time()-time0)))
                print("Finished record round in "+self._get_duration(time0)+".",info )
            if(self.compute_logits):
                # self._printw("REmove STD comparison here")
                fast_GE=np.array(fast_GE)#[np.array(x.detach().cpu()) for x in fast_GE])
                # print(fast_GE)
                label_name = ""
                for c in range(self.leakage_model.get_classes()):
                    # print(fast_GE)
                    # print("Shape of fast_GE:",len(fast_GE),len(fast_GE[0]))
                    # print("Shape of np.array(fast_GE):",np.array(fast_GE).shape)
                    fast_GE_tmp = fast_GE[:,c]
                    fast_sensis = np.zeros_like(fast_GE_tmp)
                    # print("fast_GE_tmp",fast_GE_tmp.shape)
                    label_name = f"_Label_{c}"
                    if(i==0 and c == 0):
                        self._round_record(number_of_trials)
                        res_array = np.array(list(zip(number_x_axis,fast_GE_tmp,fast_sensis,[model_name+info_for_plot+label_name]*len(fast_GE_tmp),[time1-time0]*len(fast_GE_tmp),[byte]*len(fast_GE_tmp))))
                    else:
                        tmp = np.array(list(zip(number_x_axis,fast_GE_tmp,fast_sensis,[model_name+info_for_plot+label_name]*len(fast_GE_tmp),[time1-time0]*len(fast_GE_tmp),[byte]*len(fast_GE_tmp))))
                        res_array = np.concatenate((res_array,tmp),axis=0)

            else:
                label_name = ""
                if(i==0):  #and c == 0): 
                    self._round_record(number_of_trials)
                    res_array = np.array(list(zip(number_x_axis,fast_GE,fast_sensis,[model_name+info_for_plot+label_name]*len(fast_GE),[time1-time0]*len(fast_GE),[byte]*len(fast_GE))))
                else:
                    tmp = np.array(list(zip(number_x_axis,fast_GE,fast_sensis,[model_name+info_for_plot+label_name]*len(fast_GE),[time1-time0]*len(fast_GE),[byte]*len(fast_GE))))
                    res_array = np.concatenate((res_array,tmp),axis=0)
        if("sensi" in x_axis):
            axis_info = "epoch"
        else:
            axis_info = x_axis
        if(reuse_old_result):
            inf = ""
        else:
            inf = "SENSI_"
        if(redraw_this is not None):
            res_name = redraw_this
        else:
            res_name = f"{info_for_filename}-AXIS-{axis_info}-FAST_GE_{inf}epochs-{self.epochs},byte-{byte},{self.data_manager.get_res_name(self.leakage_model)}.npy" 

        if(not os.path.exists(path)):
            os.mkdir(path)
        if(not os.path.exists(path + res_name)):
            load = False
            if(number_of_trials == 0):
                raise Error("You must first complete a record before using 0 record trials.")
        else:
            load = True
            #open(path + res_name,'a').close()
        i=0
        written = False
        while(not written):
            try:
                if(load):
                    old = np.load(file = path + res_name, allow_pickle = True)
                    if(number_of_trials==0):
                        new = old
                    else:
                        new = np.concatenate((old,res_array),axis=0)
                else:
                    new = res_array
                np.save(file = path + res_name, arr=new,allow_pickle = True)
                written=True
            except OSError:
                if(i>10):
                    raise Error("Error while saving the result")
                i+=1
                sleep(np.random.randint(0,50)/10)
        try:
            plt.clf()
            plt.rcParams.update({'font.size': font_size,
                                #  'font.weight':font_weight,
                                 "axes.titlesize":font_size,
                                 "axes.titleweight":font_weight})
            
            if(logit):
                y_axis_name = "Mean Logit for Each Class"
            else:
                y_axis_name = "Key Rank"
            if(reuse_old_result):
                new = np.array(sorted(new, key=lambda x : x[2]))
            else:
                new = np.array(sorted(new, key=lambda x : x[3]))  # Sort on model name for cross-graph consistency
            if(target_model_substring != ""):
                if(target_model_without_substring is not None):
                    info2 = f" and not \"{target_model_without_substring}\""
                else:
                    info2 = ""
                self._printw(f"Only plotting models containing \"{target_model_substring}\""+info2)
            def save_fig(fig,name_info = ""):
                ### Sort the resulting legend for cross-graph consistency
                handles, labels = plt.gca().get_legend_handles_labels()
                sorted_labels = sorted(labels)
                order = np.array([labels.index(sorted_label) for sorted_label in sorted_labels])
                handles = list(np.array(handles)[order])
                labels = list(np.vectorize(lambda x : replace_legend(x.replace("HW"," HW").replace("_"," ").replace("-"," ").replace("ID"," ID").replace("DIM","dim")\
                    .replace("dim"," dim").replace("dim ","dim").replace("  "," ")\
                        .replace("cnn exp","CNNexp").replace("cnn aeshd","CNNaeshd").replace("mlp_exp","MLPexp").replace("mlp","MLP").replace("vgg","VGG")\
                            .replace("mcnn","MCNN")))(sorted_labels))
                
                if(legend_location == "upper right"):
                    lg = plt.legend(handles=handles,labels=labels,bbox_to_anchor=(1.02, 1), loc='upper left',ncol=ncols)
                elif(legend_location =="inside"):
                    lg = plt.legend(handles=handles,labels=labels,loc="best",ncol=ncols)
                elif(legend_location == "bottom"):
                    if(x_axis == "na" or x_axis =="nt"):
                        lg = plt.legend(handles=handles,labels=labels,bbox_to_anchor=(-4.25, -0.08), loc='upper left',ncol=ncols) 
                    else:
                        lg = plt.legend(handles=handles,labels=labels,bbox_to_anchor=(0.5, -0.05), loc='upper left',ncol=ncols)  
                elif(legend_location == "absent"):
                    lg = plt.legend(handles=handles,labels=labels).set_visible(False)
                else:
                    raise Error("Illegal legend location. Use one of \"upper right\", \"inside\", \"bottom\" or \"absent\"")
                if(target_model_substring==""):
                    separator = ""
                else:
                    separator = "_"
                if(target_model_without_substring is None):
                    without = ""
                else:
                    without = "without_"+target_model_without_substring+"_"
                if(legend_location == "absent"):
                    fig.savefig(results_path+target_model_substring+without+separator+res_name.replace(".npy","_")+name_info+"."+file_extension,bbox_extra_artists=(lg))
                else:
                    fig.savefig(results_path+target_model_substring+without+separator+res_name.replace(".npy","_")+name_info+"."+file_extension,bbox_extra_artists=(lg,), bbox_inches='tight') 
                if(plot):
                    plt.show()
                plt.close()
                plt.clf()

            def draw(axes,numbers_per_axis,x_axis_name,data):
                max_y = ceil(np.max(data[:,1].astype(float)))

                def draw_sub_ax(dataframe,lab_axis,axis,title,index):
                    d_i = dataframe.loc[dataframe[x_axis_name].isin(lab_axis)]
                    plt_sns_i = sns.lineplot(data=d_i,x=x_axis_name,y=y_axis_name,hue="Model",ax=axis)
                    if(overwrite_title is None):
                        plt_sns_i.set(title=title)
                    elif(overwrite_title!=""):
                        plt_sns_i.set(title=overwrite_title)
                    if(index == len(axes)):
                        res = (lab_axis[0],lab_axis[1],lab_axis[len(lab_axis)//2],lab_axis[-1])
                    else:
                        res = (lab_axis[0],lab_axis[1],lab_axis[len(lab_axis)//2])
                    plt_sns_i.set_xticks(res)
                    plt_sns_i.set_xticklabels(res,rotation=45,horizontalalignment='right')
                    plt_sns_i.set(ylim=(0,max_y))
                    plt_sns_i.legend_.remove()
                    if(index != 1):
                        plt_sns_i.set(ylabel=None)
                        plt_sns_i.set(yticklabels=[])
                    if(index != ceil(len(axes)/2)):
                        plt_sns_i.set(xlabel=None)
                if(reuse_old_result):
                    d_init = pd.DataFrame(data,columns=[x_axis_name, y_axis_name, "Model", "Duration", "Byte"])
                else:
                    d_init = pd.DataFrame(data,columns=[x_axis_name, y_axis_name,"Sensitivity" , "Model", "Duration", "Byte"])
                d_init = d_init.astype({x_axis_name: int})
                d_init = d_init.astype({y_axis_name: float})
                
                d = d_init[d_init["Model"].str.contains(target_model_substring)] 
                if(target_model_without_substring is not None):
                    d = d[~d["Model"].str.contains(target_model_without_substring)]
                if(len(d)==0):
                    raise Error("Invalid target model substring. The saved model names are: ",d_init["Model"].drop_duplicates())
                titles = [f'{self.data_manager.get_res_name(self.leakage_model)}',"","on","byte",f'{byte}']
                for lab_axis,axis,title,index in zip(numbers_per_axis,axes,titles,range(1,len(axes)+1)):
                    draw_sub_ax(d,lab_axis,axis,title,index)
                


            if("epoch" in x_axis):
                fig = plt.figure(figsize=(8,7))
                ax = plt.axes()
                if(legend_location == "bottom"):
                    ax.set_xlabel("Epoch Number",loc="left")
                    ax.set_ylabel(y_axis_name,loc="bottom")
                
                """To reduce the maximal # of epochs to consider, uncomment:
                max_epochs_for_plot = 5
                tmp = new[:,0]
                tmp = tmp.astype(int)
                new = new[np.where(tmp<max_epochs_for_plot)]
                """
                if(reuse_old_result):
                    d_init = pd.DataFrame(new,columns=["Epoch Number", y_axis_name, "Model", "Duration", "Byte"])
                else:
                    d_init = pd.DataFrame(new,columns=["Epoch Number", y_axis_name ,"Sensitivity" , "Model", "Duration", "Byte"])
                # print("d_init",d_init)
                d_init = d_init.astype({"Epoch Number": int})
                # self._printw("REMOVE STD VARIANT HERE __ float to int")
                d_init = d_init.astype({y_axis_name: float})
                d1 = d_init[d_init["Model"].str.contains(target_model_substring)]
                if(target_model_without_substring is not None):
                    d1 = d1[~d1["Model"].str.contains(target_model_without_substring)]
                    # d1 = d1[~d1["Model"].str.contains("HW2")]# | d1["Model"].str.contains("HW_") ]
                    # print(d1[d1["Model"].str.contains("Adam")])
                if(len(d1)==0):
                    raise Error("Invalid target model substring. The saved model names are: ",d_init["Model"].drop_duplicates())

                # HW2_palette = [(0.5218824818744434, 0.8665028334663194, 0.0),
                #             (0.35298420091625415, 0.5513212923069918, 0.058284825465036924),
                #             (3.6944373735452454e-05, 0.8611353541381616, 0.9018874244983595),
                #             (0.05953934045546381, 0.5479299914869235, 0.5714272296462627)]


                # DK_palette = [[0.21568627, 0.63660131, 0.39869281],\
                #                 [0.91372549, 0.36862745, 0.05098039],\
                #                 [0.80522876, 0.1372549 , 0.55947712],\
                #                 [1.        , 0.66535948, 0.        ],\
                #                 [0.29166667, 0.29166655, 0.4057971 ],\
                #                 [0.41176461, 0.2604    , 0.16583333],\
                #                 [0.66666667, 0.87320261, 0.82614379],\
                #                 [0.99215686, 0.72679739, 0.49150327],\
                #                 [0.98300654, 0.67320261, 0.72418301],\
                #                 [1.        , 0.85359477, 0.06797386],\
                #                 [0.58333333, 0.6822916 , 0.70833327],\
                #                 [0.82352922, 0.5208    , 0.33166667]]

                # color_STD_ID = list(sns.color_palette('Wistia_r', as_cmap=False,n_colors=256)) + list(sns.color_palette('Blues_r', as_cmap=False,n_colors=256))
                # sns.set_palette(color_STD_ID) 
                # sns.set_palette(HW2_palette) 
                # HW_palette = sns.diverging_palette(115, 115, n=9,center="dark",s=100,l=92)
                # sns.set_palette(HW_palette)#sns.diverging_palette(260, 10, n=9,center="light"))  #sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))
                # sns.color_palette("bright")
                # sns.set_palette(sns.color_palette("bright",n_colors=12))
                # sns.set_palette(DK_palette)
                if(custom_palette is not None):
                    sns.set_palette(custom_palette)

                

                plt_sns = sns.lineplot(data=d1,x="Epoch Number",y=y_axis_name,hue="Model",ax=ax)
                if(overwrite_title is None):
                    plt_sns.set(title=f'Attacking {self.data_manager.get_res_name(self.leakage_model)} on byte {byte}')
                elif(overwrite_title!=""):
                    plt_sns.set(title=overwrite_title)
                # fig = plt_sns.get_figure()
                save_fig(fig)
                if("sensi" in x_axis):
                    plt.figure(figsize=(8,7))
                    d1 = d1.astype({"Sensitivity": float})
                    d1 = d1[d1["Sensitivity"]>0]  #only consider the trials which computed a sensitivity
                    if(len(d1)!=0):
                        plt_sns = sns.lineplot(data=d1,x="Epoch Number",y="Sensitivity",hue="Model")
                        if(overwrite_title is None):
                            plt_sns.set(title=f'Attacking {self.data_manager.get_res_name(self.leakage_model)} on byte {byte}')
                        elif(overwrite_title!=""):
                            plt_sns.set(title=overwrite_title)                        
                        fig = plt_sns.get_figure()
                        save_fig(fig,"SENSI")
            elif("nt" in x_axis):
                fig = plt.figure(figsize=(10,10))
                ax1 = plt.subplot2grid((2,11), (0, 0), rowspan=1,colspan=3,fig=fig)
                ax2 = plt.subplot2grid((2, 11), (0, 3), rowspan=1,colspan=4,fig=fig)
                # ax3 = plt.subplot2grid((2,11), (0, 5), rowspan=1,colspan=2,fig=fig)
                ax4 = plt.subplot2grid((2,11), (0, 7), rowspan=1,colspan=2,fig=fig)
                ax5 = plt.subplot2grid((2, 11), (0,9), rowspan=1,colspan=2,fig=fig)
                plt.subplots_adjust(wspace=0.1)

                lab_ax_1 = list([100,200,300,400,500,600,700,800,900,1000])  # 10
                lab_ax_2 = list([1000,1500,2000,2500,3000,3500,4500,5500,6500,7500]) # 5
                # lab_ax_3 = list([3000,3500,4500,5500,6500,7500]) # 6
                lab_ax_4 = list([7500,12500,17500,22500,27500]) # 5
                lab_ax_5 = list([27500,37500,47500]) # 3
                draw([ax1,ax2,ax4,ax5],[lab_ax_1 ,lab_ax_2,lab_ax_4,lab_ax_5],"Number of Profiling Traces",new)
                save_fig(fig)

            elif("na" in x_axis):
                # fig = plt.figure(figsize=(10,10))
                # ax1 = plt.subplot2grid((2,10), (0, 0), rowspan=1,colspan=3,fig=fig)
                # ax2 = plt.subplot2grid((2, 10), (0, 3), rowspan=1,colspan=2,fig=fig)
                # ax3 = plt.subplot2grid((2,10), (0, 5), rowspan=1,colspan=1,fig=fig)
                # ax4 = plt.subplot2grid((2,10), (0, 6), rowspan=1,colspan=2,fig=fig)
                # ax5 = plt.subplot2grid((2, 10), (0,8), rowspan=1,colspan=2,fig=fig)
                
                # plt.subplots_adjust(wspace=0.1)

                # lab_ax_1 = list([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                # lab_ax_2 = list([20,30,40,50,60,70,80,90,100])
                # lab_ax_3 = list([100,150,200,250,300])
                # lab_ax_4 = list([300,400,500,600,700,800,900,1000])
                # lab_ax_5 = list([1000,1500,2000,2500])
                fig = plt.figure(figsize=(8,7))
                ax = plt.axes()
                draw([ax],[list(range(1,self.data_manager.na+1))],"Number of Attack Traces",new)
                save_fig(fig)
                

            def draw_parallel_coordinates(new,x_axis):
                if(not "mesh" in self.model_name):
                    #parallel coordinates are a visualization method used for the MeshNN
                    print("Discarding the parallel coordinates.")
                    return
                d1 = pd.DataFrame(new,columns=[x_axis, y_axis_name ,"Sensitivity" , "Model", "Duration", "Byte"])
                d1 = d1.astype({x_axis: int})
                d1 = d1.astype({y_axis_name: float})
                d1 = d1.astype({"Model": str})
                d1 = d1.astype({"Sensitivity": float})
                d1 = d1.astype({"Duration": float})
                d1 = d1[~d1["Model"].str.contains("HW2")]
                d_cnn = d1[d1["Model"].str.contains("cnn_exp")]
                d1 = d1[~d1["Model"].str.contains("cnn_exp")]
                d1["First Bottleneck"] = np.vectorize(lambda x : x.split(",")[1])(np.array(d1["Model"],dtype=str))
                d1 = d1.astype({"First Bottleneck": int})
                d1["Second Bottleneck"] =np.vectorize(lambda x : x.split(",")[2])(np.array(d1["Model"],dtype=str))
                d1 = d1.astype({"Second Bottleneck": int})
                d1["Middle Neurons"] = np.vectorize(lambda x : x.split(",")[3])(np.array(d1["Model"],dtype=str))
                d1 = d1.astype({"Middle Neurons": int})
                d2 = d1.groupby(["Model",x_axis]).mean()
                d22 = d2.groupby(["Model"]).min()
                d2_1 = d2.reset_index()
                d3 = d1.groupby(["Model",x_axis]).mean().groupby(["Model"])
                d22["Sensitivity"]=d3.max("Sensitivity")["Sensitivity"]
                color = "Duration" if "AXIS-na" in res_name else "Sensitivity"
                fig = px.parallel_coordinates(d2_1, color=color,
                                            dimensions=[x_axis,'First Bottleneck', 'Second Bottleneck', 'Middle Neurons',
                                                        'Key Rank'],color_continuous_scale=px.colors.sequential.Bluered)
            #     fig.show()
                fig2 = px.parallel_coordinates(d22, color=color,
                                            dimensions=['First Bottleneck', 'Second Bottleneck', 'Middle Neurons',
                                                        'Key Rank'],
                                            color_continuous_scale=px.colors.sequential.Bluered)
            #     fig2.show()
                if("epoch" in x_axis.lower()):
                    fig.write_html(results_path+ res_name+"_EPOCHS.html")
                    fig2.write_html(results_path+ res_name+"_NOEPOCHS.html")
                else:
                    fig.write_html(results_path+ res_name+"_Number_Attack.html")
                    fig2.write_html(results_path+ res_name+"_NO_Number_Attack.html")

            draw_parallel_coordinates(new,x_axis)
        except Error as e:
            if(number_of_trials==0):
                traceback.print_exc()
                raise Error("Error while saving the plots:",e)
            else:
                # traceback.print_exc()
                self._print("Could not create the plots ! Skipping the plot creation. Error:",e)
                try:
                    plt.close()
                    plt.clf()
                except:
                    pass
        if(logit):
            self.compute_logits = False
        return None
    
    def _train_model_record(self,x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,key_attack,byte,model=None,get_metrics=False,fast=True,sensi_reg = False,fast_sensi = False):
            """ Trains the model. specialy adapted to plot the record result on the na-axis.

            Parameters
            ----------
            x_profiling : List
                    The attack traces used for unprofiled training. Should have the shape (Na,1,Ns).
            label_profiling : List[int]
                    The labels corresponding to x_attack.
            x_validation : List
                    Traces to use for metrics computations (e.g. sensitivity computation). Can be
                    the same as x_attack.
            label_validation : List[int]
                    The labels corresponding to x_attack_val.
            byte : int
                    Target byte index.
            Returns
            -------
            model : tensorflow.keras.models.Sequential
                    The trained model.
            """
            self._printdw("In train model custom.")
            ACCURACY_SIZE = 1000
            if(model == None):
                model = self._get_model()
            
            loss_fn,label_profiling = self._get_loss_label(label_profiling)
            learning_rate = 0.0010000000474974513
            optimizer = self._get_optimizer(model,learning_rate=self.learning_rate)
            # if(self.optimizer=="Adam"):
            #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # elif(self.optimizer=="SGD"):
            #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=DEFAULT_SGD_MOMENTUM) #momentum 0.5 0.9 0.99
            accuracies=list()
            train_acc=list()
            fast_GEs=list()
            sensitivity_regularizer = 0.0
            data_set=NRFDataset(x_profiling,label_profiling,device = self.data_manager.device)
            #self._printd("x prof 0 is ",torch.min(x_profiling),torch.max(x_profiling),torch.mean(x_profiling),torch.std(x_profiling))
            train_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle = self.seed is None)
            if(self.lr_schedule is not None):
                # scheduler = StepLR(optimizer, step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])
                scheduler = OneCycleLR(optimizer=optimizer,epochs=self.epochs,max_lr= 0.002,total_steps=None,steps_per_epoch=ceil(self.data_manager.nt/self.batch_size)\
                    ,pct_start=0.3,anneal_strategy='linear',cycle_momentum=False,base_momentum=0.85,max_momentum=0.95,div_factor=25,final_div_factor=1e4,three_phase=True\
                        ,last_epoch=-1,verbose=self.verbose==2)#,step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])

            self._start_record()
            for t in range(self.epochs):
                model.train()
                for i,(x,y) in enumerate(train_loader):
                # print(torch.cuda.memory_allocated(self.data_manager.device))
                    if(t == self.epochs -1 and i == len(train_loader) -1):
                        #model.show_next()
                        pass
                    y_pred = model(x) 

                    if(self.loss=="mse"):
                        #print("self loss")
                        y_pred=torch.exp(y_pred)
                    #print("pred is ",y_pred.get_device()," y is ",y.get_device()," model is ")
                    #print("pred 0 is ",y_pred[0], " y is ",y[0])
                    loss = loss_fn(y_pred, y)
                    if(loss.isnan().any()):
                        raise Error("Loss is nan ! ",loss,". Check your input traces again.")
                    # sensitivity = self.get_sensitivity_tensors(model,x,y)
                    # topk=torch.topk(torch.abs(sensitivity),2)
                    # sensitivity_regularizer = topk.values[0]
                    optimizer.zero_grad()

                    # Compute a regularization, different for each layer (as in [107])
                    if(self.lambdas!=None):
                        penalty=0.0
                        for name,p in model.named_parameters():
                            if("mask" in name):
                                continue
                            if(name=="regularized_function_1.weight"):
                                penalty+= self.lambdas[0] * sum((w.abs().sum() for w in p )) + self.lambdas[1] * sum((w.pow(2.0).sum()) for w in p) 
                            elif(name=="regularized_function_2.weight"  or ("regularized_function_" in name and "weight" in name)):
                                penalty+= self.lambdas[2] * sum((w.abs().sum()) for w in p) + self.lambdas[3] * sum((w.pow(2.0).sum()) for w in p)                        
                        loss+=penalty
                    if(False and get_metrics):
                        if(t>0):
                            pass
                            #print("loss is ",loss,loss/self.leakage_model.get_classes()," sens reg is ",sensitivity_regularizer)
                        if(self.dim == 0):
                            #print("loss/batch is ",loss/self.batch_size," sensis ",sensitivity_regularizer," sensi/batch ",sensitivity_regularizer/self.batch_size)
                            loss = (loss / self.batch_size)  #+  sensitivity_regularizer #/ self.batch_size #/loss
                        else:
                            loss = loss/self.leakage_model.get_classes()  #+ sensitivity_regularizer *10
                    loss.backward()

                    optimizer.step()
                    #print("loss is ",float(loss))
                    del loss
                if(self.lr_schedule is not None):
                    # In case you use the OneCycleLR, put the scheduler step inside the batch loop.
                    scheduler.step()
                model.eval()
                if(self.verbose==1 or self.verbose ==2):
                    #Watch out ! If the validation is in GPU, and is too big, it may be too big for the GPU RAM 
                    accuracies.append(self.get_accuracy(self._batch_predict(model,x_validation[:ACCURACY_SIZE]),label_validation[:ACCURACY_SIZE]))
                    train_acc.append(self.get_accuracy(self._batch_predict(model,x_profiling[:ACCURACY_SIZE]),label_profiling[:ACCURACY_SIZE]))
                if(t==0):
                    self._round_record(self.epochs)
            if(fast and not self.data_manager.blind):
                self._printw("Set to use all the validation traces !")


                number_x_axis = range(1,len(x_validation)+1) #list([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1500,2000,len(x_validation)])
                # output_probas = self._batch_predict(model,x_validation).detach().cpu()
                # output_probas_on_keys = self.turn_output_probas_to_key_proba(
                #     output_probas, plaintext_val, byte, mask_val)
                # res = np.zeros(shape = (len(number_x_axis)))
                if(self.dim==0):
                    # if(len(number_x_axis)>10000):
                    #     self._printw("WARNING: Recording on the NA-axis with more than 10'000 attack traces using DIM 0 might take an eternity to complete.")
                    output_probas = torch.tensor([])
    	            
                    for ind,n in enumerate(number_x_axis):
                        # if(ind%100 == 0):
                        #     print("Ev. at ",ind)
                        # print("At",ind,"/",n," using indexes",((n-1)//self.batch_size)*self.batch_size, " : ",((n-1)//self.batch_size)*self.batch_size+((n-1)%self.batch_size)+1)
                        output_probas_tmp = self._batch_predict(model,x_validation[((n-1)//self.batch_size)*self.batch_size:((n-1)//self.batch_size)*self.batch_size+((n-1)%self.batch_size)+1]).detach().cpu()
                        # print("output probas tmp is ",output_probas_tmp)
                        new_pred=torch.cat((output_probas,output_probas_tmp),0)
                        # print("new pred is ",new_pred)
                        output_probas_on_keys = self.turn_output_probas_to_key_proba(
                            new_pred, plaintext_val[:n], byte, mask_val)
                        if((n)%self.batch_size==0):
                            # print("adding new pred to output probas")
                            output_probas=new_pred
                        fast_GEs.append(Attack.get_GE(output_probas_on_keys,key_attack[0][byte]))
                else:
                    output_probas = self._batch_predict(model,x_validation).detach().cpu()
                    output_probas_on_keys = self.turn_output_probas_to_key_proba(
                        output_probas, plaintext_val, byte, mask_val)
                    for ind,n in enumerate(number_x_axis):
                        fast_GEs.append(Attack.get_GE(output_probas_on_keys[:n],key_attack[0][byte]))
                    # if(n <= 20):
                    #     trials = 100
                    # elif(n < len(x_validation)- 100):
                    #     trials = 10
                    # else:
                    #     trials= 1
                    # tmp = np.zeros(shape = (trials,))
                    # for i in range(trials):
                    #     indices = np.random.choice(output_probas_on_keys.shape[0], n, replace=False)
                    #     tmp[i] = Attack.get_GE(output_probas_on_keys[indices],key_attack[0][byte])
                    # res[ind] = np.mean(tmp)

                # fast_GEs.append(res)
            print("FA is",fast_GEs)
            # if(fast):
            #     print("Mining from",len(fast_GEs),len(fast_GEs[0]))
            #     fast_GEs = np.min(fast_GEs,axis=0)
            #     print("Mining to",len(fast_GEs))

            return model,accuracies,fast_GEs,train_acc #, torch.abs(sensitivity).detach().numpy()


    def _attack_byte_record(self,byte,get_output_probas=False,split_rate = 0.95,fast_GE = False,get_fast_ge = False):
            """Attack byte: launches a full attack on a byte. 

            Will print the result of the attack on the terminal.

            Parameters
            ----------
            byte : int
                    byte number to attack. Some bytes are harder to attack.
            """
            number_x_axis = list([100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4500,5500,6500,7500,12500,17500,22500,27500,37500,47500])
            self._printd("In attack byte profiled custom")
            x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,x_attack,\
               label_attack,plaintext_attack,key_attack,mask_attack = self.data_manager.prepare_profiled_data(byte,leakage_model = self.leakage_model ,\
                                                use_attack_as_val = None,split_rate=split_rate,dk=self.dk)
            fast_GEs = list()
            for n_t in number_x_axis: 
                model,accuracies,fast_GE,train_acc = self.train_model(x_profiling[:n_t],label_profiling[:n_t],x_validation,label_validation,plaintext_val,key_attack,byte,None,True,fast=True)
                fast_GEs.append(np.min(fast_GE,axis=-1))
            return None,None,fast_GEs