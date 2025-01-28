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
# brief     : ML blind attack on AES
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
import torch
from Attacks.attack import Attack, SensitivityMode, TrainingMethods
from Attacks.unprofiled import UnProfiled
from Ciphers.AES_leakage import AESLeakageModel
from Data.data_manager import DataManager



class Function(Enum):
    NORMAL = 0
    #HARD_WITH_EASY = 1
    DICHOTOMY = 2

class BlindUnProfiled(UnProfiled):
    """Use this class when you want to launch a real unprofiled attack (i.e. when no attack key is known).
    """
    def __init__(self,first_epochs=15,model_name="mlp",batch_size=100,loss="nlll",optimizer="Adam",leakage_model=AESLeakageModel("HW","SBox"),\
                    verbose=1,lambdas= None ,dk=False,noise=False,seed=None,\
                    sensitivity_mode=SensitivityMode.CLASSIC,data_manager = DataManager(na = "all",ns = "all",nt = 0,fs=0,file_name_attack = 'file_name_attack.h5',\
                        databases_path="/path/to/Databases/",blind=True),results_path="/path/to/PlotsResults/",\
                            info="",dim=0,threshold = 0.7,function = Function.NORMAL):  
        """Initializes a BlindUnProfiled attack.
        Parameters
        ----------
        first_epochs : int, default : 15
                how many epochs to train the NN.
        model_name : str, default : mlp
                Which NN model to use. 
        batch_size : int, default : 100
                Batch size used during training.
        loss : str, default : nlll
                Loss function to use. Can be mse, cross_entropy or nlll.
        optimizer : str, default : Adam
                Can be Adam or SGD.
        leakage_model : Ciphers.LeakageModel, default : HW(SBox)
                A LeakageModel class used for label computations.
        verbose : int, default : 1
                What should be printed out during execution ? 0 : only the result,
                1 : the different metrics (e.g. remaining time, sensitivities, accuracies),
                2 : output debug informations too.
        lambdas: List[Float], default : None
                If not None, specifies the regularization terms for the L1 and L2 loss on
                two layers. The 2*(i-1) + two components are the L1 & L2 loss for layer i.
        dk : bool, default : False
                Whether to use Domain Knowledge neurons. Those add plaintext information
                in the MLP part of the network.
        noise : bool, default : False
                Whether to add gaussian noise to the input of the training data.
        seed : int, default : None
                Which seed to use if not None.
        sensitivity_mode : SensitivityMode, default : SensitivityMode.CLASSIC
                Which algorithm to choose for the sensitivity calculation.
        data_manager : Data.DataManager, default : DataManager()
                The datamanager handles anything related to the data.
        results_path : str, default : "/path/to/PlotsResults/"
                Where to store the learning plots.
        info : str, default : ""
                A small text to insert in the result file name.
        dim : int, default : 0
                On which dimension/axis to apply the softmax filter in the NN.
        threshold : float, default : 0.4
                When the certainty is higher than this threshold, we assume the guess is correct.
        function : Attacks.blind_unprofiled.Function, default : Function.NORMAL
                Which method to use to attack a byte.
        """
       
       
       
        super().__init__(first_epochs, model_name, batch_size, loss, optimizer, leakage_model, verbose, lambdas, dk, noise, seed, False, sensitivity_mode, data_manager, results_path, TrainingMethods.PRUNING_HALF_EPOCHS_LOCAL, info, dim, threshold)
        self.function = function
        self.set_pruning_percentage(0.7)

    """    def _get_model(self,num_easy_bytes = 0):
        if(self.function == Function.NORMAL):
            return super()._get_model()
        num_samples=self.data_manager.ns
        num_classes = self.leakage_model.get_classes() * (num_easy_bytes + 1)
        if(self.model_name=="cnn_exp"):
            model = CNNexp(num_classes,num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mlp"):
            model =   MLP(num_classes,num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mcnn"):
            model =      MCNN(num_classes,num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name == "vgg"):
            model =  VGGNoise(num_classes,num_samples,self.dk,noise_std = self.noise,dim=self.dim)
        elif(self.model_name=="resnet"):
            model =    ResNet(num_classes,ns=num_samples,dk=self.dk,noise_std = self.noise,dim=self.dim)
        elif(self.model_name == "cnn_aeshd"):
            model =  NetAeshd(num_classes,num_samples,self.dk,noise_std=self.noise,dim=self.dim)
            self._printd("Using NetAeshd model !")
        else:
            raise Error("Invalid model name.")
        return model
    def get_accuracy(self,preds,labels):
        Get accuracy: computes the prediction's accuracy.

        Parameters
        ----------
        preds : List[List[float]]
                Model predictions, i.e. probability distribution of each class.
        labels: List[int], List[List[int]]
                Labels corresponding to each prediction. May or not be in categorical form.
        Returns
        -------
        float
                The prediction accuracy.
        
        if(self.function ==Function.NORMAL ):
            return super().get_accuracy(preds,label)
        else:
            label = 
        tot=0
        is_argmaxed = len(labels.shape) == 1 
        for i in range(len(preds)):
            pred=torch.argmax(preds[i])
            if(is_argmaxed):
                label = labels[i] 
            else:
                label = torch.argmax(labels[i])
            if(pred==label):
                tot = tot + 1
        return tot / len(preds)

    """

    """def attack_bytes(self, byte_range, guess_list=range(16)):
        if(isinstance(byte_range,int)):
            byte_range=[byte_range]
        if(len(guess_list) < 16):
            return super().attack_bytes(byte_range, BULLSHITguess_list)
            
        res = []
        for byte in byte_range:
            self._printd(f"Attacking byte {byte}...")
            res.append(self.attack_byte(byte,BUööSHITguess_list))
        return res
    """

    def attack_hard_byte(self,hard_byte,easy_bytes,easy_keys,guess_list=range(256),max_epochs = 20):
        if(self.dk):
            raise NotImplementedError()
        if(isinstance(easy_bytes,int)):
            easy_bytes=[easy_bytes]
        ### Get the easy labels ###
        labels_easy = None
        for byte,key in zip(easy_bytes,easy_keys):
            _,label_easy,_,_=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=key,leakage_model = self.leakage_model)
            if(labels_easy is None):
                labels_easy = label_easy
            else:
                labels_easy = torch.add(labels_easy,label_easy)  #add
       # x_attack,label_attack,plaintext_attack,key_attack = self.data_manager.prepare_unprofiled_data(hard_byte,self.dk,key_guess=???,leakage_model = self.leakage_model)
       # label_attack = torch.add(labels_easy,label_attack)
        return self.attack_byte(hard_byte,guess_list,max_epochs,labels_easy)

    def attack_byte(self,byte,guess_list=range(256),max_epochs = 100,easy_labels = None):
        if(self.function == Function.NORMAL):
            return self._attack_byte_normal(byte,guess_list,max_epochs,easy_labels)
        if(self.function == Function.DICHOTOMY):
            return self._attack_byte_dichotomy(byte,guess_list,easy_labels)

    def _attack_byte_normal(self,byte,guess_list=range(256),max_epochs = 20,easy_labels = None):
        """Attack byte: launches a full blind attack on a byte. 

        This method includes several optimizations compared to the classical (parent) attack_byte 
        method. It will first perform a classical pruning with small epochs, and then train each model
        until the certainty achieved surpasses the threshold (or it reaches the max_epochs).

        Parameters
        ----------
        byte : int
                byte number to attack. Some bytes are harder to attack.
        guess_list : list, default: range(256)
                Against which byte values to compare the right key guess. 
        max_epochs : int
                At how many epochs to stop learning even if the threshold is not reached.
        Returns
        -------
        int
                The guessing entropy.
        """


        get_fast_GE_if_not_loss = False  #setting this to True implies longer execution
        self._printd("Seed:",self.seed)

        self._printd("Num classes",self.leakage_model.get_classes())
        #data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=0,leakage_model = self.leakage_model)
        #x_attack,label_attack,plaintext_attack,key_attack=data
        key_guesses = list(guess_list)
        if(len(key_guesses) != 256):
            print("Warning: unknown result with a smaller guess_list !")
#x_attack,label_attack,plaintext_attack,key_attack=data
        sensitivities=np.zeros((256,self.data_manager.get_ns(self.leakage_model)))
        accuracies=[None] * 256
        fast_GEs = [None] * 256
        train_accs = [None] * 256
        models=[None] * 256
        
        i = 0
        certainty = [0]
        try:
            while(certainty[0] < self.threshold and i <= max_epochs):
                self._start_record()
                for key_index,key_guess in enumerate(key_guesses):
                    #self._printd("To ",key_guess," preparing: ",self.leakage_model,byte,-1,key_guess,data,na,ns)


                    data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=key_guess,leakage_model = self.leakage_model)
                    x_attack,label_attack,plaintext_attack,key_attack=data
                    if(easy_labels is not None):
                        label_attack = torch.add(label_attack,easy_labels)

                    x_attack_train=x_attack
                    label_attack_train=label_attack
                    x_attack_val=x_attack
                    label_attack_val=label_attack
                    plaintext_attack_train=plaintext_attack
                    plaintext_attack_val=plaintext_attack
                    
                    x_attack_train.requires_grad_()
                    x_attack_val.requires_grad_()
                    data_for_model = x_attack_train,label_attack_train,plaintext_attack_train,key_guess,byte,x_attack_val,label_attack_val,plaintext_attack_val
                    #x_prof,label_prof,x_prof_val,label_prof_val,plaintext_prof_val,key,byte,model=None,get_metrics=True,fast=False)
                    if(i == 0):
                    #    self.epochs = 20  -> keep same # of epochs as given 
                        model_,sensitivity,accuracy,fast_GE,train_acc = self.train_model_pruning(x_attack_train,\
                                        label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte=byte,model=None,get_metrics=True,fast=get_fast_GE_if_not_loss)#,info=key_guess==right_key)
                        sensitivities[key_guess]=np.array(sensitivity)
                        models[key_guess] = model_
                        accuracies[key_guess] = accuracy
                        fast_GEs[key_guess] = fast_GE
                        train_accs[key_guess] = train_acc

                    else:
                        self.epochs = 10
                        model_,sensitivity,accuracy,fast_GE,train_acc = self.train_model_default(x_attack_train,\
                                        label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte=byte,model=models[key_index],get_metrics=True,fast=get_fast_GE_if_not_loss)#,info=key_guess==right_key)
                        sensitivities[key_guess] = np.add(sensitivities[key_guess],sensitivity)
                        models[key_guess] = model_
                        accuracies[key_guess].extend(accuracy)
                        fast_GEs[key_guess].extend(fast_GE)
                        train_accs[key_guess].extend(train_acc)
                    
                #   print("Corretc key is ",key_attack[0][byte]," and the guess is ",key_guess)
                    if(key_index == 0):
                        self._round_record(len(key_guesses))
                        
                sens = np.array(sensitivities)
                sens=np.abs(sens[key_guesses])
                key_prob,certainty = Attack.get_keys_from_sensitivity(sens,key_guesses)
                self._printm("certainty is ",certainty," for best guesses ",key_prob[:42])

            self._end_record()
            i += self.epochs
            #print("had ",sens.shape,type(sens),type(sens[0]))
           # key_prob,certainty = Attack.get_keys_from_sensitivity(sens,key_guesses)
            self._printm("certainty is ",certainty," for best guesses ",key_prob[:42])
        except KeyboardInterrupt:
            if(self.verbose==1 or self.verbose == 2):
                print("Stopping the execution and saving the plots.")
                plots = self._prepare_figures(get_fast_GE_if_not_loss)  #plot_sensi,plot_acc,plot_acc_train,plot_GE
                for key_index,key_guess in enumerate(key_guesses):
                    if(sensitivities[key_guess] is not None):
                        self._draw_on_plots(*plots,np.abs(sensitivities[key_guess]),accuracies[key_guess],train_accs[key_guess],fast_GEs[key_guess],correct = False,key_guess=key_guess)
                    #self.draw_class_predictions(plots[1],models[key_prob[0]],x_attack_train,label_attack_train)
                plots[0].set_title("Accumulated Sensitivity for byte "+str(byte)+" - Best key guess: "+str(key_prob[0]))
                path_res_file=self._get_res_name(byte,len(key_guesses),ge = "blind")
                plt.savefig(path_res_file)
                plt.show(block=False)
            raise KeyboardInterrupt

        if(self.verbose==1 or self.verbose == 2):
            plots = self._prepare_figures(get_fast_GE_if_not_loss)  #plot_sensi,plot_acc,plot_acc_train,plot_GE
            for key_index,key_guess in enumerate(key_guesses):
                self._draw_on_plots(*plots,np.abs(sensitivities[key_guess]),accuracies[key_guess],train_accs[key_guess],fast_GEs[key_guess],correct = False,key_guess=key_guess)
                #self.draw_class_predictions(plots[1],models[key_prob[0]],x_attack_train,label_attack_train)
            plots[0].set_title("Accumulated Sensitivity for byte "+str(byte)+" - Best key guess: "+str(key_prob[0]))
            path_res_file=self._get_res_name(byte,len(key_guesses),ge = "blind")
            plt.savefig(path_res_file)
            plt.show(block=False)
        return key_prob,certainty

    def _attack_byte_dichotomy(self,byte,guess_list=range(256),easy_labels = None):
        """Attack byte dichotomy: launches a full blind attack on a byte. 

        This method includes several optimizations compared to the classical attack_byte 
        method. In this method, half of the guesses (the ones resulting in the smallest sensitivity spikes)
        are discarded after self.epoch epochs are done. Then, it will iteratively add 10 epochs and remove
        the worst guesses until the certainty achieved surpasses the threshold (or only one key guess is
        left). The goal of this method is to limit the complexity of the computation.

        Parameters
        ----------
        byte : int
                byte number to attack. Some bytes are harder to attack.
        guess_list : list, default: range(256)
                Against which byte values to compare the right key guess. 
        Returns
        -------
        int
                The guessing entropy.
        """
        if(self.threshold < 0.7):
            self._printw("Warning, dichotomy with small threshold may not work.")
        key_guesses = list(guess_list)
        if(len(key_guesses) != 256):
            self._printw("Warning: unknown result with a smaller guess_list !")
        get_fast_GE_if_not_loss = False  #setting this to True implies longer execution
        self._printd("Seed:",self.seed)

        self._printd("Num classes",self.leakage_model.get_classes())
        #data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=0,leakage_model = self.leakage_model)
        #x_attack,label_attack,plaintext_attack,key_attack=data
        sensitivities=np.zeros((256,self.data_manager.get_ns(self.leakage_model)))
        accuracies=[None] * 256
        fast_GEs = [None] * 256
        train_accs = [None] * 256
        models=[None] * 256
        i = 0
        certainty = [0]
        result_tracker = np.zeros((256))
        occurence_tracker = np.zeros((256))
        while(certainty[0] < self.threshold and len(key_guesses) > 1):
            self._start_record()
            i = 0
            for key_index,key_guess in enumerate(key_guesses):
                #self._printd("To ",key_guess," preparing: ",self.leakage_model,byte,-1,key_guess,data,na,ns)


                data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=key_guess,leakage_model = self.leakage_model)
                x_attack,label_attack,plaintext_attack,key_attack=data
                if(easy_labels is not None):
                    label_attack = torch.add(label_attack,easy_labels)

                x_attack_train=x_attack
                label_attack_train=label_attack
                x_attack_val=x_attack
                label_attack_val=label_attack
                plaintext_attack_train=plaintext_attack
                plaintext_attack_val=plaintext_attack
                
                x_attack_train.requires_grad_()
                x_attack_val.requires_grad_()
                data_for_model = x_attack_train,label_attack_train,plaintext_attack_train,key_guess,byte,x_attack_val,label_attack_val,plaintext_attack_val
                #x_prof,label_prof,x_prof_val,label_prof_val,plaintext_prof_val,key,byte,model=None,get_metrics=True,fast=False)
                if(i == 0):
                #    self.epochs = 20  -> keep same # of epochs as given 
                    model_,sensitivity,accuracy,fast_GE,train_acc = self.train_model_pruning(x_attack_train,\
                                    label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte=byte,model=None,get_metrics=True,fast=get_fast_GE_if_not_loss)#,info=key_guess==right_key)
                    sensitivities[key_guess] = np.array(sensitivity)
                    models[key_guess] = model_
                    accuracies[key_guess] = accuracy
                    fast_GEs[key_guess] = fast_GE
                    train_accs[key_guess] = train_acc

                else:
                    self.epochs = 10
                    model_,sensitivity,accuracy,fast_GE,train_acc = self.train_model_default(x_attack_train,\
                                    label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte=byte,model=models[key_guess],get_metrics=True,fast=get_fast_GE_if_not_loss)#,info=key_guess==right_key)
                    sensitivities[key_guess] = np.add(sensitivities[key_guess],sensitivity)
                    models[key_guess] = model_
                    accuracies[key_guess].extend(accuracy)
                    fast_GEs[key_guess].extend(fast_GE)
                    train_accs[key_guess].extend(train_acc)
                
            #   print("Corretc key is ",key_attack[0][byte]," and the guess is ",key_guess)
                if(key_index == 0):
                    self._round_record(len(key_guesses))

            self._end_record()
            i += self.epochs
            sens = np.array(sensitivities)
            sens=np.abs(sens[key_guesses])
            key_prob,certainty = Attack.get_keys_from_sensitivity(sens,key_guesses)
            self._printm("certainty is ",certainty," for best guesses ",key_prob)
            result_tracker[key_prob[0]] += certainty[0]
            occurence_tracker[key_prob[0]] += 1
            key_guesses = key_prob[:(len(key_prob)+1) // 2]
            #remove additionnally all > threshold guesses:
            retain = np.where(np.array(certainty) < self.threshold)
            np.insert(retain,0,0)
            key_guesses = key_guesses[retain[:min(len(key_guesses)-1,len(retain)-1)]]
        if(self.verbose==1 or self.verbose == 2):
            plots = self._prepare_figures(get_fast_GE_if_not_loss)  #plot_sensi,plot_acc,plot_acc_train,plot_GE
            for key_guess in (list(guess_list)):
                self._draw_on_plots(*plots,np.abs(sensitivities[key_guess]),accuracies[key_guess],train_accs[key_guess],fast_GEs[key_guess],correct = False,key_guess=key_guess)
                #self.draw_class_predictions(plots[1],models[key_prob[0]],x_attack_train,label_attack_train)
        
        print(f"BYTE {byte} RESULT SUM:",np.argmax(result_tracker)," RESULT OCCURENCE:",np.argmax(occurence_tracker)," RESULT LAST:",key_prob[0])
        
        if(self.verbose==1 or self.verbose == 2): #
            colors=["b","g","c","m","y","k","peru","purple","olive","pink","forestgreen","moccasin"]
            plots[0].set_title("Accumulated Sensitivity for byte "+str(byte)+" - Best key guess: "+str(key_prob[0])+", of color: "+colors[key_prob[0] % len(colors)])
            path_res_file=self._get_res_name(byte,len(key_guesses),ge = "blind")
            plt.savefig(path_res_file)
            plt.show(block=False)
        
        return key_prob,certainty