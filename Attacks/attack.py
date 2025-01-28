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
# brief     : ML base class for AES attacks
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
from abc import ABC, abstractmethod
from enum import Enum
import heapq
import math
import os
from multiprocessing import Pool,cpu_count
from time import gmtime, strftime, time
import numpy as np
import torch
from tensorflow.keras.utils import to_categorical
import torch.nn.utils.prune as prune
import copy
from Architectures.torch_models import MCNN, MLP_AESRD, MLP_ASCAD, VGG16, AgnosticModel, MLPexp,CNNbest, MeshNN, NetAeshd, CNNexp, MLP,ResNet, Simple_AES_RD,VGGNoise,CNN_MPP16, CNN_zaid_desync0, CNN_zaid_desync50, CNN_zaid_desync100, NoConv_desync0, NoConv_desync50, NoConv_desync100, MLP_AESRD_IMB
from Data.custom_datasets import NRFDataset3, PredictionDataset 
from error import Error
from torch.utils.data import DataLoader

class TrainingMethods(Enum):
    DEFAULT = 0
    """Default training."""
    PRUNING_LOCAL = 1
    """LTH pruning done layer-wise."""
    PRUNING_HALF_EPOCHS_LOCAL = 1.5
    """LTH pruning done layer-wise, using only half as many epochs in the second training."""
    PRUNING_GLOBAL = 5
    """LTH pruning done on the whole model. May disable entire layers."""
    PRUNING_HALF_EPOCHS_GLOBAL = 5.5
    """LTH pruning done on the whole model. May disable entire layers. Uses only half as many epochs in the second training."""
    ADVERSARIAL = 2
    MIXUP = 3
    CROSS_VALIDATION = 4
    CUSTOM = 5  #to be implemented by the user

class SensitivityMode(Enum):
    CLASSIC=0
    ABSOLUTE_VALUES=1
    ON_RAW_TRACE = 2
    GRAD_CAM=3
    GRAD_CAM_PLUS_PLUS=4

DEFAULT_SGD_MOMENTUM=0.99
DEFAULT_NOISE_STDVAR=0.25

class Attack(ABC):
    """The base class for any SCA Attacks in this module.

    Allows to perform an attack on a chosen cipher. 

    """
    def __init__(self,model_name,batch_size,loss,optimizer,leakage_model,\
                    verbose,lambdas,dk,noise,seed,data_manager,results_path,\
                        training,info,dim,threshold = 0.8,lr_schedule=None,learning_rate="default"):
        """ Initializes an Attack.


        Parameters
        ----------
        model_name : str
                Which NN model to use.
        batch_size : int
                Batch size used during training.
        loss : str
                Loss function to use. Can be mse, cross_entropy or nlll.
        optimizer : str
                Can be Adam or SGD.
        leakage_model : Ciphers.LeakageModel
                A LeakageModel class used for label computations.
        verbose : int
                What should be printed out during execution ? 0 : only the result,
                1 : the different metrics (e.g. remaining time, sensitivities, accuracies),
                2 : output debug informations too.
        lambdas: List[Float]
                If not None, specifies the L1 & L2 regularization terms of two layers.
                The first two values (by default 0.03,0.03) are the L1 and L2 regularization 
                for a model's layer whose name is "regularized\_function\_1". The next two values 
                are the L1 & L2 regularization for any layer in the model whose name contain
                "regularized\_function\_".
        dk : bool
                Whether to use Domain Knowledge neurons. Those add plaintext information
                in the MLP part of the network.
        noise : bool
                Whether to add gaussian noise to the input of the training data.
        seed : int
                Which seed to use. If set to None, doesn't use any seed.
        data_manager : Data.DataManager
                The datamanager handles anything related to the data.
        results_path : str
                Where to store the learning plots.
        training : attacks.attack.TrainingMethods, default : Trainingmethods.DEFAULT
                Which training method to use.
        info : str, default : ""
                A small text to insert in the result file name.
        dim : int, default : 0
                On which dimension/axis to apply the softmax filter in the NN.
        threshold : float, default : 0.4
                When the certainty is higher than this threshold, we assume the guess is correct.
                To be used in blind attacks.
        lr_schedule : List[int,float] | None. Default : None.
                learning rate Scheduler parameter. If not None, after each lr_schedule[0] epochs,
                multiply the learning rate by lr_schedule[1].
        """
        torch.set_num_threads(max(1,os.cpu_count()//3))  #limits the utilization of the CPU a bit. should not be higher than cpu_count()//2
        self.info=info
        self.lr_schedule=lr_schedule
        self.model_name=model_name.lower()
        self.batch_size=batch_size
        self.loss=loss.lower()
        self.verbose=verbose
        self.learning_rate=learning_rate
        self._warnings = set()
        self.dim=dim
        self.threshold=threshold
        if(self.loss=="mse"):
            self._printm("To achieve best results (speed and GE), avoid the MSE loss.")
        self.optimizer=optimizer
        self.leakage_model=leakage_model
        self.lambdas=lambdas
        self.dk=dk
        if(noise):
            self.noise=DEFAULT_NOISE_STDVAR
        else:
            self.noise=None
        self.time1=None
        self.results_path=results_path
        self.data_manager = data_manager 
        if(seed != None):
            self.set_seed(seed) #int(time()*1000) % 2**31
        else:
            self.seed = None
        self.training=training
        self._set_training()
        
    @abstractmethod
    def get_pruning_percentage(self):
        pass
    def set_pruning_percentage(self,percentage):
        self.LTH = percentage

    def _get_optimizer(self,model,start=0,learning_rate = "default"):

        if(self.optimizer=="RMSprop"):
            if(learning_rate == "default"):
                learning_rate = 0.00001
            optimizer = torch.optim.RMSprop(list(model.parameters())[start:], lr=learning_rate)
        else:
            if(learning_rate == "default"):
                learning_rate = 0.0010000000474974513
            if(self.optimizer=="Adam"):
                optimizer = torch.optim.Adam(list(model.parameters())[start:], lr=learning_rate)
            elif(self.optimizer=="SGD"):
                optimizer = torch.optim.SGD(list(model.parameters())[start:], lr=learning_rate,momentum=DEFAULT_SGD_MOMENTUM)
            elif(self.optimizer=="Nesterov"):
                optimizer = torch.optim.SGD(list(model.parameters())[start:], lr=learning_rate,momentum=DEFAULT_SGD_MOMENTUM,nesterov=True)
            elif(self.optimizer=="Adagrad"):
                optimizer = torch.optim.Adagrad(list(model.parameters())[start:], lr=learning_rate)
            elif(self.optimizer=="Adadelta"):
                optimizer = torch.optim.Adadelta(list(model.parameters())[start:], lr=learning_rate)
            else:
                raise Error(f"Optimizer {self.optimizer} not supported.")
        return optimizer
    def _get_model(self):
        """Returns the model corresponding to the init arguments."""
        num_samples=self.data_manager.get_ns()
        if(self.model_name=="cnn_exp"):
            model = CNNexp(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="cnn_best"):
            model = CNNbest(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="cnn_zaid0"):
            model = CNN_zaid_desync0(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="cnn_zaid50"):
            model = CNN_zaid_desync50(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="cnn_zaid100"):
            model = CNN_zaid_desync100(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="no_conv0"):
            model = NoConv_desync0(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="no_conv50"):
            model = NoConv_desync50(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim) 
        elif(self.model_name=="no_conv100"):
            model = NoConv_desync100(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)   
        elif(self.model_name in ["CNN_MPP16","MPP","MPP16","cnn_mpp16","mpp16","mpp"]):
            model = CNN_MPP16(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name == "agnostic"):
            model = AgnosticModel(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mlp"):
            model =    MLP(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mlp_ascad"):
            model =    MLP_ASCAD(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim) 
        elif(self.model_name=="mlp_aesrd"):
            model =    MLP_AESRD(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif("mlp_aesrd_imb" in self.model_name):
            tmp=self.model_name.split(",")
            activation = tmp[1].lower()
            neurons=tmp[2:]
            for i in range(len(neurons)):
                neurons[i] = int(neurons[i])
            model =    MLP_AESRD_IMB(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,activation=activation,neurons=neurons)
        elif(self.model_name=="simple_aes_rd" or self.model_name == "simplified_aesrd"):
            model =    Simple_AES_RD(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mcnn"):
            model =      MCNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name == "vgg"):
            model =  VGGNoise(self.leakage_model.get_classes(),num_samples,self.dk,noise_std = self.noise,dim=self.dim)
        elif(self.model_name == "vgg16"):
            model = VGG16(num_classes = self.leakage_model.get_classes(),ns = num_samples,dk = self.dk,noise_std = self.noise,dim=self.dim)
        elif(self.model_name=="resnet"):
            model =    ResNet(self.leakage_model.get_classes(),ns=num_samples,dk=self.dk,noise_std = self.noise,dim=self.dim)
        elif(self.model_name == "cnn_aeshd"):
            model =  NetAeshd(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        elif(self.model_name=="mlp_exp"):
            model =     MLPexp(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        # elif("mesh_old" in self.model_name):
        #     # first_model = MLPexp(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        #     # first_model.load_state_dict(torch.load(os.path.dirname(__file__) + "/"+f"Models/state_MLPexp_dim0_{str(self.leakage_model)}_nrf.pt"))
        #     model = MeshNNOld(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,\
        #         batch_size=self.batch_size,first_model = "mlp" )# torch.load(os.path.dirname(__file__) + "/"+f"Models/MLPexp_dim1_{str(self.leakage_model)}_nrf.pt").eval())
        elif("meshMLPexp" in self.model_name):
            num_neurons_middle = int(self.model_name[9:])
            # print("Setting ",num_neurons_middle)
            # self._printw("Uses an old MLP dim 0")
            # first_model = MLP(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
            # first_model.load_state_dict(torch.load(os.path.dirname(__file__) + "/"+f"Models/state_mlp_dim1_{str(self.leakage_model)}_nrf.pt"))
            model = MeshNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,batch_size=self.batch_size,
            first_model = "mlp_exp",num_neurons_middle=num_neurons_middle) #
        elif("mesh" in self.model_name):
            if(len(self.model_name)>6):
                try:
                    infos = self.model_name.split(",")
                    first_bottlneck_size = int(infos[1])
                    second_bottleneck_size = int(infos[2])
                    num_neurons_middle = int(infos[3])
                    model = MeshNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,batch_size=self.batch_size,first_model="cnn_exp",\
                        first_bottleneck_size_per_class=first_bottlneck_size,second_bottleneck_size_per_class=second_bottleneck_size,num_neurons_middle=num_neurons_middle)
                except:
                    self._printw("Warning: will use the MeshNN with default arguments.")
                    model = MeshNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,batch_size=self.batch_size,first_model="cnn_exp")
            else:
                model = MeshNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,batch_size=self.batch_size,first_model="mlp_aesrd")
        # elif("mesh" in self.model_name):
        #     num_neurons_middle = int(self.model_name[4:])
        #     # print("Setting ",num_neurons_middle)
        #     # self._printw("Uses an old MLP dim 0")
        #     # first_model = MLP(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim)
        #     # first_model.load_state_dict(torch.load(os.path.dirname(__file__) + "/"+f"Models/state_mlp_dim1_{str(self.leakage_model)}_nrf.pt"))
        #     model = MeshNN(self.leakage_model.get_classes(),num_samples,self.dk,noise_std=self.noise,dim=self.dim,batch_size=self.batch_size,
        #     first_model = "cnn_exp",num_neurons_middle=num_neurons_middle)#torch.load(os.path.dirname(__file__) + "/"+f"Models/mlp_dim0_{str(self.leakage_model)}_nrf.pt").eval())
        else:
            raise Error("Invalid model name")
        model.to(self.data_manager.device)
        # self._printd(sum(p.numel() for p in model.parameters()))
        return model
                        #(x_attack_train,label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte,model=None,get_metrics=True,fast=False)
    
    def _draw_class_predictions(self,plot_extra,model_,x_attack_any,label_attack_any,id_in_hw_visualize=False):
        """Draws the confusion matrices on the given plots.
        
        Currently set to consider the probability density over each class instead of just the most probable.
        """
        # self._printdw("Currently set to consider the probability density over each class instead of just the most probable.")
        output_probas = self._batch_predict(model_,x_attack_any,detach=True,numpy=False)
        # self._printw("Using exp in the confusion matrices.")
        # output_probas = np.exp(output_probas)
        # output_probas= np.array(torch.flatten(\
        #     torch.argmax(ass,dim=1)\
        #         .detach()).cpu())
        if(len(label_attack_any.shape)!=1):
            label_attack=np.array(torch.argmax(label_attack_any,dim=1).cpu())
        else:
            label_attack=np.array(label_attack_any.cpu())
        counters = np.zeros((self.leakage_model.get_classes(),self.leakage_model.get_classes()))
        for i in range(len(counters)):
            where = np.argwhere(label_attack == i).flatten()
            # print("where is",where)
            counters[i] = np.array(torch.sum(output_probas[where],dim = 0).cpu()) / max(1,len(where)) #2023 change
            # counters[i] = np.array(np.sum(output_probas[where], 0)) / len(where)
            # for j in range(len(counters[i])):
            #     # preds = list(output_probas[where]).count(j)
            #     print("shape wh ",output_probas[where])
            #     preds = torch.sum(output_probas[where][::,j]).cpu()
            #     counters[i][j] = preds 
        counters = np.array(counters)
        # mn = np.repeat(np.mean(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
        # std = np.repeat(np.std(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
        # counters = (counters - mn) / std
        # counters = (counters -mn) # / std
        # sum_counters = np.sum(counters,0)
        # sum_std_counters=np.sum((counters - mn) / std,0)
        hws=np.array([bin(n).count("1") for n in range(0,256)])
        if(id_in_hw_visualize):
            # self._printw("Confusion representation is BROKEN ! Diagonal is set to mean.")
            # counters[np.diag_indices_from(counters)] = np.mean(counters)
            vmax = np.max(counters)
            vmin = np.min(counters)
            # sums_std_mean = 0.0
            for i in range(1,8):
                plot_extra[i].imshow((counters.T[np.where(hws==i)]).T,vmin=vmin,vmax=vmax,norm="log")
                # print("STD",i,"is",np.std((counters.T[np.where(hws==i)]).T))
                # stds = np.std((counters.T[np.where(hws==i)]).T,0)
                # sums_std_mean+=np.mean(stds)
            #     print("Mean of STD ",i,np.mean(stds),"STD of STD",np.std(stds))
            # print("Mean of the sum of the mean of the std:",sums_std_mean/7)
            plot_extra[0].imshow(counters,norm="log")
            # plot_extra[0].legend(loc=(1.04, 0))


        else:
                # print("Raw counter",i,"has mean",np.mean(counters[indexs]),"and std",np.std(counters[indexs]),"and is",counters[indexs])
            # print("\nStandardized counter",i,"is",sum_std_counters[indexs])
        # print("\nNEXT\n")
        # for i in range(9):
            # indexs=np.where(hws==i)
            # print("Raw counter",i,"is",sum_counters[indexs])
            # print("\nStandardized counter",i,"has mean",np.mean(sum_std_counters[indexs]),"and std",np.std(sum_std_counters[indexs]),"and is",sum_std_counters[indexs])


        # np.sum(counters)
        # print("min",np.min(counters),"max",np.max((counters)))
            plot_extra.imshow(counters)

    @abstractmethod
    def attack_byte(self,byte):
        """Attack byte: launches a full attack on a byte.

        Will print the result of the attack on the terminal.

        Parameters
        ----------
        byte : int
                Byte number to attack. Some bytes may be harder to attack than others.
                
        Returns
        -------
        GE : int
                The guessing entropy.        
        """
        pass

    def set_seed(self,seed):
        """Sets the seed.

        Fixes the randomness of each component of the attack, allowing reproducibility of results.

        Parameters
        ----------
        seed : int
        """
        self.seed=seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.data_manager.set_seed(self.seed)

    @abstractmethod
    def attack_bytes(self,byte_range):
        """Launches attack_byte() method on each given byte.

        Parameters
        ----------
        byte_range : List[int],int
                On which byte(s) to launch a SCA.
        
        Returns
        -------
        key_ranking : List[int]
                The key ordered in decreased order of probability
        key_certainty :  float
                An estimated probability that GE = 1.
        """
        pass

    def _print(self,*a,**k):
        """Alias for a print."""
        print(*a,**k) if self.verbose >= 0 else lambda *a, **k: None  # print
    def _printm(self,*a,**k):
        """Prints only if verbose is greater than 0."""
        print(*a,**k) if self.verbose >= 1 else lambda *a, **k: None  # _printmetrics
    def _printd(self,*a,**k):
        """Prints only if verbose >=2. """
        print(*a,**k,flush=True) if self.verbose >= 2 else lambda *a, **k: None  # _printdebug
    def _printw(self,*a,**k):
        """Prints a given warning only if not already printed before."""
        if(a not in self._warnings):
            self._warnings.add(a)
            self._print(*a,**k)
    def _printdw(self,*a,**k):
        """Prints a given warning only if not already printed before and verbose is >=2."""
        if(a not in self._warnings):
            self._warnings.add(a)
            self._printd(*a,**k)
    def _start_record(self):
        """Starts recording the time by setting time1 attribute to current time."""
        self.time1=time()
    def _round_record(self,total_range=None):
        """Prints time elapsed since last start_record() call.

        Additionally, it prints how much time is remained, assuming total_range is not None.

        Parameters
        ----------
        total_range : int, default : None
                How many rounds are being done in total.

        """
        if(total_range==None):
            info=""
        else:
            info="Will finish at max: "+strftime('%d-%m-%Y  %H:%M:%S', gmtime( 60*60 * 2 +time() + (total_range-1)*(time()-self.time1)))
        self._printm("Finished round in "+self._get_duration()+".",info )
    def _get_duration(self,time_ = None):
        if(time_ is None):
            time_ = self.time1
        a = strftime('%dd %H:%M:%S', gmtime(time()-time_))
        d = int(a[:2]) - 1
        d = "{:02}".format(d)
        a = d + a[2:]
        return a
    def _end_record(self):
        """Prints the total duration since last start_record() call."""
        self._printm("Total duration: "+self._get_duration())
    def __str__(self):
        return str(vars(self))
    def __repr__(self) -> str:
        return str(self)
    def _get_res_name(self,info=""):
        """Get the result file name."""
        comp=""
        if(self.lr_schedule is not None):
            comp+="LR_"+str(self.lr_schedule).replace(" ","")+","#+str(self.lr_schedule[1])+","
        if(self.dk):
            comp+="DK,"
        if(self.noise):
            comp+="noise,"
        if(self.train_model==self.train_model_pruning):
            if(self.training == TrainingMethods.PRUNING_LOCAL or self.training == TrainingMethods.PRUNING_GLOBAL):
                comp+= "LTH"
            elif(self.training == TrainingMethods.PRUNING_HALF_EPOCHS_LOCAL or self.training == TrainingMethods.PRUNING_HALF_EPOCHS_GLOBAL):
                comp += "LTHH"
            comp+=str(self.get_pruning_percentage()*10).replace(".","")+","
        elif(self.train_model==self.train_model_adversarial):
            comp+="ADV,"
        comp += self.data_manager.get_res_name()
        if(info != ""):
            info += "_"
        return self.results_path+info+self.info+"_"+comp+f'dim_{self.dim},batch_{self.batch_size},{self.leakage_model},'+\
            f'epochs_{self.epochs},{self.loss},{self.model_name},{self.optimizer},'+str(self.lambdas).replace(" ","")+f",seed_{self.seed}.pdf"

    @abstractmethod
    def _get_super_name(self,info):
        pass


    def get_accuracy(self,preds,labels):
        """Get accuracy: computes the prediction's accuracy.

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
        """
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

    def _get_loss_label(self,labels):
        """Returns the loss function to be used and the labels treated accordingly."""
        if(self.loss=="mse"):
            # print("labels were",type(labels[0]))
            loss_fn = torch.nn.MSELoss(reduction='mean')
            # labels = labels.float()
            if(len(labels.shape)==1):
                labels = torch.from_numpy(to_categorical(labels.cpu(),self.leakage_model.get_classes())).to(self.data_manager.device)
                labels = labels.float()
                # print("labels are now",labels)
        elif(self.loss=="nll" or self.loss=="nlll"):
            loss_fn=torch.nn.NLLLoss(reduction='mean') #weight = self.leakage_model.get_class_balance().to(self.data_manager.device),
            labels = labels.long()
            if(len(labels.shape) != 1 ):
                labels=torch.argmax(labels,dim=1)   
        elif("cross" in self.loss):
            loss_fn = torch.nn.CrossEntropyLoss(weight = self.leakage_model.get_class_balance().to(self.data_manager.device),reduction='mean')
        else:
            raise Error("Loss",self.loss," not implemented. Check your spelling again.")
        return loss_fn,labels

    def get_sensitivity_tensors(self,model, traces, labels):
        """Computes the sensitivity value, as from :footcite:t:`[104]Timon_2019`. 

        It is computed using the derivative of the loss function with respect to the input.

        .. footbibliography::
        
        Parameters
        ----------
        model : torch.nn.Module
                The trained model.
        traces : List[List[float]] 
                The power traces.
        labels : List[int]
                The training labels.

        Raises
        -------
        AttributeError
                If the traces do not requires_grad().
        Returns
        -------
        torch.tensor
                A tensor of size ns with the according sensitivity values. 

        """
        def help_batch_sensi(traces_subset,labels_subset,not_h=None):
            """Batched sensitivity calculation for big data management."""
            if(not traces_subset.requires_grad):
            # print("did not")
                must_require = False
                traces_subset.requires_grad_(True)
            else:
                must_require = True
            preds = model(traces_subset)
            # print("preads are ",preds)
            if(self.loss == "mse"):
                preds=torch.exp(preds)
            traces_subset.retain_grad()
            if(traces_subset.grad != None):
                traces_subset.grad.zero_()
            tmp = loss_fn(preds, labels_subset)

            tmp.backward(retain_graph=True)
        #  print("grads are ",traces_subset.grad)
        #    traces_subset.grad.requires_grad_()
            if(self.sensitivity_mode == SensitivityMode.CLASSIC):
                #self._printd("before")
                grads = traces_subset.grad #* traces_subset
                #grads.requires_grad_()
               # print("ns is ",self.data_manager.get_ns())
               # pca_trace = PCA(250).fit_transform(np.array(torch.reshape(traces_subset,(250,500)).detach().cpu()))
                #print("shape pca is ",len(pca_trace),len(pca_trace[0]))
            #    pca_trace = torch.repeat_interleave(torch.from_numpy(pca_trace).to(self.data_manager.device),2 , dim=1)
              #  pca_trace = torch.reshape(pca_trace,(250,1,500))
        #        not_h = torch.reshape(not_h,(not_h.shape[0],1,not_h.shape[1]))
                grads = grads * (traces_subset.detach())# torch.abs(not_h) #pca_trace.detach() #traces_subset.detach()
                # print("grads are",grads)
                # torch.mul(traces_subset.grad, traces_subset)
            elif(self.sensitivity_mode == SensitivityMode.ON_RAW_TRACE):
                grads = traces_subset.grad
                not_h = torch.reshape(not_h,(not_h.shape[0],1,not_h.shape[1]))
                grads = grads * not_h 
            elif(self.sensitivity_mode == SensitivityMode.ABSOLUTE_VALUES):
                grads = torch.abs(traces_subset.grad)
            elif(self.sensitivity_mode == SensitivityMode.GRAD_CAM):
                raise NotImplementedError
            elif(self.sensitivity_mode == SensitivityMode.GRAD_CAM_PLUS_PLUS):
                raise NotImplementedError
            #grads.requires_grad_(True)
            traces_subset.requires_grad_(must_require)
           # grads.retain_grad()
            return grads

        loss_fn,labels=self._get_loss_label(labels)
        #for p in model.parameters():
         #   p.requires_grad = False
        batch_size = self.batch_size
        data_set=NRFDataset3(traces,labels,self.data_manager.x_attack_not_horiz,self.data_manager.device)
        train_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle = self.seed is None)
        sensis = None
        detach = False
        for i,(t,l,t_not_h) in enumerate(train_loader):
            if(i==0):
                sensi=help_batch_sensi(t,l,t_not_h)
                if(detach):
                    sensi = sensi.detach()
                sensis = sensi
            else:
                sensi=help_batch_sensi(t,l,t_not_h)
                if(detach):
                    sensi = sensi.detach()
                sensis = torch.cat((sensis,sensi))
      #  for p in model.parameters():
       #     p.requires_grad = True
        # print("sensis before sum",sensis.shape)
        # print("sensis after sum ",torch.sum(sensis, 0).shape)
        return torch.sum(sensis, 0) #[0]

    
    def turn_output_probas_to_key_proba(self,output_probas, plaintexts,byte_position,masks = None):
        '''Turns the probabilities from being a probability on the value of the label sorted on the label, to a probability
            on the secret key sorted with the secret key. In case of huge datasets, will use multiprocessing.

        Parameters
        ----------
        output_probas : List[float]
                Output probabilites of the ML model (model.predict(x_attack)) with to_categorical y_train data.
        plaintexts : List[List[int]]
                List of plaintexts.
        byte_position : int
                The target byte index.

        Returns
        -------
        List[List[float]]
                List of probabilities sorted on the secret key for each plaintext

        Raises
        ------
        Errors
                When the data went through GPU memory and back to cpu via tensor.cpu().numpy() and
                a parallel computing is launched.
                When the CPU memory is being exhausted.

        '''

        self._printw("USING SOFTMAX ON OUTPUT PROBAS DISABLED")
        if(len(output_probas)>400):  #use parallelism only for big data
            # print("CPU COUNT",cpu_count())
            buckets = min(16,max(cpu_count()-1,1))
           # set_start_method('spawn')
            with Pool(buckets) as p:
                ptx_arg = np.array_split(plaintexts[:,byte_position],buckets)
                if(masks is not None):
                    mask_arg = np.array_split(masks[:,byte_position],buckets)
                else:
                    mask_arg = [None] * buckets
                proba_arg = np.array_split(output_probas,buckets)
                arg = zip(ptx_arg,proba_arg,[self.leakage_model] * buckets,[byte_position] * buckets,mask_arg)
                output_probas_on_keys_par = p.starmap(_help_turn_output_probas_to_key_proba, arg)
                res = list()
                for r in output_probas_on_keys_par:
                    for r_i in r:
                        res.append(r_i)
            return np.array(res)
        if(self.leakage_model.is_ID()):
                output_probas_on_keys=np.zeros((len(output_probas),256))
                for i in range(len(output_probas)):
                    respective_real_keys=self.leakage_model.recover_key_byte_hypothesis(plaintexts[i][byte_position],byte_position,masks[i][byte_position])
                    sorted_idx=respective_real_keys.argsort()
                    output_probas_on_keys[i] = np.array(output_probas[i][sorted_idx])  #set output_probas_on_keys[i] to Probability(key=i)
        else:
            output_probas_on_keys=np.zeros((len(output_probas),256))
            # output_probas = torch.nn.Softmax(dim=1)(output_probas)
            for i in range(len(output_probas)):
                respective_real_keys= self.leakage_model.recover_key_byte_hypothesis(plaintexts[i][byte_position],byte_position,masks[i][byte_position])
                for j in range(len(respective_real_keys)):
                    for real_key in respective_real_keys[j]:
                        output_probas_on_keys[i][real_key] = output_probas[i][j] #/ len(respective_real_keys[j]) #removing the  /len() is much better
        return output_probas_on_keys

    
    def get_fast_GE(self,model, traces, plaintext_attack, byte, key_byte, fast_size=500,masks = None):
        """Computes the fats guessing entropy by only considering a subset of the plaintexts/keys.

        Parameters
        ----------
        model : tensorflow.keras.models
                ML model.
        traces : List[List[float]]
                Validation/attack traces.
        plaintext_attack : List[List[int]]
                Validation or attack plaintexts.
        byte : int
                Target byte index.
        key_byte : int
                The value of the target key byte
        fast_size : int, default : 500
                How many traces to consider at maximum to compute the Guessing Entropy.
        
        Returns
        -------
        int
                Returns the Guessing Entropy on a subset of the traces.
        
        """
        if(fast_size is not None):
            fast_size = min(fast_size, len(traces))
        traces = traces[:fast_size]
        plaintext_attack = plaintext_attack[:fast_size]
        output_probas = self._batch_predict(model,traces).detach().cpu()#,verbose=0)
        
        output_probas_on_keys = self.turn_output_probas_to_key_proba(
            output_probas, plaintext_attack.cpu(), byte,masks)
        return Attack.get_GE(output_probas_on_keys, key_byte)
    @staticmethod
    def get_GE(output_proba_on_key, key_byte):
        '''Computes the guesing entropy for one byte.

        Parameters
        ----------
        output_proba_on_key : List[List[float]]
                List of probabilities sorted on the secret key for each plaintext.
        key_byte : int
                The value of the target key byte.

        Returns
        -------
        int
                The Guessing Entropy of the attack.
        '''
        probas = np.sum((output_proba_on_key), axis=0) #removed np.log
        k = 256
        idx = np.argpartition(probas, -k)[-k:]  # Indices not sorted
        guesses = idx[np.argsort(probas[idx])][::-1]  # Indices sorted by value from largest to smallest
        for i, guess in enumerate(guesses):
            if(guess == key_byte):
                return i+1

    @staticmethod
    def get_keys_from_probas(output_proba_on_key):
        probas = np.sum((output_proba_on_key), axis=0)  #  -> each key has a proba with it
        guesses = heapq.nlargest(256, range(len(probas)), probas.take)  #  -> order those probas, highest first
        certainty0 = 1-math.e**(-(probas[guesses[0]] - probas[guesses[1]]))# / 2.0  # -> p(key is first) - p(key is second)
        certainty = math.e**(-probas[guesses[0]])# / 2.0  # -> p(key is first) - p(key is second)
        #print("res are ",certainty,guesses)
        s  = 0.0
        for g in probas:
            s += g
        
      #  np.median(probas)
        
       # print("mean:",np.mean(probas),"median:",np.median(probas),(probas))
        perce = 1 - abs(probas[guesses[0]])*100 / abs(s)
        print("certainty is ",certainty," probas is ",probas[guesses[0]]," next ",probas[guesses[1]]," exped: ",math.e**probas[guesses[1]]," SUM :",s,"c0",certainty0," perce : ",perce)
        return guesses,certainty

    @staticmethod
    def get_GE_from_sensitivity(sensitivity):
        ''' Computes the GE of the non profiling process for one byte.

        Parameters
        ----------
        sensitivities : List[List[float]]
                List of sensitivities, ** with the sensitivity of the right key at the end.**
        
        Returns
        -------
        int
                The Guessing Entropy of the attack
        '''
        # print("Sensi shape ",sensitivity)
        sensis_scores=np.amax(sensitivity,axis=-1)
        sensis_ids=(heapq.nlargest(256, range(len(sensis_scores)), sensis_scores.take))
        return sensis_ids.index(len(sensitivity)-1) + 1

    @staticmethod
    def get_keys_from_sensitivity(sensitivity,keys):
        ''' Computes the ranking of the keys given sensitivity values. 

        This function is to be used in blind attacks.

        Parameters
        ----------
        sensitivities : List[List[float]]
                List of sensitivities.
        
        Returns
        -------
        int
                The ranking of the keys.
        '''
        sensis_scores=np.amax(sensitivity,axis=-1)
        sensis_ids=(heapq.nlargest(256, range(len(sensis_scores)), sensis_scores.take))  #  get index of highest
        # First idea for certainty: ratio between spike size and difference between 2 best spikes:
        certainty = [(sensis_scores[sensis_ids[0]] - sensis_scores[s]) / sensis_scores[sensis_ids[0]] for s in sensis_ids[1:]]
        return np.array(keys)[sensis_ids],certainty
    
    
    def _batch_predict(self,model,traces,detach=True,numpy = False):
        """Performs a conventional model prediction in batches to reduce instant memory usage.
        """
        data_set=PredictionDataset(traces,device = self.data_manager.device,numpy=numpy)
        train_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle = False)  #NEVER set shuffle = True
        preds = None
        for i,(t) in enumerate(train_loader):
           # print("t shape is ",t.shape)
            pred=model(t)
            if(i==0):
                if(detach):
                    pred = pred.detach()
                if(numpy):
                    pred=pred.cpu()
                preds = pred
            else:
                if(detach):
                    pred = pred.detach()
                if(numpy):
                    preds = np.concatenate((preds,pred.cpu()))
                else:
                    preds = torch.cat((preds,pred))
        return preds
        

    def train_model_pruning(self,x_prof,label_prof,x_prof_val,label_prof_val,plaintext_prof_val,mask_val,key,byte,model=None,get_metrics=True,fast=False,sensi_reg = False,fast_sensi = False):
            """ Trains the model using LTH pruning methods. 

            Parameters
            ----------
            x_prof : List
                    The attack traces used for unprofiled training. Should have the shape (Na,1,Ns).
            label_prof : List[int]
                    The labels corresponding to label_prof.
            x_prof_val : List
                    Traces to use for metrics computations (e.g. sensitivity computation). Can be
                    the same as x_prof.
            label_prof_val : List[int]
                    The labels corresponding to x_prof_val.
            plaintext_prof_val ; List[int]
                    The plaintexts corresponding to x_prof_val
            key : List[List[int]]
                    The attack key(s). 
            byte : int
                    Target byte index.
            model : nn.Module, default : None
                    If not set to None, will re-use the model given instaed of creating a new.
            get_metrics : bool, default : True
                    Whether to calculate validation accuracy and training accuracy.
            fast : bool, default : False
                    Whether to calculate the fast Guessing Entropy during learning. May induce
                    additionnal delays.
            Returns
            -------
            model : torch.nn.Module
                    The trained model.
            """
            
            def prune_model_global_unstructured(model, proportion, globally ):
                module_tups = []
                i=0
                is_input_module = True
                proportion_tmp = proportion
                for module in model.modules():
                    if(hasattr(module,"weight")):
                        if(is_input_module):#or i == 3):
                            proportion = 1
                            is_input_module = False
                        else:
                            proportion = proportion_tmp
                        #print("modulde is",i)
                        if(not globally):
                            prune.l1_unstructured(module, 'weight', proportion)#, n=1, dim=1)
                        module_tups.append((module, 'weight'))
                        i+=(len(module.weight))
                #return model
                proportion = proportion_tmp
                if(isinstance(proportion,int)):
                    proportion = i - proportion
                if(globally):
                    prune.global_unstructured(
                    parameters=module_tups, pruning_method=prune.L1Unstructured,
                    amount=proportion
                    )
                return model
            if(model is not None):
                print("Warning: discarding non null model input.")
            model=self._get_model()
            model_initial = copy.deepcopy(model).state_dict()
            lambdas = self.lambdas
            self.lambdas = None
            res = self.train_model_default(x_prof,label_prof,x_prof_val,label_prof_val,plaintext_prof_val,mask_val,key,byte,model=model,get_metrics=False,fast=False,sensi_reg=False,fast_sensi = False)
            self.lambdas = lambdas
            model = res[0]
            globally = self.training in [TrainingMethods.PRUNING_GLOBAL,TrainingMethods.PRUNING_HALF_EPOCHS_GLOBAL]
            model = prune_model_global_unstructured(model,proportion =self.get_pruning_percentage(),globally = globally)  #DEFAULT_PRUNING_PERCENTAGE)
            new_dict = copy.deepcopy(model_initial)
            for k in model_initial.keys():
                if("weight" in k):
                    new_dict[k+"_orig"] = new_dict.pop(k)
            for k in model.state_dict():
                if("weight_mask" in k):
                    new_dict[k] = model.state_dict()[k]
                elif("bias" in k):
                    new_dict[k] = model.state_dict()[k]
            model.load_state_dict(new_dict)
            initial_epochs = self.epochs
            if(self.training == TrainingMethods.PRUNING_HALF_EPOCHS_LOCAL):
                self.epochs = self.epochs // 2  #use less epochs for second training. In [49], they use 1) 300 epochs and 2) 50 epochs.
            res = self.train_model_default(x_prof,label_prof,x_prof_val,label_prof_val,plaintext_prof_val,mask_val,key,byte,model=model,get_metrics=get_metrics,fast=fast,sensi_reg=sensi_reg,fast_sensi = fast_sensi)
            self.epochs = initial_epochs
            return res
    
    @abstractmethod
    def train_model_default(self,**kwargs):
        """Default methods for training."""
        pass

    def _set_training(self):
        if(self.training == TrainingMethods.DEFAULT):
            self.train_model = self.train_model_default
        elif(self.training in [TrainingMethods.PRUNING_LOCAL,TrainingMethods.PRUNING_HALF_EPOCHS_LOCAL, TrainingMethods.PRUNING_GLOBAL,TrainingMethods.PRUNING_HALF_EPOCHS_GLOBAL]):
            self.train_model = self.train_model_pruning 
        elif(self.training == TrainingMethods.ADVERSARIAL):
            self.train_model = self.train_model_adversarial
        elif(self.training == TrainingMethods.CROSS_VALIDATION):
            self.train_model = self.train_model_cross_validation
        elif(self.training == TrainingMethods.CUSTOM):
            self.train_model = self.train_model_custom
        else:
            raise Error("Invalid training argument. Should be of type Trainingmethods.")
            
    def attack_key(self,timeout = 2 * 60 * 60,reuse_this_key_probas = None):
        """Blind key attack.
        
        Performs a real-setting attack on the whole key. Will first call attack_bytes on all byte,
        and then make a best-effort brute-force search to find the correct key if any ciphertext was
        available in the dataset.

        Parameters
        ----------
        timeout : int, default : 7200
                After how many seconds to stop the brute-force search.
        reuse_this_key_probas : str, default : None
                This argument should be the path to a .npy file containing a previous attack_bytes(range(16))
                result.
        
        
        """

        def get_key_guess_range(key_ranking,key_certainty,min_certainty,trial_number=0,max_guess = 4):
            #For best results, set the max_guess to the estimated upper bound of the GE of the most diffcult byte.
            if(key_certainty >= self.threshold):
                return key_ranking[0:1]
            max_guess += trial_number
            a = - math.log(1.0/max_guess) / (self.threshold - min_certainty)
            guess_range = min(round(max_guess * math.e**(-((key_certainty-min_certainty)*a)) + trial_number),255)
            return key_ranking[0:guess_range]
        if(reuse_this_key_probas is not None):
            storage_name = reuse_this_key_probas
        else:
            storage_name = os.path.dirname(__file__) + "/KeyProbasCache/" + self._get_super_name(self.data_manager.file_name_attack.replace(".h5","")).replace(".pdf","").replace(self.results_path,"")+".npy"
        if(not os.path.exists(storage_name)):
            key_probas = self.attack_bytes(range(16))
            np.save(storage_name,np.array(key_probas, dtype=object))
        else:
            key_probas = np.load(storage_name, allow_pickle=True)
            print("Re-using previous attack results.")
        ptx_check,cipher_check = self.data_manager.get_check_data()
        if(cipher_check is None or ptx_check is None):
            print("Since no attack ciphertext is available, you only have the key probability distribution.")
            return key_probas
        #print("Key Probas is ",key_probas)
        best_key = [k[0][0] for k in key_probas]
        print("best key is ",best_key)
        if(self.leakage_model.check_key(ptx_check,cipher_check,best_key)): #self.data_manager.key_attack[0]
            print("Incredible. All bytes had GE = 1. The key is ",best_key,".")
            return best_key

        print(key_probas[1],type(key_probas[1]))
        if(hasattr(key_probas[1],"__len__")):
            key_ranking,key_certainty = [k[0] for k in key_probas],[0.25 for k in key_probas]
        else:
            key_ranking,key_certainty = [k[0] for k in key_probas],[k[1] for k in key_probas]
        key_certainty = np.array(key_certainty)
        key_ranking = np.array(key_ranking)
        certainty_ranking = (heapq.nlargest(16, range(len(key_certainty)), key_certainty.take))
        key_unwrapper = [certainty_ranking.index(i) for i in range(16)]
        key_ranking,key_certainty = key_ranking[certainty_ranking],key_certainty[certainty_ranking]
        min_certainty = key_certainty[15]
        while(min_certainty > self.threshold):
            self.threshold*=1.2
        start_time = time()
        guess_count = 0
        key_ranking_visited = [0] * 16
        for i in range(256):
            if(i>0):
                key_ranking_visited = [len(k_r) for k_r in key_ranking_loop]
                self._printd(np.array(key_ranking_visited)[key_unwrapper])
            self._printd("increase i")
            key_ranking_loop = [get_key_guess_range(k_rc[0],k_rc[1],min_certainty,i,max_guess=2) for k_rc in zip(key_ranking,key_certainty)]
            for i_a,b_a in enumerate(key_ranking_loop[0]):
             for i_b,b_b in enumerate(key_ranking_loop[1]):
              for i_c,b_c in enumerate(key_ranking_loop[2]):
               for i_d,b_d in enumerate(key_ranking_loop[3]):
                for i_e,b_e in enumerate(key_ranking_loop[4]):
                 for i_f,b_f in enumerate(key_ranking_loop[5]):
                  for i_g,b_g in enumerate(key_ranking_loop[6]):
                   for i_h,b_h in enumerate(key_ranking_loop[7]):
                    for i_i,b_i in enumerate(key_ranking_loop[8]):
                     for i_j,b_j in enumerate(key_ranking_loop[9]):
                      for i_k,b_k in enumerate(key_ranking_loop[10]):
                       for i_l,b_l in enumerate(key_ranking_loop[11]):
                        for i_m,b_m in enumerate(key_ranking_loop[12]):
                         for i_n,b_n in enumerate(key_ranking_loop[13]):
                          for i_o,b_o in enumerate(key_ranking_loop[14]):
                           new = i_o >= key_ranking_visited[14]\
                                  or i_n >= key_ranking_visited[13]\
                                  or i_m >= key_ranking_visited[12]\
                                  or i_l >= key_ranking_visited[11]\
                                  or i_k >= key_ranking_visited[10]\
                                  or i_j >= key_ranking_visited[9]\
                                  or i_i >= key_ranking_visited[8]\
                                  or i_h >= key_ranking_visited[7]\
                                  or i_g >= key_ranking_visited[6]\
                                  or i_f >= key_ranking_visited[5]\
                                  or i_e >= key_ranking_visited[4]\
                                  or i_d >= key_ranking_visited[3]\
                                  or i_c >= key_ranking_visited[2]\
                                  or i_b >= key_ranking_visited[1]\
                                  or i_a >= key_ranking_visited[0]
                           if(new):
                            index = 0
                           else:
                            index = key_ranking_visited[15]
                           for i,b_p in enumerate(key_ranking_loop[15][index:]):
                               guess_count += 1
                               key_loop = np.array([b_a,b_b,b_c,b_d,b_e,b_f,b_g,b_h,b_i,b_j,b_k,b_l,b_m,b_n,b_o,b_p])
                               key_guessed = key_loop[key_unwrapper]
                               #self._printd("testing ",key_guessed)
                               if(self.leakage_model.check_key(ptx_check,cipher_check,key_guessed)):
                                   print("Success ! The correct key is ",key_guessed," found after ",guess_count," guesses in ",strftime('%H:%M:%S', gmtime(time()-start_time)))
                                   return key_guessed
                               elif(time() > start_time + timeout):
                                   print("No success within the timeout...")
                                   return None


def _help_turn_output_probas_to_key_proba(ptx,probas,leakage_model,byte_position,masks):
    """Helper function for parallel computation. """
    output_probas_on_keys=np.zeros((len(ptx),256)) 
    if(leakage_model.is_ID()):
        for i in range(len(ptx)):
            respective_real_keys=leakage_model.recover_key_byte_hypothesis(ptx[i],byte_position,masks[i]) 
            sorted_idx=respective_real_keys.argsort()
            output_probas_on_keys[i] = np.array(probas[i][sorted_idx])
    else:
        for i in range(len(ptx)):
            respective_real_keys= leakage_model.recover_key_byte_hypothesis(ptx[i],byte_position,masks[i])
            for j in range(len(respective_real_keys)):
                for real_key in respective_real_keys[j]:
                    output_probas_on_keys[i][real_key] = probas[i][j]  #/ len(respective_real_keys[j])  # +1: ptx offset
    return output_probas_on_keys