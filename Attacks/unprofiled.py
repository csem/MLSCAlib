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
# brief     : ML non profiled attack on AES
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import DataLoader
from Attacks.attack import DEFAULT_SGD_MOMENTUM, Attack, TrainingMethods,SensitivityMode
from Ciphers.AES_leakage import AESLeakageModel
from Data.custom_datasets import NRFDataset
from Data.data_manager import DataManager
from error import Error
from Data.custom_manager import ImbalanceResolution, CustomDataManager
from torch.optim.lr_scheduler import StepLR,OneCycleLR

class UnProfiled(Attack):
    def __init__(self,epochs=15,model_name="mlp",batch_size=100,loss="nlll",optimizer="Adam",leakage_model=AESLeakageModel("HW","SBox"),\
                    verbose=1,lambdas= [0.03,0.03,0.0008,0.0008],dk=False,noise=False,seed=None,split_non_profiling_attack_set=False,\
                    sensitivity_mode=SensitivityMode.CLASSIC,data_manager = DataManager(na = "all",ns = "all",nt = 0,fs=0,file_name_attack = 'file_name_attack.h5',\
                        databases_path="/path/to/Databases/"),results_path="/path/to/PlotsResults/",\
                            training = TrainingMethods.DEFAULT,info="",dim=0,LTH = 0.8,lr_schedule=None,learning_rate="default"):
        """ Initializes an Attack.

        Parameters
        ----------
        epochs : int, default : 50
                How many epochs to train the NN.
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
        lambdas: List[Float], default : [0.03,0.03,0.0008,0.0008]
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
                Which seed to use. If set to None, will use a default standard value of 5437.
        split_non_profiling_attack_set : bool, default : True
                Whether to separate the training data from the data used to calculate the
                sensitivity and validation accuracies. If you plan to use a MeshNN, setting it
                to False will avoid Errors.
        sensitivity_mode : SensitivityMode, default : SensitivityMode.CLASSIC
                Which algorithm to choose for the sensitivity calculation.
        data_manager : Data.DataManager, default : DataManager()
                The datamanager handles anything related to the data.
        results_path : str, default : "/path/to/PlotsResults/"
                Where to store the learning plots.
        training : attacks.attack.TrainingMethods, default : TrainingMethods.DEFAULT
                Which training method to use.
        info : str, default : ""
                A small text to insert in the result file name.
        dim : int, default : 0
                On which dimension/axis to apply the softmax filter in the NN.
        LTH : float, default : 0.7
                Percentage of the weights to mask in the LTH pruning. Only used if the training
                method is set to TrainingMethods.PRUNING or TrainingMethods.PRUNING_HALF_EPOCHS.
        lr_schedule : List[int,float] | None. Default : None.
                learning rate Scheduler parameter. If not None, after each lr_schedule[0] epochs,
                multiply the learning rate by lr_schedule[1].
        """
        super().__init__(model_name=model_name,batch_size=batch_size,loss=loss,optimizer=optimizer,\
                        leakage_model=leakage_model,verbose=verbose,lambdas=lambdas,dk=dk,noise=noise,seed=seed,\
                        data_manager=data_manager,results_path=results_path,training=training,info=info,dim=dim,\
                            lr_schedule=lr_schedule,learning_rate=learning_rate)
        self.epochs=epochs
        self.split_non_profiling_attack_set=split_non_profiling_attack_set
        self.sensitivity_mode=sensitivity_mode
        self.LTH = LTH
        self._printd(self)

    def get_pruning_percentage(self):
        return self.LTH

    def attack_bytes(self,byte_range,guess_list=range(256)):
        """Launches attack_byte() method on each given byte.

        Parameters
        ----------
        byte_range : List[int],int
                On which byte(s) to launch a SCA.
        """
        if(isinstance(byte_range,int)):
            byte_range=[byte_range]
        res = []
        for i,byte in enumerate(byte_range):
            self._printd(f"Attacking byte {byte}...")
            res.append(self.attack_byte(byte,guess_list,i == len(byte_range)-1))
        return res
            
    def attack_bytes_with_min_traces(self,byte_range,guess_list=range(256)):
        """Launches attack_byte_with_min_traces() method on each given byte.
        
        Parameters
        ----------
        byte_range : List[int],int
                On which byte(s) to launch a SCA.
        guess_list : int, default: range(256)
                Against which byte values to compare the right key guess. 
        """
        if(self.train_model != self.train_model_default):
            raise NotImplementedError("If you do not want to use default learning, you have to use default attack_bytes method.")
            
        if(isinstance(byte_range,int)):
            byte_range=[byte_range]
        for byte in byte_range:
            self._printd(f"Attacking byte {byte}...")
            self.attack_byte_with_min_traces(byte,guess_list)
    
    def _prepare_figures(self):
        """Instantiates the plots."""
        fig=plt.figure(figsize=(15,15))
        plt.subplots_adjust(hspace=0.7)
        plot_sensi = plt.subplot2grid((8, 2), (0, 0), rowspan=2,colspan=2,fig=fig)
        plot_sensi.set_title("Accumulated Sensitivity")
        plot_acc = plt.subplot2grid((8, 2), (2, 0), rowspan=2, colspan=2,fig=fig)
        plot_acc.set_title("Validation Accuracy")
        plot_acc_train = plt.subplot2grid((8, 2), (4, 0), rowspan=2, colspan=2,fig=fig)
        plot_acc_train.set_title("Training Accuracy")

        plot_extra=plt.subplot2grid((8,4),(6,0),rowspan=2,colspan=1,fig=fig)
        plot_extra.set_title("Right Key Confusion")
        plot_extra1=plt.subplot2grid((8,4),(6,1),rowspan=2,colspan=1,fig=fig)
        plot_extra1.set_title("Right Key Confusion Val")
        plot_extra2=plt.subplot2grid((8,4),(6,2),rowspan=2,colspan=1,fig=fig)
        plot_extra2.set_title("Wrong Key Confusion")
        plot_extra3=plt.subplot2grid((8,4),(6,3),rowspan=2,colspan=1,fig=fig)
        plot_extra3.set_title("Wrong Key Confusion Val")

        return plot_sensi,plot_acc,plot_acc_train,[plot_extra,plot_extra1,plot_extra2,plot_extra3]

    # def _draw_class_predictions(self,plot_extra,model_,x_attack_any,label_attack_any):
    #     """Draws the confusion matrices."""
    #     ass = self._batch_predict(model_,x_attack_any,detach=True)
    #     res3= np.array(torch.flatten(\
    #         torch.argmax(ass,dim=1)\
    #             .detach()).cpu())
    #     if(len(label_attack_any.shape)!=1):
    #         label_attack=np.array(torch.argmax(label_attack_any,dim=1).cpu())
    #     else:
    #         label_attack=np.array(label_attack_any.cpu())
    #     counters = np.zeros((self.leakage_model.get_classes(),self.leakage_model.get_classes()))
    #     for i in range(len(counters)):
    #         where = np.argwhere(label_attack == i)
    #         for j in range(len(counters[i])):
    #             preds = list(res3[where]).count(j)
    #             counters[i][j] = preds 
    #     counters = np.array(counters)
    #     mn = np.repeat(np.mean(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
    #     std = np.repeat(np.std(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
    #     counters = (counters - mn) / std
    #     plot_extra.imshow(counters)
    def _draw_on_plots(self,plot_sensi,plot_acc,plot_acc_train,plot_extra,sensitivity,accuracies,train_acc,fast_GEs,correct,key_guess):
        """Draws the different metrics on the given plots."""
        colors=["b","g","c","m","y","k","peru","purple","olive","pink","forestgreen","moccasin"]
        if(correct):
            plot_sensi.plot(sensitivity,'red',linewidth=4,alpha=0.5)
            plot_acc.plot(accuracies,'red',linewidth=4,alpha=0.5)  #left empty for the class preds !
            plot_acc_train.plot(train_acc,'red',linewidth=4,alpha=0.5)
            #plot_extra.plot(fast_GEs,'red',linewidth=4,alpha=0.5)
        else:
            #print("sensitivity ",sensitivity," train acc ",train_acc)
            plot_sensi.plot(sensitivity,colors[key_guess % len(colors)],linewidth=1,alpha=0.4)#,label="Max Random "+str(sum_sensi)+sum_abs_sensi)
            plot_acc.plot(accuracies,colors[key_guess % len(colors)],linewidth=1,alpha=0.4)#,label="Random "+str(sum_acc))
            plot_acc_train.plot(train_acc,colors[key_guess % len(colors)],linewidth=1,alpha=0.4)
           # plot_extra.plot(fast_GEs,colors[key_guess % len(colors)],linewidth=1,alpha=0.4)

    def attack_byte(self,byte,guess_list=range(256),display_sensi_hist = False,split_rate = 0.95):
        """Attack byte: launches a full attack on a byte. 

        Will print the result of the attack on the terminal.

        Parameters
        ----------
        byte : int
                byte number to attack. Some bytes are harder to attack.
        guess_list : list, default: range(256)
                Against which byte values to compare the right key guess. 
        display_sensi_hist : bool, default : False
                Whether to prompt a window tieh the history of the sensitivities across the epochs.
        split_rate : float, default : 0.88
                Which percentage of the traces to discard during training and use for validation.
                Warning: if you use a MeshNN, set it carefully !
        Returns
        -------
        int
                The guessing entropy.
        """
        get_fast_GE_if_not_loss = False 
        models=list()
        if(self.verbose==1 or self.verbose == 2):
            plots = self._prepare_figures()  #plot_sensi,plot_acc,plot_acc_train,plot_extra

        data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=0,leakage_model = self.leakage_model,split_val = self.split_non_profiling_attack_set,split_rate=split_rate)
        if(self.data_manager.blind):
            raise Error("This class does not handle real attacks. Use the BlindUnProfiled class instead.")
        x_attack_train,label_attack_train,plaintext_attack_train,key_attack,x_attack_val,label_attack_val,plaintext_attack_val=data
        self._printdw("NA:",len(x_attack_train),", NS:",len(x_attack_train[0]))
        key_guesses = list(guess_list)
        
        right_key=key_attack[0][byte]
        if(right_key in key_guesses):
            key_guesses.pop(key_guesses.index(right_key))
        key_guesses.append(right_key)

        sensitivities=list()
        sensitivities_slider = list()
        self._start_record()
        for i,key_guess in enumerate(key_guesses):
            if(not(i==0 and key_guess == 0)):
                data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=key_guess,leakage_model = self.leakage_model,split_val = self.split_non_profiling_attack_set,split_rate=split_rate)
            x_attack_train,label_attack_train,plaintext_attack_train,key_attack,x_attack_val,label_attack_val,plaintext_attack_val=data
            self._printdw("train len",len(x_attack_train),"val len ",len(x_attack_val))
            x_attack_val.requires_grad_(False)
            model_,sensitivity,accuracies,fast_GEs,train_acc,sensi_slider = self.train_model(x_attack_train,\
                            label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte=byte,model=None,get_metrics=True,fast=get_fast_GE_if_not_loss)#,info=key_guess==right_key)
            models.append(model_)
            sensitivities_slider.append(sensi_slider)
            sensitivities.append(sensitivity)
            sensitivity=np.abs(sensitivity)
            if(self.verbose==1 or self.verbose == 2):
                if(key_guess==key_attack[0][byte]):
                    self._draw_class_predictions(plots[-1][0],model_,x_attack_train,label_attack_train)
                    self._draw_class_predictions(plots[-1][1],model_,x_attack_val,label_attack_val)
                elif(key_guess == key_guesses[0]):
                    self._draw_class_predictions(plots[-1][2],model_,x_attack_train,label_attack_train)
                    self._draw_class_predictions(plots[-1][3],model_,x_attack_val,label_attack_val)
                self._draw_on_plots(*plots,sensitivity,accuracies,train_acc,fast_GEs,correct = key_guess==key_attack[0][byte],key_guess=key_guess)
            if(key_guess == 0):
                self._round_record(len(key_guesses))
        self._end_record()
        ge = Attack.get_GE_from_sensitivity(sensitivities)
        self._print("Guessing entropy =",ge," for ",len(key_guesses)," guesses on byte ",byte)
        if(self.verbose!=0):
            path_res_file=self._get_res_name(byte,len(key_guesses),ge)
            
            plt.savefig(path_res_file)
            plt.show(block=False)
            plt.close()
            if(display_sensi_hist and self.epochs>1 and self.verbose>=2):
                display_sensi_slider(sensitivities_slider)
        return ge

    def _get_res_name(self,byte,key_guesses_len,ge):
        return super()._get_res_name(f"non_profiled,byte_{byte},guesses_{key_guesses_len},GE_{ge},{self.sensitivity_mode.name}") #.replace(".pdf","") 
    def _get_super_name(self,info=""):
        return super()._get_res_name(info)
    def attack_byte_with_min_traces(self,byte,guess_list=range(256)):
        """Deprecated. Function to guess a key byte.
        
        Performs an unprofiled ML SCA as described in :footcite:t:`[104]Timon_2019` and  
        :footcite:t:`[107]10.1145/3474376.3487285`. It will train each model (resp. to a key guess) 
        with 1 batch until the GE equals 1. Warning: if the guess range (self.guess_range) is too
        small, the result may look random.

        .. footbibliography::
        
        Parameters
        ----------
        byte : int
                byte number to attack. The 3,12 and 14 are the most difficult  :footcite:t:`[107]10.1145/3474376.3487285`.
        guess_list : int, default: range(256)
                Against which byte values to compare the right key guess. 
        Returns
        -------
        List[float]            
                Returns a list of sensitivities, of size [(self.guess_range,ns)]

        """
        if(isinstance(self.data_manager,CustomDataManager) and self.data_manager.imbalance_resolution is not ImbalanceResolution.NONE):
            raise Error("Min trace with imbalance resolution is not supported.")
        if(self.train_model != self.train_model_default or self.data_manager.blind):
            raise NotImplementedError("Unprofiled attacks with a minimum # attack trace indicator is only \
            implemented for default learning.")
        data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=0,leakage_model = self.leakage_model)
        x_attack_train,label_attack_train,plaintext_attack_train,key_attack,x_attack_val,label_attack_val,plaintext_attack_val=data
        
        right_key=key_attack[0][byte]
        key_guesses = list(guess_list)
        if(right_key in key_guesses):
            key_guesses.pop(key_guesses.index(right_key))
        key_guesses.append(right_key)

        labels_train=[torch.empty((0),device = self.data_manager.device,dtype=int)] * (len(key_guesses))
        labels_val = [torch.empty((0),device = self.data_manager.device,dtype=int)] * (len(key_guesses))
        sensitivities=[None] * (len(key_guesses))
        accuracies = [None] * (len(key_guesses))
        models= [None] * (len(key_guesses))
        self._start_record()
        for j in range(self.batch_size,self.data_manager.na,self.batch_size):
            self._printd("j is ",j)
            for i,key_guess in enumerate(key_guesses):
                data=self.data_manager.prepare_unprofiled_data(byte,self.dk,key_guess=key_guess,leakage_model = self.leakage_model,split_val=False,min_id=j-self.batch_size,max_id=j)
                x_attack_train,label_attack_train,plaintext_attack_train,key_attack,X,x_attack_val,label_attack_val,plaintext_attack_val=data
                
                x_attack_train.requires_grad_()
                X.requires_grad_()
                #self._printd("labels len:",labels.size()," traces len: ",traces.size())
                labels_train[i]=torch.cat((labels_train[i],label_attack_train))
                labels_val[i] = torch.cat((labels_val[i],label_attack_val))
                models[i],_,_,_,_,_ = self.train_model(x_attack_train,label_attack_train,x_attack_val,label_attack_val,plaintext_attack_val,\
                                    key_guess,byte,model=models[i],get_metrics=False,fast=False)
                sensitivities[i] = list(self.get_sensitivity_tensors(models[i],X,labels_train[i]).detach().cpu().numpy())
            if(j == 0):
                self._round_record(len(key_guesses))
            if(Attack.get_GE_from_sensitivity(sensitivities)==1):
                self._end_record()
                self._print("Minimum number of traces to reach GE = 1 for byte ",byte, ":",j )
                return j
            else:
                self._printd(" got ",Attack.get_GE_from_sensitivity(sensitivities))
        self._end_record()
        self._print("Could only reach GE =", Attack.get_GE_from_sensitivity(sensitivities), "for byte ",byte, "with ",self.data_manager.na," attack traces." )
    
    def train_model_adversarial():
        raise NotImplementedError("")

    def train_model_default(self,x_attack,label_attack,x_attack_val,label_attack_val,plaintext_attack_val,key_guess,byte,model=None,get_metrics=True,fast=False,sensi_reg = False,fast_sensi = False):
        """ Trains the model

        Parameters
        ----------
        x_attack : List
                The attack traces used for unprofiled training. Should have the shape (Na,1,Ns).
        label_attack : List[int]
                The labels corresponding to x_attack.
        x_attack_val : List
                Traces to use for metrics computations (e.g. sensitivity computation). Can be
                the same as x_attack.
        label_attack_val : List[int]
                The labels corresponding to x_attack_val.
        plaintext_attack_val : List[List[int]]
                The plaintexts corresponding to x_attack_val.
        key_guess : int
                Value of the guessed key byte.
        byte : int
                Target byte index.
        model : torch.nn.Module, default : None
                In case of gradual learning (i.e. when you want to re-use a previous model and train it again).
        get_metrics : bool, default : True
                Whether to calculate training metrics such as the sensitivity and accuracy.
        fast : bool, default : False
                Whether to calculate the Fast GE training metric. It will slow down the computation. If set to 
                False, it will return the loss value at each epoch instead of the GE.
        sensi_reg : bool, default : False
                Whether to sue sensitivity as a regularization technique.
        fast_sensi : bool, default : False
                Whether to return, for each epoch, the maximum sensitivity spike for each guess. (Not implemented yet !)

        Returns
        -------
        model : torch.nn.Module
                The trained model.
        sensitivity : numpy.array
                The sum over each epoch of the sensitivity values. If get_metric is False, an empty list is returned.
        accuracies : List
                The accuracies for each epoch. If get_metric is False, an empty list is returned.
        fast_GEs : List
                The fast Guessing entropy for each epoch. If get_metric is False, an empty list is returned.
        train_acc : List
                The training accuracy for each epoch. If get_metric is False, an empty list is returned.
        sensi_slider : List[List]
                A List containing the history of the sensitivity evolution through the epochs. 
        """
        assert not (sensi_reg and not get_metrics)
        ACCURACY_SIZE_LIMIT = None  #Can set this to 1000 for example. Will then reduce the metrics computation time.
        if(model==None):
            model=self._get_model()
        loss_fn,label_attack = self._get_loss_label(label_attack)
        learning_rate = 0.0010000000474974513
        optimizer = self._get_optimizer(model,learning_rate=self.learning_rate)
        # if(self.optimizer=="Adam"):
        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # elif(self.optimizer=="SGD"):
        #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=DEFAULT_SGD_MOMENTUM) #momentum 0.5 0.9 0.99
        accuracies=list()
        train_acc=list()
        fast_GEs=list()
        sensi_slider = list()

        data_set=NRFDataset(x_attack,label_attack,device = self.data_manager.device)
        train_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle = False ) #self.seed is None)
        sensitivity_regularizer = torch.tensor(0.,requires_grad=True)
        #torch.autograd.set_detect_anomaly(True)  #Set this in case you have an Autograd problem somewhere
        if(self.lr_schedule is not None):
            # scheduler = StepLR(optimizer, step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])
            scheduler = OneCycleLR(optimizer=optimizer,epochs=self.epochs,max_lr= 0.002,total_steps=None,steps_per_epoch=ceil(self.data_manager.nt/self.batch_size)\
                 ,pct_start=0.3,anneal_strategy='linear',cycle_momentum=False,base_momentum=0.85,max_momentum=0.95,div_factor=25,final_div_factor=1e4,three_phase=True\
                    ,last_epoch=-1,verbose=self.verbose==2)#,step_size=self.lr_schedule[0], gamma=self.lr_schedule[1])
        for t in range(self.epochs):
            model.train()
            loss_tracker = 0.0

            for i,(x,y) in enumerate(train_loader):
                x,y = DataManager._unison_shuffled_copies(x,y)
                # print("X is ",x)
                y_pred = model(x.detach())
                # print("Pred are ",y_pred)
                # print("y_pred",torch.exp(y_pred))
                if(self.loss=="mse"):
                    y_pred=torch.exp(y_pred)  #compensate for LogSoftMax
                    loss = loss_fn(y_pred,y)
                else:
                    # print("y pred",y_pred.double()," y ",y.double())
                    loss = loss_fn(y_pred,y)
                if((i)%10 == 0):
                    print("loss is",loss," at epoch",t,"in batch",i)
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
                if(sensi_reg):
                    self._printw("Using Sensi Reg !")
                    print("Loss is",loss,"sensi ref is",sensitivity_regularizer)
                    if(self.dim == 0):
                        loss = loss #/  sensitivity_regularizer # (loss / self.batch_size) #/ self.batch_size #/loss
                    else:
                        loss = loss # / sensitivity_regularizer #(loss/self.leakage_model.get_classes())
                loss.backward(retain_graph=False)
                optimizer.step()
            if(self.lr_schedule is not None):
                # In case you use the OneCycleLR, put the scheduler step inside the batch loop.
                scheduler.step()
            model.eval()
            if(t==0):
                if(get_metrics):
                    sensitivity = (self.get_sensitivity_tensors(model,x_attack,label_attack))
                    if(sensi_reg):
                        sensitivity_regularizer =  torch.sum(torch.abs(sensitivity))
                    sensitivity=sensitivity.detach()
                    # print("init sensi ",sensitivity)
                    sensi_slider.append(sensitivity)
                    # topk=torch.topk(torch.abs(sensitivity),2)
                    # sensitivity_regularizer = topk.values[0] 
                else:
                    sensitivity=torch.empty((0))
            elif(get_metrics):
                sensitivity_i = self.get_sensitivity_tensors(model,x_attack,label_attack)
                if(sensi_reg):
                    sensitivity_regularizer =  torch.sum(torch.abs(sensitivity_i))
                # print("sneis from ",sensitivity)
                sensitivity=sensitivity.add(sensitivity_i.detach())
                # print("sensi is now",sensitivity)
                # topk=torch.topk(torch.abs(sensitivity),1)#len(sensitivity_i))
                # sensitivity_regularizer = topk.values[0]
                sensi_slider.append(sensitivity)
            if(get_metrics):
                accuracies.append(self.get_accuracy(self._batch_predict(model,x_attack_val[:ACCURACY_SIZE_LIMIT]),label_attack_val[:ACCURACY_SIZE_LIMIT]))
                train_acc.append(self.get_accuracy(self._batch_predict(model,x_attack[:ACCURACY_SIZE_LIMIT]),label_attack[:ACCURACY_SIZE_LIMIT]))
                # if(fast):
                #     fast_GEs.append(self.get_fast_GE(model,x_attack_val,plaintext_attack_val,byte,key_guess,fast_size=ACCURACY_SIZE_LIMIT))
                # else:    
                #     fast_GEs.append(loss_tracker)

        return model,torch.abs(sensitivity).cpu().numpy(),accuracies,fast_GEs,train_acc,sensi_slider

    def train_model_cross_validation(self):
        raise NotImplementedError("")

    def __str__(self):
        comp = ""
        if(self.split_non_profiling_attack_set):
            comp += ",which splits the set for validation, "
        return "Non-Profiled SCA Attack on AES "+comp+ "with "+super().__str__()

def display_sensi_slider(sensitivities_slider):
    """Function to display an interactive history of the sensitivity w.r.t. the epoch number.

    Parameters
    ----------
    sensitivities_slider : List[List[float]]
            The accumulated sensitivity values for each epoch.
    
    """
    for s in sensitivities_slider:
        for i in range(len(s)):
            s[i] = s[i].cpu().numpy()
    sensitivities_slider = np.array(sensitivities_slider)
    def get_sensi_epoch(epoch):
        return np.abs(sensitivities_slider[:,epoch])

    def draw(axis,sensis):
        colors=["b","g","c","m","y","k","peru","purple","olive","pink","forestgreen","moccasin"]
        for i,s in enumerate(sensis):
            if(i == len(sensis)-1):
                axis.plot(s,'red',linewidth=4,alpha=0.5)
            else:
                axis.plot(s,colors[i % len(colors)],linewidth=1,alpha=0.4)
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25)
    ax = fig.subplots()
    draw(ax,get_sensi_epoch(len(sensitivities_slider[0])-1))
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    # Properties of the slider
    s_factor = Slider(ax_slide, 'Epoch number',
                    1, len(sensitivities_slider[0]), valinit=len(sensitivities_slider[0]), valstep=1)
    # Updating the plot
    def update(val):
        current_v = s_factor.val
        ax.cla()
        draw(ax,get_sensi_epoch(current_v-1))
        fig.canvas.draw()
    # Calling the function "update" when the value of the slider is changed
    s_factor.on_changed(update)
    plt.show(block=True)
