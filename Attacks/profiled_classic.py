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
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neighbors, svm
from Data.data_manager import DataManager
from Attacks.attack import  Attack, TrainingMethods
from Ciphers.AES_leakage import AESLeakageModel


class ProfiledClassic(Attack):
    def __init__(self,epochs=7,model_name="knn", leakage_model=AESLeakageModel("HW","SBox"),\
                    verbose=1,dk=False,noise=False,seed=None,\
                     data_manager=DataManager(nt=47000,na=2000,ns=100,fs=250,force_cpu=True),results_path="/path/to/PlotsResults/",\
                            info="",n=5):
        """ Initializes an Attack.

        Parameters
        ----------
        epochs : int, default : 15
                how many epochs to train the NN.
        model_name : str, default : knn
                Which model to use. SVM or knn.
        leakage_model : Ciphers.LeakageModel, default : ID(SBox)
                A LeakageModel class used for label computations.
        verbose : int, default : 1
                What should be printed out during execution ? 0 : only the result,
                1 : the different metrics (e.g. remaining time, sensitivities, accuracies),
                2 : output debug informations too.
        dk : bool, default : False
                Whether to use Domain Knowledge neurons. Those add plaintext information
                in the MLP part of the network.
        noise : bool, default : False
                Whether to add gaussian noise to the input of the training data.
        seed : int, default : None
                Which seed to use. If set to None, will use a default standard value, 5437.
        data_manager : Data.DataManager, default : DataManager()
                The datamanager handles anything related to the data. Only pass a DataManager
                with force_cpu set to True.
        results_path : str, default : "/path/to/PlotsResults/"
                Where to store the learning plots.
        info : str, default : ""
                A small text to insert in the result file name.
        n : int, default : 5
                How many nearest neighbors to consider in a knn attack.
        """
        super().__init__(model_name=model_name, batch_size=100,loss="nlll",optimizer="Adam",\
                            leakage_model=leakage_model,verbose=verbose,lambdas=None,dk=dk,noise=noise,\
                                seed=seed,data_manager=data_manager,results_path=results_path,training=TrainingMethods.DEFAULT,info=info,dim=None,\
                                    threshold=0.9)
        self.epochs=epochs
        self.n = n
    def get_pruning_percentage(self):
        return self.LTH
    def attack_byte(self,byte,get_output_probas=False):
        """Attack byte: launches a full attack on a byte. 

        Will print the result of the attack on the terminal.

        Parameters
        ----------
        byte : int
                byte number to attack. Some bytes are harder to attack.
        """
        self._printd("In attack byte profiled")

  #      fig=plt.figure(figsize=(15,15))
        
   #     plot_class = plt.subplot2grid((8, 2), (4, 0), rowspan=2, colspan=2,fig=fig)
        #plot_class.set_title("Training Accuracy")
  #      plot_GE=plt.subplot2grid((8,2),(6,0),rowspan=2,colspan=2,fig=fig)
        #plot_GE.set_title("Fast Guessing Entropy")

        x_profiling,label_profiling,x_validation,label_validation,plaintext_val,mask_val,x_attack,\
            label_attack,plaintext_attack,key_attack,mask_attack = DataManager.tensor_to_numpy(self.data_manager.prepare_profiled_data(byte,leakage_model = self.leakage_model ,\
                                            use_attack_as_val=None,split_rate=0.95,dk=self.dk))
        print("x prof is ",x_profiling.shape)
        x_profiling=x_profiling.reshape([x_profiling.shape[0],x_profiling.shape[2]])
        x_attack=x_attack.reshape([x_attack.shape[0],x_attack.shape[2]])
        print("new shape is ",x_attack.shape)
       # x_profiling,label_profiling,x_validation,label_validation,plaintext_val,x_attack,\
        #    label_attack = DataManager.profiled_data_to_tensor(x_profiling,label_profiling,x_validation,label_validation,plaintext_val,x_attack,\
         #   label_attack)
        model=self.train_model_default(x_profiling,label_profiling,x_validation,label_validation,plaintext_val,key_attack,byte,None,True,True)

        #plot_sensi.plot(sensitivit,linewidth=1,alpha=0.4)#,label="Max Random "+str(sum_sensi)+sum_abs_sensi)
       # plot_class.plot(train_acc,'g',linewidth=1,alpha=0.4)
        #plot_GE.plot(fast_GEs,'g',linewidth=1,alpha=0.4)
        output_probas = self._batch_predict(model,x_attack,detach=False,numpy=True)#.detach().cpu()

           ###################### Test 09.05.2022 for normalized class assignement density ##########################33

        if(False):
            ass = output_probas# model_(x_attack_train)
            #res = torch.sum(torch.argmax(ass,dim=1).detach())
            res3= np.array((\
                np.argmax(ass,axis=1)\
                    ).flatten())
            label_attack_=np.array(np.argmax(label_attack,axis=1))
            #print("res 3 is  ",res3," labs are ",label_attack_)
            counters =np.zeros((self.leakage_model.get_classes(),self.leakage_model.get_classes()))
            for i in range(len(counters)):
                where = np.argwhere(label_attack_ == i)

                classPreds = list(np.where(label_attack_ == i)[0])
                #print(where)
                for j in range(len(counters[i])):
                    #print(res3[where],list(res3[where]))
                    preds = list(res3[where]).count(j)
                    #print(classPreds.count(j))
                    counters[i][j] = preds #classPreds.count(j)
            #print("counter is ",counters)
            counters = np.array(counters)
            mn = np.repeat(np.mean(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
            std = np.repeat(np.std(counters, axis=1, keepdims=True), counters.shape[1], axis=1)
            counters = (counters - mn)/ std


            plot_class.imshow(counters)

                ###################### End ####################################################################33


        # print("plaintext_attack",plaintext_attack)
        output_probas_on_keys=self.turn_output_probas_to_key_proba(output_probas,plaintext_attack,byte)
        if(not self.data_manager.blind):
            ge = Attack.get_GE(output_probas_on_keys,key_attack[0][byte])
            print("Guessing entropy =  ",ge," for byte",byte)
            if(ge==1):
                min_ge1=ProfiledClassic.get_min_trace_for_GE1(output_probas_on_keys,key_attack[0][byte])
                print("Minimum attack traces to reach GE = 1: ",min_ge1," for byte",byte)
            else:
                min_ge1=0
            if(not get_output_probas):
                return ge,min_ge1
            else:
                return ge,min_ge1,output_probas_on_keys
        else:
            res_key_probas = Attack.get_keys_from_probas(output_proba_on_key=output_probas_on_keys)
            if(self.verbose!=0):
                plt.savefig(self._get_res_name(byte,ge="blind"))
            print(f"Byte {byte} attack result: ",res_key_probas)
            return res_key_probas

    def _get_res_name(self,byte,ge):
        return super()._get_res_name(f"profiled,byte{byte},GE_{ge}")
    def _get_super_name(self,info=""):
        return super()._get_res_name(info)
    def attack_bytes(self,byte_range):
        return super().attack_bytes(byte_range)
    def train_model_adversarial(self):
        raise NotImplementedError("")
        

    def train_model_default(self,x_profiling,label_profiling,x_validation,label_validation,plaintext_val,key_attack,byte,model=None,get_metrics=False,fast=True,sensi_reg = False):
        """ Trains the model.

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
        # print("label_profiling",label_profiling)
        # label_profiling=np.argmax(label_profiling,axis=-1)
        if(self.model_name == "svm"):
            #print(x_profiling.shape)
            lin_clf = svm.LinearSVC()
            lin_clf.fit(x_profiling, label_profiling)
            return lin_clf.decision_function
        elif(self.model_name == "knn"):
            clf = neighbors.KNeighborsClassifier(n_neighbors=self.n, weights="uniform")
            # print("label_profiling",label_profiling)
            # print("x_profiling",x_profiling)
            # print("Shape of label_profiling:",label_profiling.shape)
            # print("Shape of x_profiling:",x_profiling.shape)
            clf.fit(x_profiling,label_profiling)
            return clf.predict_proba
        pass
            #dec = lin_clf.decision_function([[1]])
            #dec.shape[1]

    @staticmethod
    def get_min_trace_for_GE1(output_proba_on_key,key_byte,step=1):
        '''Computes the minimum number of traces required to get GE = 1

        Iteratively considers more and more attack traces until it is possible
        to reach a Guessing Entropy of 1.

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
        '''
        k=1
        while(np.argmax(np.sum(output_proba_on_key[:k],axis=0))!=key_byte and k<=len(output_proba_on_key)):
            k=k+step
        if(np.argmax(np.sum(output_proba_on_key[:k],axis=0))!=key_byte):
            return -k
        else:
            return k


    def __str__(self):
        return "ProfiledClassic SCA Attack on AES with"+ super().__str__()
    def _draw_graph_validation_epoch(self,history,info,save=True):
        '''Draws two plots, the loss/accuracy against the epochs
        '''
        
        
        history_dict = history.history

        self._printd("The history contains ",history_dict)
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss'+ "binary_crossentropy")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        if(save):
            plt.savefig(self.results_path+info + '.pdf')
        else:
            plt.show()
