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
# brief     : Data manager for SCA attacks
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
from enum import Enum
import os
import pickle
import subprocess

import h5py
import numpy as np
import torch

from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

from error import Error


class _StateMachine(Enum):
    CREATION = 0
    ONLY_CHECKING = 1
    FUNCTION = 2

class PreProcessing(Enum):
    HORIZONTAL_STANDARDIZATION = 0
    REMOVE_MEAN_SCALE_1_1 = 1
    REMOVE_MEAN_SCALE_0_1 = 2
    SCALE_1_1 = 3
    SCALE_0_1 = 4

class DataManager(object):
    """DataManager : handles anything needed for the data.

    Prepares the attack dataset upon creation and computes the labels on demand.
    You should not change the leakage model on the fly.

    """
    def __init__(self,na="all",ns="all",nt="all",fs=0,file_name_attack='file_name_attack.h5',\
            file_name_profiling =None,\
                 databases_path="/path/to/Databases/",has_countermeasures=False,blind = False,\
                 force_cpu = False,remove_mask = False,pre_processing = PreProcessing.HORIZONTAL_STANDARDIZATION):
        """ Initializes the DataManager. 

        Does also basic attack preprocessing, such as data standardization/centering, data shuffling and,
        if no countermeasures are used, compute the Point of Interest of the data and makes sure the 
        resulting traces contain the PoI.

        Parameters
        ----------
        na : int, str
                How many attack traces to use.
        ns : int, str
                How many samples per trace to use.
        nt : int, str
                How many profiling traces to use.
        fs : int, default : 0
                First sample to use in each trace. Acts like an offset.
        file_name_attack : str
                The file name containing the attack (and optionally Profiling traces if file_name_profiling is not None)
                traces and the metadata in .h5 format.
        file_name_profiling : str, default : None
                If given, will use the profiling traces in this file (and the attack traces from file file_name_attack).
        databases_path : str, default : "/path/to/Databases/"
                The directory of the target file.
        has_countermeasures : bool, default : False
                Whether the underlying data was generated using counteremasures.
        blind : bool, default : False
                When the dataset does not contain the attack key(s), the blind attribute will automatically be set to True.
                In case there is an attack key information available, setting blind to True will discard this information.
        force_cpu : bool, default : False
                If true, won't use the GPU's even if available.
        remove_mask : bool, default : False
                In case of a masked implementation, will remove the mask from the data if given in the dataset .h5 file.
        """
        self.force_cpu = force_cpu
        self.remove_mask = remove_mask
        self.state = _StateMachine.CREATION
        self.file_name_attack=file_name_attack
        self.file_name_profiling = file_name_profiling
        self.databases_path=databases_path
        self.data=None
        self.has_countermeasures=has_countermeasures
        self._poi = None
        self.na,self._ns,self.nt = DataManager._set_numbers(na,ns,nt)
        self.fs=fs
        self.blind = blind  #whether we want to use the attack keys
        self.check_data = None,None
        self.profiling = nt != 0
        self.pre_processing = pre_processing
        if(torch.cuda.is_available() and not self.force_cpu):
            self.device = get_a_free_gpu()
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            
    def set_device(self,device):
        self.device = device
    def _finish_init(self,leakage_model):
        """Loads the data to start."""
        self.state = _StateMachine.FUNCTION
        if(self.file_name_profiling is None):
            self.file_name_profiling = self.file_name_attack
        if(leakage_model == None or self.has_countermeasures):
            self.x_profiling,self.x_attack,self.plaintext_profiling,\
                self.plaintext_attack,self.key_profiling,self.key_attack\
                    ,self.mask_prof,self.mask_att = self._load(na=self.na,ns=self._ns,nt=self.nt,fs=self.fs,leakage_model=leakage_model)
        else:
            #will use the leakage model to make sure the resulting traces include the PoI.
            self.x_profiling,self.x_attack,self.plaintext_profiling,\
                self.plaintext_attack,self.key_profiling,self.key_attack\
                    ,self.mask_prof,self.mask_att = self._secure_load(leakage_model)
        self.x_profiling,self.plaintext_profiling,self.key_profiling,self.mask_prof = DataManager._unison_shuffled_copies(self.x_profiling,self.plaintext_profiling,self.key_profiling,self.mask_prof)
        self.x_attack,self.plaintext_attack,self.key_attack,self.mask_att = DataManager._unison_shuffled_copies(self.x_attack,self.plaintext_attack,self.key_attack,self.mask_att)
        #print("x attack ",self.x_attack[0:10])
        self.x_attack_not_horiz = torch.clone(self.x_attack)
        mn = torch.mean(self.x_attack_not_horiz, axis=1, keepdims=True).repeat((1,self.x_attack_not_horiz.shape[1]))
        self.x_attack_not_horiz = self.x_attack_not_horiz - mn
        self.x_attack_not_horiz = self.x_attack_not_horiz.float()
        self.x_profiling,self.x_attack=self.pre_process_data(self.x_profiling,self.x_attack)
        
        # DataManager._horizontal_standardization(self.x_profiling,self.x_attack)
        # self.x_attack = self._scale_mean(self.x_attack)
        self.na=len(self.x_attack)
        self._ns=len(self.x_attack[0])
        if(len(self.x_profiling)<5):
            self.nt=0  
        else:
            print("previsou exp were wrong ! ")
            # self.x_profiling = self._scale_mean(self.x_profiling)
        self.profiling = self.nt > 0
        if(True or self.profiling):
            self._temp_x_attack = torch.clone(self.x_attack)
            # self._temp_x_attack = self._temp_x_attack.float().reshape([len(self._temp_x_attack),1,len(self._temp_x_attack[0])])
           # self._temp_x_attack = torch.from_numpy(self._temp_x_attack)
            #self._temp_x_attack = torch.transpose(self._temp_x_attack.float(),1,2)
        else:
            self._temp_x_attack=None
    
    def get_ns(self,leakage_model = None):
        """Returns the number of sample per trace.
        
        Parameters
        ----------
        leakage_model : LeakageModel, default : None
                If the DataManager has not been fully initialized, it need a 
                leakage model in order to compute the number of samples. O/W, 
                you can ommit it.
        
        """
        if(self.state != _StateMachine.FUNCTION and leakage_model is None):
            raise Error("Can't get ns value before data creation without a leakage_model.")
        elif(self.state != _StateMachine.FUNCTION):
            self._finish_init(leakage_model)
        return self._ns
    def __str__(self):
        return str(vars(self))
    def __repr__(self) -> str:
        return str(self)
    @property
    def ns(self):
        raise Error("Call self.get_ns() instead.")

    def get_res_name(self,leakage_model=None):
        """Returns a string describing the DataManager attributes.     
        
        Parameters
        ----------
        leakage_model : LeakageModel, default : None
                If the DataManager has not been fully initialized, it need a 
                leakage model in order to compute the res name. O/W, 
                you can ommit it.
        
        """
        if(self.state != _StateMachine.FUNCTION and leakage_model is None):
            raise Error("Can't get res name value before data creation without a leakage_model.")
        elif(self.state != _StateMachine.FUNCTION):
            self._finish_init(leakage_model)
        comp=self.file_name_attack[:20].replace(".h5","")+"_"
        if(self.profiling):
            comp += f",Nt_{self.nt}" 
        return f"_{comp},Na_{self.na},Ns_{self._ns}"

    def get_check_data(self):
        """This function returns a plaintext/ciphertext pair that can further be used to test a key."""
        if(self.state == _StateMachine.CREATION):
            in_file = h5py.File(self.databases_path+self.file_name_attack, "r")
            plaintext_attack = in_file['Attack_traces/metadata']['plaintext'][0]
            try:
                cipher_check = in_file['Attack_traces/metadata']['ciphertext'][0]
                self.check_data = plaintext_attack,cipher_check
            except:
                self.check_data = None,None
            finally:
                in_file.close()
                self.state = _StateMachine.ONLY_CHECKING
        return self.check_data
    
    def set_seed(self,newseed):
        """Setter for the seed."""
        if(self.state == _StateMachine.FUNCTION):
            print("Warning: it's a bit late to set the seed !")
        if(newseed != None):
            np.random.seed(newseed)


    def prepare_unprofiled_data(self,byte,dk,key_guess,leakage_model,split_val = True,split_rate = 0.88,min_id=None,max_id=None):
        '''Prepares the data for each key guess in a non profiled attack.

        Loads, standardizes, transforms to binary-categorical labels. It also reshapes the 
        data to allow usage on the models. The min_id/max_id are used in incremental learning
        to find the minimum number of attack traces needed to reach GE=1 for example.

        Parameters
        ----------
        byte : int 
                Which byte to attack ? (between 0 and 15).
        dk : bool
                Whether to use Domain Knowledge neurons.
        key_guess : int
                The hypotetical value of the target key byte.
        leakage_model : LeakageModel
                The underlying LeakageModel used for label computation.
        split_val : bool, default : False
                Whether to split the attack in distinct training/validation sets. If set to
                False, will use the same set in validation than in tranining.
        split_rate : float, default : 0.8
                In case split_val is True, how much data to use for training only. The reset
                being for validation.
        min_id : int, default : None
                The start index of the traces to be used.
        max_id : int, default : None
                The end index of the traces to be used.
        
        Returns
        -------
        List[List[float]]
                Attack traces
        List[List[int]]
                It is the labels of the attack traces, computed with the key_guess.
        List[List[int]]
                Plaintexts corresponding to the attack traces.
        List[List[int]] 
                List of the attack keys (na times the same key in a list).
        List[List[float]], _
                If max_id is not None, will return all the traces until max_id.
        List[List[int]]
                It is the labels of the validation traces, computed with the key_guess.
        List[List[int]]
                Plaintexts corresponding to the validation traces.
        '''
        if(self.state != _StateMachine.FUNCTION):
            self._finish_init(leakage_model)
        
        label_attack=leakage_model.get_small_profiling_labels(self.plaintext_attack[min_id:max_id], byte, key_guess,profiling=False,mask=self.mask_att[min_id:max_id])
#        label_attack=to_categorical(label_attack,leakage_model.get_classes())
        label_attack = label_attack.astype(np.int64)
        label_attack=torch.from_numpy(label_attack).to(self.device)
        if(dk):
            x_attack_accumulated = self._add_domain(self.x_attack[:max_id],self.plaintext_attack[:max_id],byte)
            x_attack_accumulated = x_attack_accumulated.float().reshape([len(x_attack_accumulated),1,len(x_attack_accumulated[0])])
           # x_attack_accumulated,label_attack=self._to_tensor_device([x_attack_accumulated,label_attack])
            #x_attack_accumulated = torch.transpose(x_attack_accumulated.float(),1,2)
            x_attack_res = x_attack_accumulated[min_id:max_id]
        else:
            x_attack_res = self._temp_x_attack[min_id:max_id]
            x_attack_accumulated=self._temp_x_attack[:max_id]
        if(not split_val):
            x_val = x_attack_res
            label_val = label_attack
            plaintext_val = self.plaintext_attack
            x_attack_res_res = x_attack_res
            label_attack_res = label_attack
            plaintext_attack_res = self.plaintext_attack
        else:
            t_size= int(x_attack_res.size()[0] * split_rate)
            x_val,x_attack_res_res=x_attack_res[t_size:],x_attack_res[:t_size]
            label_val,label_attack_res=label_attack[t_size:],label_attack[:t_size]
            plaintext_val,plaintext_attack_res=self.plaintext_attack[t_size:],self.plaintext_attack[:t_size]
            
        if(max_id != None):
            return x_attack_res_res,label_attack_res,plaintext_attack_res,self.key_attack,x_attack_accumulated,x_val,label_val,plaintext_val
        else:
            return x_attack_res_res.float(),label_attack_res,plaintext_attack_res,self.key_attack,x_val.float(),label_val,plaintext_val
    def prepare_profiled_data(self,byte,leakage_model,use_attack_as_val=True,split_rate=0.95,dk=False):
        '''Prepares the data for profiled attacks.
            
        Parameters
        ----------
        byte : int
                The target byte index.
        leakage_model : LeakageModel
                The underlying LeakageModel used for label computation.
        use_attack_as_val : bool
                Whether to use the attack traces as validation set. If set to False, will split the profiling traces
                in training/validation. If set to None, will use the attack traces as validation only if the 
                attack is not blind.
        split_rate : float, default : 0.95
                In case use_attack_as_val is set to False, will split the profiling traces in training/validation
                traces according to split_rate ratio.
        dk : bool
                Whether to use Domain Knowledge neurons.

        Returns
        -------
        List[List[float]]
                Traces to use for training.
        List[List[int]]
                Labels of the training traces.
        List[List[float]]
                Traces used for validation. May be the same as the attack traces.
        List[List[int]]
                Labels corresponding to the validation traces.
        List[List[int]]
                Plaintexts corresponding to the validation traces.
        List[List[float]]
                Attack traces
        List[List[int]]
                It is the labels of the attack traces, computed with the key_guess.
        List[List[int]]
                Plaintexts corresponding to the attack traces.
        List[List[int]] 
                List of the attack keys (na times the same key in a list).       
        '''
        #x_profiling = scale_mean(x_profiling)
        #x_attack = scale_mean(x_attack)
        if(self.state != _StateMachine.FUNCTION):
            self._finish_init(leakage_model)
        if(self.nt == 0):
            raise Error("No profiling traces present for the profiled attack. Please consider using an UnProfiled attack instead.")
        if(use_attack_as_val is None):
            use_attack_as_val = not self.blind
        label_profiling=leakage_model.get_small_profiling_labels(self.plaintext_profiling, byte, self.key_profiling,True,self.mask_prof)
        label_profiling = label_profiling.astype(np.int64)
#        label_profiling=to_categorical(label_profiling,leakage_model.get_classes())
        label_attack=leakage_model.get_small_profiling_labels(self.plaintext_attack, byte, self.key_attack,True,self.mask_att)
#        label_attack=to_categorical(label_attack,leakage_model.get_classes())
        label_attack = label_attack.astype(np.int64)
        label_profiling = torch.from_numpy(label_profiling).to(self.device)
        label_attack = torch.from_numpy(label_attack).to(self.device)
        if(dk):
            x_profiling_res=self._add_domain(self.x_profiling,self.plaintext_profiling,byte)
            x_attack_res=self._add_domain(self.x_attack,self.plaintext_attack,byte)
        else:
            x_profiling_res = (self.x_profiling)  #.to(self.device) #was torch.clone
            x_attack_res = (self.x_attack)  #.to(self.device)  #was torch.clone
        x_profiling_res=x_profiling_res.float().reshape([len(x_profiling_res),1,len(x_profiling_res[0])])
        x_attack_res=x_attack_res.float().reshape([len(x_attack_res),1,len(x_attack_res[0])])
        if(use_attack_as_val):
            x_val,label_val,plaintext_val,mask_val=x_attack_res,label_attack,self.plaintext_attack,self.mask_att
        else:
            x_profiling_res,label_profiling,x_val,label_val,plaintext_profiling_res,plaintext_val,mask_train,mask_val = \
                    self._split_train_validation(x_profiling_res,label_profiling,self.plaintext_profiling,self.mask_prof,split_rate=split_rate)
            #we don't care about the plaintext_profiling_res anymore as we have the labels.
        return x_profiling_res,label_profiling,x_val,label_val,plaintext_val,mask_val,x_attack_res,label_attack,self.plaintext_attack,self.key_attack,self.mask_att
    
    @staticmethod
    def tensor_to_numpy(array):
        """Turns an array of tensors to numpy arrays."""
        return [t.cpu().numpy() for t in array]
    
    def _load(self,na,ns,nt,fs=0,leakage_model = None):
        '''Loads the dataset from a .h5 file.

        Parameters
        ----------
        na : int
                Number of attack traces (with the same key).
        ns : int
                Number of sample (in each trace).
        nt : int
                Number of profiling traces.
        fs : int, default : 0
                First sample (in each trace, where is the first useful sample).
        leakage_model : LeakageModel
                The underlying LeakageModel used for label computation.
                
        Returns
        -------
        List[List[float]],List[List[float]],List[List[int]],List[List[int]],List[List[int]],List[List[int]]
                Returns x_profiling,x_attack,plaintext_profiling,plaintext_attack,key_profiling,key_attack,
                where x_ stands for trace (the power trace, or wave).
        '''
        name_low = self.file_name_attack.lower()
        fresh_profiling_data = "variable" in name_low or not ("ascad" in name_low or "aes_hd" in name_low \
            or "aes_rd" in name_low or "dpa_contest" in name_low)  # fresh profililng data means a different key for each profiling trace
        if(leakage_model is not None and leakage_model.needs_cipher_and_not_ptx()):
            plain_info = "ciphertext"
        else:
            plain_info = "plaintext"
        in_file_att = h5py.File(self.databases_path+self.file_name_attack, "r")
        in_file_prof = h5py.File(self.databases_path+self.file_name_profiling, "r")
        if(ns==None):
            last_sample = None
        else:
            last_sample = fs + ns
        if(not fresh_profiling_data):   #to use data classified in the profiling or attack set in the other set
            nt_load = None
            na_load = None #there is a bug here
        else:
            nt_load = nt
            na_load = na
        x_profiling = np.array(in_file_prof['Profiling_traces/traces'], dtype=np.float64)[:nt_load, fs:last_sample]
        x_attack = np.array(in_file_att['Attack_traces/traces'], dtype=np.float64)[:na_load, fs:last_sample]
        # print("X attack is ",x_attack,"x prof",x_profiling)
        # assert False
        plaintext_profiling = np.array(in_file_prof['Profiling_traces/metadata'][plain_info][:nt_load],dtype=np.int64)
        plaintext_attack = np.array(in_file_att['Attack_traces/metadata'][plain_info][:na_load],dtype=np.int64)
        if(self.remove_mask):
            try:
                mask_prof = np.array(in_file_prof['Profiling_traces/metadata']['masks'][:nt_load],dtype=np.int64)
            except:
                mask_prof = np.zeros_like(plaintext_profiling)
            try:
                mask_att = np.array(in_file_att['Attack_traces/metadata']['masks'][:na_load],dtype=np.int64)
            except:
                mask_att = np.zeros_like(plaintext_attack)
        else:
            mask_att = np.zeros_like(plaintext_attack)
            mask_prof = np.zeros_like(plaintext_profiling)

        try:
            key_profiling = np.array(in_file_prof['Profiling_traces/metadata']['key'][:nt_load],dtype=np.int64)
        except:
            key_profiling = np.empty_like(plaintext_profiling)
            #key_profiling = np.zeros(shape = (x_profiling.shape[0],16),dtype=np.int8)
        try:
            if(plain_info == "plaintext"):
                cipher_check = np.array(in_file_att['Attack_traces/metadata']['ciphertext'][0],dtype=np.int64)
                self.check_data = plaintext_attack[0],cipher_check
            else:
                plain = np.array(in_file_att['Attack_traces/metadata']['plaintext'][0],dtype=np.int64)
                self.check_data = plain,plaintext_attack[0]
        except:
            self.check_data = None,None
        if(self.blind):
            key_attack = np.zeros(shape = (x_attack.shape[0],16),dtype=np.int8)
            self.blind = True
        else:
            try:
                key_attack = np.array(in_file_att['Attack_traces/metadata']['key'][:na_load],dtype=np.int64)
                self.blind = False
            except:
                print("No attack key available. Launching a blind attack.")
                key_attack = np.zeros(shape = (x_attack.shape[0],16),dtype=np.int8)
                self.blind = True
                
        in_file_att.close()
        in_file_prof.close()
        if(not fresh_profiling_data):
            
            if(na == None):
            # if(last_na == None):
                print("Implicit NA choosing for un-fresh dataset is not recommended. Using ",len(x_profiling)//2," attack traces.")
                last_na = len(x_profiling)//2
            elif(na + nt > len(x_attack) + len(x_profiling)):
                raise Error(f"The data file has not enough traces. It has only {len(x_profiling) + len(x_attack)} of them.")
            elif(len(x_attack) < na):
                last_na = na - len(x_attack)
            else:
                last_na = 0

            if(nt == None):
            # if(last_na == None):
                print("Implicit Nt choosing for un-fresh dataset is not recommended. Using ",len(x_profiling)//2," profiling traces.")
                last_nt = len(x_profiling)//2
            elif(nt != 0 and na + nt > len(x_attack) + len(x_profiling)):
                raise Error(f"The data file has not enough traces. It has only {len(x_profiling) + len(x_attack)} of them.")
            elif(len(x_profiling) < nt):
                last_nt = nt - len(x_profiling)
            else:
                last_nt = 0
            if(last_nt == 0):
                if(nt > 0 and "ASCAD" in self.file_name_attack):
                    print("Discarding attack set ! As [104] Timon Benjamin.")
                    x_attack =  x_profiling[:na]
                    plaintext_attack = plaintext_profiling[:na]
                    key_attack = key_profiling[:na]
                    mask_att = mask_prof[:na]
                else:
                    x_attack = np.append(x_attack, x_profiling[:last_na],axis = 0)
                    plaintext_attack = np.append(plaintext_attack,plaintext_profiling[:last_na],0)
                    key_attack = np.append(key_attack,key_profiling[:last_na],0)
                    mask_att = np.append(mask_att,mask_prof[:last_na],0)
                if(nt==None):
                    last_nt = None
                else:
                    last_nt = last_na + nt
                x_profiling = x_profiling[last_na:last_nt]
                plaintext_profiling = plaintext_profiling[last_na:last_nt]
                key_profiling = key_profiling[last_na:last_nt]
                # print("key_profiling",key_profiling[0])
                mask_prof = mask_prof[last_na:last_nt]
            elif(last_na == 0):
                x_profiling = np.append(x_profiling, x_attack[:last_nt],axis = 0)
                plaintext_profiling = np.append(plaintext_profiling,plaintext_attack[:last_nt],0)
                key_profiling = np.append(key_profiling,key_attack[:last_nt],0)
                mask_prof = np.append(mask_prof,mask_att[:last_nt],0)
                if(na==None):
                    last_na = None
                else:
                    last_na = last_nt + na
                x_attack = x_attack[last_nt:last_na]
                plaintext_attack = plaintext_attack[last_nt:last_na]
                key_attack = key_attack[last_nt:last_na]
                # print("key_profiling",key_profiling[0])
                mask_att = mask_att[last_nt:last_na]
            # print("ns ",len(x_attack[0]))
        if(leakage_model is not None):
            key_profiling = leakage_model.get_used_keys(key_profiling)
            key_attack = leakage_model.get_used_keys(key_attack)
        if(na != None and len(x_attack) < na):
            raise Error(f"The data file has not enough attack traces. It has only {len(x_attack)} of them.")
        if(nt != None and len(x_profiling)<nt):
            raise Error(f"The data file has not enough profiling traces. It has only {len(x_profiling)} of them.")
        if(len(x_attack)==0):
            raise Error("The data file contains no attack traces. Please give a data with attack traces.")
        if(ns != None and len(x_attack[0])!=ns):
            raise Error(f"The data file has not enough samples per trace. It has only {len(x_attack[0])} of them instead of {ns}.")
        if(len(x_profiling)>0 and len(x_profiling[0]) != len(x_attack[0])):
            raise Error(f"The profiling and attack traces does not have equal sample size ({len(x_profiling[0])} vs {len(x_attack[0])}). Please specify a ns.")
        output_data = x_profiling,x_attack,plaintext_profiling,plaintext_attack,key_profiling,key_attack,mask_prof,mask_att
        x_profiling,x_attack,plaintext_profiling,plaintext_attack,key_profiling,key_attack,mask_prof,mask_att = self._to_tensor(output_data)

        return x_profiling.to(self.device),x_attack.to(self.device),plaintext_profiling,plaintext_attack,key_profiling,key_attack,mask_prof,mask_att

    def _secure_load(self,leakage_model):
        """ Secure load: makes sure the samples include the Point of Interest (PoI).

        It's main goal is to compute a proper fs (first sample) value. It should be 
        just before a PoI. If a non null fs value has been set by the user at the creation of 
        the DataManager, it will use this value instead.

        Raises
        -------
        Error
                When the file named file_name is not existent.

        Returns
        -------
        List[**]
                Output of the load() function with computed arguments.
        """
        if(self._ns == None or self.has_countermeasures or self.fs != 0):
            return self._load(na=self.na,ns=self._ns,nt=self.nt,fs=self.fs,leakage_model=leakage_model)
        previous_poi = self._recover_poi(leakage_model)
        if(previous_poi == -1):
            x_profiling,x_attack,plaintext_profiling,plaintext_attack,\
                key_profiling,key_attack,mask_prof,mask_att = self._load(*DataManager._set_numbers("all","all","all"),fs=0,leakage_model=leakage_model)
            if(not self.blind):
                waves=np.concatenate((x_profiling.detach().cpu(),x_attack.detach().cpu()))
                plaintexts=np.concatenate((plaintext_profiling.detach().cpu(),plaintext_attack.detach().cpu()))
                keys=np.concatenate((key_profiling.detach().cpu(),key_attack.detach().cpu()))
            elif(self.nt == 0):
                return self._load(na=self.na,ns=self._ns,nt=self.nt,fs=self.fs,leakage_model=leakage_model)
            else:
                waves = x_profiling.cpu()
                plaintexts = plaintext_profiling.cpu()
                keys = key_profiling.cpu()
            poi=leakage_model.get_snr_poi(waves,plaintexts,keys)
            self._store_poi(poi,leakage_model)
            if(self._ns < poi):
                fs = poi - (self._ns) // 2
            else:
                fs = 0
            print("Automatic PoI surrounding, setting fs to",fs)
            x_profiling = x_profiling[:self.nt,fs:fs+self._ns]
            x_attack = x_attack[:self.na,fs:fs+self._ns]
            return x_profiling,x_attack,plaintext_profiling[:self.nt],plaintext_attack[:self.na],\
                        key_profiling[:self.nt],key_attack[:self.na],mask_prof[:self.nt],mask_att[:self.na]
        else:
            if(self._ns < previous_poi):
                fs = previous_poi - (self._ns) // 2
            else:
                fs = 0
            print("Automatic PoI surrounding, setting fs to",fs)
            return self._load(self.na,self._ns,self.nt,fs=fs,leakage_model=leakage_model)
    def _store_poi(self,poi,leakage_model):
        """Function to store the calculated Point of Interest (PoI) in the file .PoIs.pickle."""
        print("The average PoI of this dataset is located at ",poi)
        self._poi=poi
        poi_path=os.path.dirname(__file__) + "/.PoIs.pickle" 
        if(os.path.exists(poi_path)):
            PoIs=pickle.load(open(poi_path,"rb"))
            if(self.file_name_profiling in PoIs):
                if(leakage_model.leakage_location in PoIs[self.file_name_profiling]):
                    print("Warning, overwritting the poi. Should not happen.")
                else:
                    pass
            else:
                PoIs[self.file_name_profiling] = {}
        else:
            PoIs={}
            PoIs[self.file_name_profiling] = {}
        PoIs[self.file_name_profiling][leakage_model.leakage_location] = poi
        pickle.dump(PoIs, open(poi_path,"wb"))
    def _recover_poi(self,leakage_model):
        """Function to recover the calculated Point of Interest (PoI) from the file .PoIs.pickle."""
        if(self._poi!=None):
            return self._poi
        poi_path=os.path.dirname(__file__) + "/.PoIs.pickle"
        if(os.path.exists(poi_path)):
            PoIs=pickle.load(open(poi_path,"rb"))
            if(self.file_name_profiling in PoIs):
                if(leakage_model.leakage_location in PoIs[self.file_name_profiling]):
                    self._poi = PoIs[self.file_name_profiling][leakage_model.leakage_location]
                    return PoIs[self.file_name_profiling][leakage_model.leakage_location]
        return -1
    def _scale_mean(self,X,remove_mean=True,scale_from_1_1 = True):
        """ Preprocesses the data by removing the mean and scaling in [-1,1].
        
        Parameters
        ----------
        X : List[List[float]]
                The data to be processed. 
        
        Returns
        -------
        List[List[float]]   
                The processed data, centered and scaled between -1 and 1.
        """
        if(remove_mean):
            mn = torch.mean(X, axis=1, keepdims=True).repeat((1,X.shape[1]))#, axis=1)
            # std = torch.std(X, axis=1, keepdims=True).repeat( (1,X.shape[1]))#, axis=1)
        # X_attack_processed = (X_attack - mn)/std
            X_meaned = (X - mn)# / std
        else:
            X_meaned = X
        
        if(scale_from_1_1):
            max_abs_scaler = preprocessing.MaxAbsScaler()
        else:
            max_abs_scaler = preprocessing.MinMaxScaler()
        X_train_maxabs = max_abs_scaler.fit_transform(X_meaned.detach().cpu()) 
        return torch.from_numpy(X_train_maxabs).to(self.device)
    
    def _add_domain(self,traces,ptx,byte):
        """Function to add ptx_i as a one-hot vector to the traces.

        This functions is used when Domain Knowledge neurons are used.

        Parameters
        ----------
        traces : List[List[float]]
                Power traces.
        ptx : List[List[int]]
                Plaintexts corresponding to the traces.
        byte : int
                Target byte index.
        
        Returns
        -------
        List[List[float]]
                A list where each element is a list of the traces plus
                a chunc of the plaintext in a one-hot manner.
        """
        traces_plain=torch.empty((traces.shape[0],traces.shape[1]+256)).to(self.device)
        for i in range(len(traces)):
            traces_plain[i]=torch.cat((traces[i],torch.tensor(to_categorical(ptx[i][byte].cpu(),256)).to(self.device)))
        return traces_plain
    def pre_process_data(self,x_prof,x_attack):
        def process_help(x):
            if(self.pre_processing==PreProcessing.REMOVE_MEAN_SCALE_1_1):
                return self._scale_mean(x,remove_mean=True,scale_from_1_1=True)
            elif(self.pre_processing==PreProcessing.SCALE_1_1):
                return self._scale_mean(x,remove_mean=False,scale_from_1_1 = True)
            elif(self.pre_processing==PreProcessing.REMOVE_MEAN_SCALE_0_1):
                return self._scale_mean(x,remove_mean=True,scale_from_1_1 = False)
            elif(self.pre_processing==PreProcessing.SCALE_0_1):
                return self._scale_mean(x,remove_mean=False,scale_from_1_1 = False)
        if(self.pre_processing==PreProcessing.HORIZONTAL_STANDARDIZATION):
            return DataManager._horizontal_standardization(x_prof,x_attack)
        else:
            x_attack=process_help(x_attack)
            if(len(x_prof)>0):
                x_prof=process_help(x_prof)
            return x_prof,x_attack
    @staticmethod
    def _split_train_validation(X, y,z,m=None ,split_rate=0.9) :
        '''Splits this data in to two distinct groups for training and validation.

        Parameters
        ----------
        X : List[List[float]]
                Training data.
        y : List[List[int]]
                Training labels.
        z : List[List[int]]
                Training plaintexts.
        split_rate : float
                % of X's (y's) data that will be assigned to validation
        
        Returns
        -------
        (List[float]],List[List[int]],List[List[float]],List[List[int])
                The training set (traces & labels) followed by the validation set (traces & labels).
        '''
        break_point = int(y.shape[0] * split_rate)
        X_train = X[0:break_point,:]
        X_val = X[break_point:,:]
        y_train = y[0:break_point]
        y_val = y[break_point:]
        z_train = z[0:break_point]
        z_val = z[break_point:]
        if(m is not None):
            m_train = m[0:break_point]
            m_val = m[break_point:]
        else:
            m_train = None
            m_val = None
        return X_train, y_train, X_val, y_val,z_train,z_val,m_train,m_val
    
    def _to_tensor(self,data):
        """ Transforms data components into tensors.
        
        Parameters
        ----------
        data : (numpy.array,numpy.array)
                A tuple of numpy arrays.

        Returns
        -------
        (torch.tensor,torch.tensor)
                The two numpy arrays transormed into torch tensors.
        """
        return [(torch.from_numpy(x)) for x in data]
    @staticmethod
    def _horizontal_standardization(X_profiling, X_attack):
        """Perform a horizontal stadardization on the two parameters.
        
        Parameters
        ----------
        X_profiling : List[List[float]]
                First list to standardize horizontally.
        X_attack : List[List[float]]
                Second list to standardize horizontally.

        Returns
        -------
        List[List[float]],List[List[float]]
                The input lists standardized and centered.
        """

        mn = torch.mean(X_profiling, axis=1, keepdims=True).repeat( (1,X_profiling.shape[1]))#, axis=1)
        std = torch.std(X_profiling, axis=1, keepdims=True).repeat( (1,X_profiling.shape[1]))#, axis=1)
        X_profiling_processed = (X_profiling - mn)/std

        mn = torch.mean(X_attack, axis=1, keepdims=True).repeat((1,X_attack.shape[1]))#, axis=1)
        std = torch.std(X_attack, axis=1, keepdims=True).repeat( (1,X_attack.shape[1]))#, axis=1)
        X_attack_processed = (X_attack - mn)/std

        return X_profiling_processed, X_attack_processed
    @staticmethod
    def _unison_shuffled_copies(a, b,c=None,d = None):
        """Shuffles a,b and c with the same index reassignment.
        
        Parameters
        ----------
        a : List[object]
        b : List[object]
        c : List[object], None
        seed : int, default : None
                The seed to use for randomization of the shuffling.
        
        Raises
        -------
        Error
                When the input lists don't have the same sizes.
        
        Returns
        -------
        List[object],List[object], (List[object],None)
                Each input list shuffled the same way.
        """ 
        if(len(a)!= len(b) or ((c is not None) and  len(b) != len(c)) or ((d is not None) and  len(b) != len(d))):
            raise Error("The lists should have the same size to perform unison shuffling.")
        p = np.random.permutation(len(a))
        if(c is not None):
            if(d is not None):
                return a[p],b[p],c[p],d[p]
            else:
                return a[p],b[p],c[p]
        else:
                return a[p],b[p]
    @staticmethod
    def _set_numbers(na,ns,nt):
        """Translates the na,ns,nt numbers to values usable in slices.

        Parameters
        ----------
        na : int, str
                Amount of requested attack traces. Can be set to "all" or "min".
        ns : int, str
                Amount of samples to consider in each trace. Can be set to "all".
        nt : int, str
                Amount of profiling traces to use. Can be set to "all".
                
        Returns
        -------
        na,ns,nt : int, None
                The numbers (na,ns,nt) of traces, samples that will effectively be used for the attack.
                Sets some numbers to None since slicing works as follows: tab[2:None] = tab[2:].
                
        """
        if(na in ["min",0] and nt!=0):
            print("Warning, discarding na = min.")
        if(na in ['all','min',0,"max",None]):
            na = None  
        if(nt in ['all','max',None]):
            nt = None  
        elif(nt<6): #account for .h5 initialization which needs to initialize 1-2 traces in the profiling set
            nt=0
        if(ns in ['all','max',None]):
            ns = None  
        return na,ns,nt

def get_a_free_gpu():
    """Returns the available GPU torch device with the most free memory. 

    Returns
    -------
    torch.device
            A torch device able to carry the data.
    """
    ### Caution: needs the right environment variable to be set. (CUDA_DEVICE_ORDER=PCI_BUS_ID)
    # This is done in the __init__.py.

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    device_name = 'cuda:'+str(list(gpu_memory_map.keys())[np.argmax(np.array(list(gpu_memory_map.values())))])
    return torch.device(device_name)
