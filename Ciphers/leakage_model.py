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
# brief     : Abstract class for cipher definitions
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
from abc import ABC, abstractmethod

import numpy as np
import torch

from error import Error


class LeakageModel(ABC):
    """ Base class for leakage modelling. Each instantiation should be tailored to a specific cipher."""

    def __init__(self,leakage_model):
        """Instantiates the class.

        Parameters
        ----------
        leakage_model : str
                which leakage labelling to use. Can be ID,HW,LSBS,LSB,MSB, HW2 or HW3.

        Raises
        -------
        Error
                when an unimplemented leakage labelling is given.
        """
        self.leakage_model=leakage_model
        if(self.leakage_model=="ID"):
            self.num_classes=256
            self.class_arrangement=np.array(range(0,256))
        elif(self.leakage_model in ["HW","HD"]):
            self.num_classes=9
            self.class_arrangement=np.array([bin(n).count("1") for n in range(0,256)])
        elif(self.leakage_model == "HW2"):
            self.num_classes = 2
            def get_class(n):
                if(bin(n).count("1") < 4):
                    return 0
                elif(bin(n).count("1") > 4):
                    return 1
                else:
                    return 1
            self.class_arrangement=np.array([get_class(n)  for n in range(0,256)])
        elif(self.leakage_model == "LSBS"):
            self.num_classes=4
            self.class_arrangement=np.array([n & 0b11 for n in range(0,256)])
        elif(self.leakage_model=="LSB"): 
            self.num_classes=2    
            self.class_arrangement=np.array([n & 0b1 for n in range(0,256)])
        elif(self.leakage_model=="MSB"): 
            self.num_classes=2    
            self.class_arrangement=np.array([(n & 0b10000000)>>7 for n in range(0,256)])
        else:
            raise Error("Leakage model ",self.leakage_model," not implemented.")
    @abstractmethod
    def needs_cipher_and_not_ptx(self):
        """To know whether the LeakageModel needs the ciphertext instead of the plaintext."""
        pass

    @abstractmethod
    def get_used_keys(self,keys):
        """Get the key used in the labelling.

        Usually, a cipher may produce round keys. Depending on the targetted
        sub-function, the labelling may depend on a round key instead of the 
        plain private key. This function returns the corresponding round key
        or the plain key depending on the leakage model.

        Parameters
        ----------
        keys : List[List[int]]
                List of the keys.
        
        Returns
        -------
        round_keys : List[List[int]]
                Round keys.
        """
        pass
    def get_type(self):
        """Returns the leakage labelling."""
        return self.leakage_model
    def is_ID(self):
        """Returns True iff the labelling is the identity."""
        return self.leakage_model == "ID"
    
    def is_balanced(self):
        """Returns whether the labeling is balanced."""
        return self.leakage_model not in ["HW","HW2","HW3"]
    
    def get_class_balance(self):
        """Could return a 1D Tensor of weights to counter class imbalance. 
        
        Since it doesn't work well (see :footcite:t:`[29]picek:hal-01935318`), it is disabled and set to always return
        a balanced distribution. One should use SMOTE instead (see :footcite:t:`[108]chawla2002smote`).

        .. footbibliography::
        """
        always_return_balanced = True
        if(not always_return_balanced and self.leakage_model == "HW"):
            return   1.0/ torch.tensor([1.,8.,28.,56.,70.,56.,28.,8.,1.])
        elif(not always_return_balanced and self.leakage_model == "HW2"):
            return  1.0 / torch.tensor([93.,163.])
        elif(not always_return_balanced and self.leakage_model == "HW2"):
            return  1.0 / torch.tensor([37,256-37])
        else:
            return torch.ones(self.num_classes)
    def get_imbalanced_classes(self):
        if(self.leakage_model in["HW","HD"]):
            return [0,8]
        elif(self.leakage_model == "HW2" or self.leakage_model == "HW3"):
            return [0]
        else:
            return []
    def get_classes(self):
        """Returns the number of classes corresponding to the labelling."""
        return self.num_classes
    
    def label(self,output,before_output = None):
        if(before_output is not None):
            real = self.class_arrangement[output ^ before_output]
        else:
            real = self.class_arrangement[output]
        return real
    
    def get_same_label_values(self,guessed_label):
        """Returns the list of different values leading to the given guessed label."""
        return np.where(self.class_arrangement==guessed_label)[0]
    
    @abstractmethod
    def check_key(self,plaintext,ciphertext,key):
        """Checks if the key given is correct.
        
        Parameters
        ----------
        plaintext : List[int]
                A plaintext.
        ciphertext : List[int]
                A ciphertext corresponding to the plaintext.
        key : List[int]
                A guessed key.
        
        Returns

        bool
                Whether the given key is correct.
        
        """
        pass
    
    @abstractmethod
    def get_labelled_output(self,ptx_i,key_i,mask_i=None,byte_position=None):
        """ Get labelled output: given a subset of the plaintext and a subset of the key, 
            recover the labelled output of the leakage location.

            Parameters
            ----------
            ptx_i : int
                    i-th byte of a plaintext.
            key_i : int
                    i-th byte of a key.
            mask_i : int, default : None
                    i-th byte of the mask.
            
            Returns
            -------
            int
                    label of the output of the target function with given inputs.
        """
        pass
    @abstractmethod
    def recover_input(self,label_i,ptx_i,byte_position=None,mask_i=0):
        """ Recover input: given a label and a subset of the plaintext,
            recover the possible key(s) leading to the label.

            Parameters
            ----------
            label_i : int
                    the label corresponding to the ptx_i and a key.
            ptx_i : int
                    i-th byte of a plaintext.
            mask_i : int, default : 0
                    i-th byte of the mask.
            
            Returns
            -------
            int,List[int]
                    In case of a ID labelling, returns the possible key leading to the gien label.
                    Otherwise, returns a list with all possible keys leading to that label.
        """
        pass

    @abstractmethod
    def get_snr_poi(self,traces_waves,plaintexts,keys):
        """Calculates the position of the highest SNR peak if no countermeasures are implemented.

        Parameters
        ----------
        traces_waves : List
                The list of profiling traces.
        plaintexts : List
                The list of plaintexts.
        keys : List
                The list of keys
        Returns
        -------
        int
                The location of the mean of the highest SNR peaks over each byte.
        """
        pass
    def get_small_profiling_labels(self,plaintext,byte_position,keys,profiling = True,mask = None):
        """Returns a list of the corresponding label.
        
        Computed for example as label = HW(SBox(plaintext[i][byte_pos] XOR keys[i]) XOR mask[i][byte_pos]), for i in [0,255] 
        
        Parameters
        ----------
        plaintext : List[List[int]]
                list of plaintexts.
        byte : int
                target byte number.
        keys : int,List[int]
                For non-profiling attacks: an int <16 representing a guess for the key byte at position byte.
                For Profiling attacks: the list of profiling keys.
        mask : List[List[int]], default : None
                List of masks used. If none, will not xor the target function's output with masks.
                Should be of the same shape than the plaintext.
        Returns
        -------
        List[int]
                list of labels computed as label = Label(output of target function) for each ptx/key pair.
        """
        if(not profiling):
            key=keys
        labels=np.zeros(len(plaintext))
        for i in range(len(plaintext)):
            if(profiling):
                key=keys[i][byte_position]
            if(mask is not None):
                labels[i]=self.get_labelled_output(plaintext[i][byte_position] , key,mask[i][byte_position],byte_position)
            else:
                labels[i]=self.get_labelled_output(plaintext[i][byte_position] , key)
        return labels
    def recover_key_byte_hypothesis(self,plaintext_i,byte_position=None,mask_i = 0):
        """Returns the most key byte related to the guessed label.

            plaintext_i : int
                    a plaintext byte.
            
            Returns
            -------
            List[int]
                    list with likelihood for each key

        """
        #if(guessed_labels is None):
        guessed_labels = range(self.get_classes())
        if(self.is_ID()):
            key_hypos = np.zeros(len(guessed_labels))
            for i in guessed_labels:
                aes_in=self.recover_input(guessed_labels[i],plaintext_i,byte_position,mask_i)
                key_hypos[i]=aes_in
        else:
            key_hypos = list()
            for i in guessed_labels:
                aes_in=self.recover_input(guessed_labels[i],plaintext_i,byte_position,mask_i)
                key_hypos.append(aes_in)
        return key_hypos
    
    def __str__(self):
        return self.leakage_model
    def __repr__(self) -> str:
        return self.leakage_model


