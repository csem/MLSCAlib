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
# brief     : AES specific tools
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
import chipwhisperer.analyzer as cwa
import numpy as np
from chipwhisperer import Trace
from Crypto.Cipher import AES
from Ciphers.PySCA_tools import AES_Sbox_inv, AES_sbox, get_last_key
from Ciphers.leakage_model import LeakageModel
from error import Error
import pickle
import os

class AESLeakageModel(LeakageModel):
    """ **AES** (**ECB**) Leakage Model class. Encapsulates all the tools necessary to recover the labels of the attack.
    """
    def __init__(self,leakage_model,leakage_location="SBox"):
        """Instantiates the **AES** Leakage Model.

        Parameters
        ----------
        leakage_model : str
                Which leakage labelling to use. Can be **ID**, **HW**, **LSBS**, **LSB**, **MSB** or **HW2**.
        leakage_location : str
                Targetted **AES** function. May be *AddRoundKey*, *SBox*, *LastSBox*, or *key*.
                The LastSBox labelling is only possible when ciphertext information is
                available.
        """
        super().__init__(leakage_model)
        self.leakage_location=leakage_location
            
    def get_labelled_output(self,ptx_i,key_i,mask_i=0,byte_position=None):
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
                    Label of the output of the target function with given inputs.
        """
        before_out_tmp = None
        if(self.leakage_location == "SBox"):
            if(self.leakage_model == "HD"):
                before_out = ptx_i ^ key_i
                return self.class_arrangement[AES_sbox[before_out] ^ before_out] - 1
            else:
                target_val=AES_sbox[ptx_i ^ key_i]
        elif(self.leakage_location == "key"):
            target_val = key_i
        elif(self.leakage_location == "LastSBox"):
            assert byte_position % 4 == 0  #since the current implementation discards the shiftRows
            if(self.leakage_model == "HD"):
                ctx_i = ptx_i
                before_shift = ctx_i ^ key_i
                return self.class_arrangement[ctx_i ^ AES_Sbox_inv[before_shift]]
            else:
                ctx_i = ptx_i
                before_shift = ctx_i ^ key_i
                # target_val = AES_Sbox_inv[before_shift]  #only use output of sBox
                target_val = before_shift
        elif(self.leakage_location == "AddRoundKey"):
            target_val  = ptx_i ^ key_i
            if(self.leakage_model == "HD"):
                before_out_tmp = ptx_i
        else:
            raise Error("Invalid leakage location")
        return self.label(target_val ^ mask_i,before_out_tmp)
                
    def recover_input(self,label_i,ptx_i,byte_position=None,mask_i = 0):
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
                    In case of a ID labelling, returns the possible key leading to the given label.
                    Otherwise, returns a list with all possible keys leading to that label.
        """

        if(self.leakage_location=="SBox"):
            if(self.is_ID()):
                input_was = AES_Sbox_inv[int(label_i)^mask_i] ^ ptx_i
            elif(self.leakage_model!="HD"):
                input_was = [AES_Sbox_inv[label^mask_i] ^ ptx_i for label in self.get_same_label_values(label_i)]
            else:
                assert mask_i == 0
                input_was = [AES_sbox[label^int(ptx_i)]^int(ptx_i) for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location=="AddRoundKey"):
            if(self.is_ID()):
                input_was = int(label_i) ^ ptx_i ^ mask_i
            else:
                input_was = [label^mask_i ^ ptx_i for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location=="key"):
            if(self.is_ID()):
                input_was = int(label_i)
            else:
                input_was = [label for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location == "LastSBox"):
            pass
            if(self.is_ID()):
                input_was = int(label_i)^mask_i ^ ptx_i
            elif(self.leakage_model!="HD"):
                input_was = [label^mask_i^ ptx_i for label in self.get_same_label_values(label_i)]
            else:
                assert mask_i == 0
                input_was = [AES_sbox[label^int(ptx_i)]^int(ptx_i) for label in self.get_same_label_values(label_i)]
        return input_was
    
    def needs_cipher_and_not_ptx(self):
        """To know whether the LeakageModel needs the ciphertext instead of the plaintext."""
        return self.leakage_location == "LastSBox"
    
    def get_used_keys(self,keys):
        """Get the key used in the labelling.

        Usually, a cipher may produce round keys. Depending on the targetted
        sub-function (if the target is the LastSBox), the labelling may depend
        on a round key instead of the  plain private key. This function returns 
        the corresponding round key or the plain key depending on the leakage model. 

        Parameters
        ----------
        keys : List[List[int]]
                List of the keys.
        
        Returns
        -------
        round_keys : List[List[int]]
                Round keys.
        """
        if(self.leakage_location != "LastSBox"):
            return keys
        else:
            return np.array([get_last_key(k) for k in keys])
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
        plaintext,ciphertext,key = bytes(list(plaintext)),bytes(list(ciphertext)),bytes(list(key))
        cipher = AES.new(key, AES.MODE_ECB)
        ciphertext_dec = cipher.encrypt(plaintext)
        return ciphertext_dec == ciphertext

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
        class MockTraces(object):
            """MockTraces : Class containing Traces
            
            Class having the same interface as chipwhisperer.analyser.Traces in order to use the chipwhisperer SNR calculation.
            """
            def __init__(self):
                self.traces=[]
            def add_trace(self,wave,ptx,key):
                self.traces.append(Trace(np.array(wave),ptx,None,key))
            def add_traces(self,waves,plains,keys):
                for w,p,k in zip(waves,plains,keys):
                    self.add_trace(w,p,k)
            def __len__(self):
                return len(self.traces)
            def __getitem__(self,idx):
                return self.traces[idx]

        traces=MockTraces()
        traces.add_traces(traces_waves,plaintexts,keys)
        if(self.leakage_location == "SBox"):
            model=cwa.leakage_models.sbox_output
        elif(self.leakage_location=="AddRoundKey" or self.leakage_location == "key"):
            model=cwa.leakage_models.plaintext_key_xor
        elif(self.leakage_location == "LastSBox"):
            model = cwa.leakage_models.last_round_state
        maxes_add=[]
        for b in range(16):
            snr = cwa.calculate_snr(traces, leak_model=model, bnum=b, db=False)
            maxes_add.append(np.argmax(np.array(snr)))
        poi=int(np.mean(maxes_add))
        return poi
   
    def __str__(self):
        return self.leakage_model + "("+self.leakage_location+")"
    def __repr__(self) -> str:
        return self.leakage_model + "("+self.leakage_location+")"
