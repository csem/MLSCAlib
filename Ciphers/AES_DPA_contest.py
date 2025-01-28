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
from Ciphers.PySCA_tools import AES_Sbox_inv, AES_sbox
from Ciphers.leakage_model import LeakageModel
from error import Error
class DPAContestLeakage(LeakageModel):
    """ **AES** (**ECB**) Leakage Model class. Encapsulates all the tools necessary to recover the labels of the attack.
    """
    def __init__(self,leakage_model,leakage_location="SBox"):
        """Instantiates the **AES** Leakage Model as used in the DPA contest challenge :footcite:t:`[229]DBLP:conf/date/NassarSGD12`.
    
        .. footbibliography::

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
        self.leakage_location = leakage_location
        self.mask = [  3 , 12 , 53 , 58 , 80  ,95 ,102 ,105 ,150 ,153, 160, 175 ,197, 202, 243 ,252]
        print("DPAcontest leakagemodel class. Experimental. If you use our dataset, the only attack supported is on byte 0.")
    def mask_offset(self,off,i):
        return self.mask[(off + i) % 16]
    def sub(self,X,i):
        return AES_sbox[X^self.mask[i%16]] ^ self.mask[(i+1)%16]
    def inv_sub(self,X,i):
        return AES_Sbox_inv[X^self.mask[(i+1)%16]] ^ self.mask[i%16]
    def get_labelled_output(self,ptx_i,key_i,offset_i=None,byte_position=None):
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
        assert offset_i is not None and byte_position is not None
        before_out = None
        if(self.leakage_location == "SBox"):
            ptx2 = ptx_i ^self.mask_offset(offset_i,byte_position)
            ptx3 = ptx2 ^ key_i
            if(self.leakage_model == "HD"):
                before_out = ptx3
            target_val = self.sub(ptx3,offset_i)
        elif(self.leakage_location == "key"):
            target_val = key_i
        elif(self.leakage_location == "LastSBox"):
            raise Error("Can not use LastSBox location with DPAAContestV4 leakage model.")
        elif(self.leakage_location == "AddRoundKey"):
            ptx2 = ptx_i ^self.mask_offset(offset_i,byte_position)
            ptx3 = ptx2 ^ key_i
            target_val  = ptx3
            if(self.leakage_model == "HD"):
                before_out = ptx_i
        else:
            raise Error("Invalid leakage location")
        return self.label(target_val ,before_out)
                
    def recover_input(self,label_i,ptx_i,byte_position=None,offset_i = None):
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
        assert offset_i is not None and byte_position is not None
        # print("recover arguments",label_i,ptx_i,byte_position,offset_i)
        if(self.leakage_location=="SBox"):
            if(self.is_ID()):
                input_was = self.inv_sub(int(label_i),offset_i) ^ ptx_i ^ self.mask_offset(offset_i,byte_position)
            else:
                input_was = [self.inv_sub(label,offset_i) ^ ptx_i ^ self.mask_offset(offset_i,byte_position) for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location=="AddRoundKey"):
            if(self.is_ID()):
                input_was = int(label_i) ^ ptx_i ^self.mask_offset(offset_i,byte_position)
            else:
                input_was = [label ^ ptx_i ^self.mask_offset(offset_i,byte_position)for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location=="key"):
            if(self.is_ID()):
                input_was = int(label_i)
            else:
                input_was = [label for label in self.get_same_label_values(label_i)]
        elif(self.leakage_location == "LastSBox"):
            raise Error("Ciphertext not availlable in the DPACOntestV4. Could not use the LastSBox target.")
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
            raise Error("Can not use LastSBox location with DPAAContestV4 leakage model.")
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
        raise NotImplementedError()

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
        raise NotImplementedError()
   
    def __str__(self):
        return self.leakage_model + "("+self.leakage_location+")"
    def __repr__(self) -> str:
        return self.leakage_model + "("+self.leakage_location+")"