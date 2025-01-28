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
# brief     : Adds/removes noise to the traces
# author    : AISY lab & modified by Lucas D. Meier
# date      : 2023
# copyright : publicly available code snippets on https://github.com/AISyLab/Denoising-autoencoder
# On some functions : Copyright (c) 2020 AISyLab @ TU Delft

from enum import Enum
import os
import shutil
import sys
import numpy as np
from math import ceil

import torch
from Data.AISY_tools import AisyTools
from Architectures.autoencoders import old_cnn, MODEL_CONSTANT
from tensorflow.keras import models
from Data.data_manager import _StateMachine, DataManager, PreProcessing
from error import Error
import h5py
from collections.abc import Iterable
from sklearn.decomposition import PCA
from scipy.signal import decimate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule,OneSidedSelection,RandomUnderSampler
from imblearn.combine import SMOTETomek
from pandas import DataFrame

class NoiseTypes(Enum):
    """Available noise & countermeasures types.

    
    """
    GAUSSIAN = 0
    RANDOM_DELAY = 1
    CLOCK_JITTER = 2
    SHUFFLING = 3
    SHUFFLING_VARIANT = 3.5
    MA = 4  #Can also be used as a Noise deformation 

class _NoiseStateMachine(Enum):
    CREATION = 0
    NOISE_ADDITION = 1
    AUTOENCODER_COMPUTATION = 2
    NOISE_FINISH_USAGE_IN_ATTACK = 3

class ImbalanceResolution(Enum):
    """Imbalance resolution techniques. 

    For help, see: https://imbalanced-learn.org/stable/references/index.html .

    """
    NONE = 0
    SMOTE = 1 #best choice
    ONESIDED_SELECTION = 2  #OSS
    NEIGHBOURHOOD_CLEANING_RULE = 3  #NCR
    RANDOM_UNDERSAMPLER = 3.5
    OSS_SMOTE = 4
    """Applies first a *ONESIDED_SELECTION* and then *SMOTE*."""
    NCR_SMOTE = 5
    """Applies first a *NEIGHBOURHOOD_CLEANING_RULE* and then *SMOTE*."""
    SMOTE_TOMEK = 6
    """Applies first *SMOTE* and then *TOMEK*."""


NOISE_DIR="NoisyTraces/"
MODEL_DIR="ModelCache/"
class CustomDataManager(DataManager):
    """Custom Data Manager: class that handles countermeasures and autoencoder noise removal.

    This class allows to add countermeasures to a clean dataset. It also allows to test
    autoencoder-based noise removal as performed in :footcite:t:`[217]rmnoise`. In this class,
    the words "noise" and "countermeasures" refer to the same thing. Additionally, the class
    allows to try different techniques combatting the imbalance class issues inspired 
    from :footcite:t:`[29]picek:hal-01935318`. Moreover, you can pre-process the traces with the following
    techniques: *PCA*, *MA*, *LDA*, *QDA*, *decimation*, *append_noise*, *SUB-MEAN*, *MESSERGES*, *SYNTHETIC*. Refer to
    their function for detailed explanation.

    The inner class working follows a **FSM**, with the states described in the StateMachine enum. It can only
    go forward (i.e. after calling attack specific functions such as *prepare_x_data*, it won't
    accept adding noise or using autoencoders) for efficiency reasons. This means:
    
    #. You can't add noise after having used the autoencoder or created attack data (via *prepare_data* or *get_res_name*).
    #. You can't use the autoencoder after creation of attack data.
    #. You don't have to add any noise or use the autoencoders to make the attack.
    #. You can't add noise/preprocessing/autoencoder techniques after having queried *self.get_ns*.

    This class will allways calculate the noise for all the traces on all samples. It will
    store the results in a *.h5* file once it reaches the *NOISE_FINISH_USAGE_IN_ATTACK* state.
    The output file_name is composed of the original *file_name* plus each noise added in it.
        
    The class will use the output of the autoencoder if the function remove_noise was called.
    Otherwise, it will use the output of the function add_noise if it was called. If none of these
    functions were called, the CustomDataManager will act as a regular DataManager.
    

    .. footbibliography::

    """

    def __init__(self,na="all",ns="all",nt="all",fs=0,clean_file_name='file_name_attack.h5',\
                    file_name_profiling ='file_name_attack.h5',\
                        databases_path="/path/to/Databases/", has_countermeasures=False,blind = False,force_cpu=False,\
                            overwrite=False,imbalance_resolution = ImbalanceResolution.NONE,sampling_strategy = 'auto',\
                            shuffling_level = 20,clock_range=4, delay_amplitude = 5,gaussian_noise = 8,noisy_file_name = None,\
                                remove_mask = False,pre_processing = PreProcessing.HORIZONTAL_STANDARDIZATION):
        """ Initializes the CustomDataManager. 

        Parameters
        ----------
        na : int, str
                how many attack traces to use.
        ns : int, str
                how many samples per trace to use.
        nt : int, str
                how many profiling traces to use.
        fs : int, default : 0
                First sample to use in each trace. Acts like an offset.
        clean_file_name : str
                The file name containing the traces and the metadata in .h5 format. Should be "clean".
        seed : int, default : 5437
                The seed to use to partition and shuffle the data.
        databases_path : str, default : "/path/to/Databases/"
                The directory of the target file.
        has_countermeasures : bool, default : False
                Whether the underlying data was generated using counteremasures.
        blind : bool, default : False
                Whether to ignore the attack keys.
        force_cpu : bool, default : False
                If true, won't use the GPU's even if available.  
        imbalance_resolution : ImbalanceResolution
                In case of imbalanced data, which technique to use on the training data to counter imbalance.

                * NONE: no technique.
                * UNDERSAMPLE: discard some samples from the majority classes.
                * SMOTE: use a SMOTE technique, which creates new rare class samples from neighboring points. 

                Caution: when using SMOTE, you will not be able to compute the fast GE in non profiled attacks 
                (or the fast GE of the training data in profiled attacks).
        sampling_strategy : str, list or callable, default : 'auto'
                If imbalance_resolution is set to a downsampling strategy. Sampling information to downsample the data set.
                When str, specify the class targeted by the resampling. Note the the number of samples will not be equal in each.
                Possible choices are:

                * *'majority'*: resample only the majority class;
                * 'not minority': resample all classes but the minority class;
                * 'two minorities': resample only two minorities classes;
                * 'not majority': resample all classes but the majority class;
                * 'all': resample all classes;
                * 'auto': equivalent to 'not minority'.
                    
                When list, the list contains the classes targeted by the resampling.
                When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
                The values correspond to the desired number of samples for each class.
        shuffling_level : int, default : 20
                For the shuffling noise, intensity of the shuffling. Number of sample pairs to swap.
        clock_range : int, default : 4
                For the clock jitter noise, intensity of the clock jittering.
        delay_amplitude : int, default : 10
                For the Random Delay noise, how big the delays should be.
        gaussian_noise : int, default : 8
                For the gaussian noise, how big the noise should be. Zero means no noise at all.
        overwrite : bool, default : False
                Whether to overwrite existing noisy/denoised data. As the noise addition/removal is
                file specific, use this option only for test purposes.
        noisy_file_name : str, default : None
                In case you have generated noisy traces yourself, specify the path to those traces in here.
                If you specify a file, calling add_noise() will discard your file and create a new noisy file.
        """

        super().__init__(na,ns,nt,fs,file_name_attack=clean_file_name,file_name_profiling=file_name_profiling,\
                            databases_path=databases_path,has_countermeasures = has_countermeasures,\
                                blind = blind,force_cpu=force_cpu,remove_mask=remove_mask,pre_processing=pre_processing)
        self.noise_state = _NoiseStateMachine.CREATION
        self.clean_file_name = clean_file_name
        self.noisy_file_name = noisy_file_name
        self.autoencoder_output_file_name = None
        self.noises = set()
        self.sampling_strategy=sampling_strategy
        self.shuffling_level = shuffling_level
        self.clock_range= clock_range
        self.delay_amplitude = delay_amplitude
        self.gaussian_noise = gaussian_noise
        self.overwrite = overwrite
        self.imbalance_resolution = imbalance_resolution
        self.preprocessings={"PCA":(False,None),"pearson":(False,None),"MA":(False,None,None),"LDA":False,"QDA":False,"decimation":(False,None),"append_noise":(False,None),
                                "SUB-MEAN": False,"MESSERGES":False,"SYNTHETIC":(False,0.0)}
        if(not os.path.exists(self.databases_path + NOISE_DIR)):
            os.mkdir(self.databases_path + NOISE_DIR)
        if(not os.path.exists(os.path.dirname(__file__) + "/"+MODEL_DIR)):
            os.mkdir(os.path.dirname(__file__) + "/"+MODEL_DIR)

    def add_noise(self,noise_types):
        """Adds noise/countermeasures to the data.
        
        This function can only be called once.
        
        Parameters
        ----------
        noise_types : NoiseTypes, List[NoiseTypes]
                The types of noise to add. Can be a list of noises or just one NoiseTypes.
        
        """
        if(self.noise_state in [_NoiseStateMachine.AUTOENCODER_COMPUTATION,_NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK]):
            raise Error("Invalid state. Can't add noise after autoencoder usage or attack data creation.")
        elif(self.noise_state == _NoiseStateMachine.NOISE_ADDITION):
            raise Error("Invalid state. Can only add noises once.")
        if(not isinstance(noise_types,Iterable)):
            noise_types = [noise_types]
        if(len(noise_types) == 0):
            raise Error("Can't add no noise. Please enter at least one or multiple NoiseTypes.")
        self.noise_state = _NoiseStateMachine.NOISE_ADDITION
        for n in noise_types:
            if(not isinstance(n,NoiseTypes)):
                raise TypeError("Each noise type should be instance of the NoiseTypes enum.")
            self.noises.add(n)
        self.noisy_file_name = "NOISY_" + self._get_noise_file_name() +self.clean_file_name
        noisy_path_file = self.databases_path + NOISE_DIR +self.noisy_file_name
        if((not self.overwrite) and os.path.exists(noisy_path_file) and os.path.getsize(noisy_path_file) > 0): 
            return noisy_path_file
        elif(not os.path.exists(noisy_path_file)):
            #create empty file !
            open(noisy_path_file,'a').close()

        clean_path_file = self.databases_path + self.clean_file_name
        train_data, attack_data = AisyTools.load_only_traces(clean_path_file)
        shutil.copy(clean_path_file, noisy_path_file)
        try:
            out_file = h5py.File(noisy_path_file, "r+")
        except:
            print("Error: can't open HDF5 file '%s' for writing ..." %
                noisy_path_file)
            sys.exit(-1)
        na_all = len(attack_data)
        nt_all = len(train_data)
        all_traces = np.concatenate((train_data, attack_data), axis=0)

        ns_all = len(all_traces[0])
        if(NoiseTypes.MA in self.noises):
            all_traces = CustomDataManager._moving_average(all_traces, window_size=4,step = 4)
            ns_all = len(all_traces[0])
        if(NoiseTypes.SHUFFLING in self.noises):
            all_traces = AisyTools.addShuffling(all_traces,self.shuffling_level)
            ns_all = len(all_traces[0])
        if(NoiseTypes.SHUFFLING_VARIANT in self.noises):
            all_traces = AisyTools.addShuffling_variant(all_traces)
            ns_all = len(all_traces[0])
        if(NoiseTypes.RANDOM_DELAY in self.noises):
            all_traces = AisyTools.addRandomDelay(all_traces,self.delay_amplitude)
            ns_all = len(all_traces[0])
        if(NoiseTypes.CLOCK_JITTER in self.noises):
            all_traces = AisyTools.addClockJitter(all_traces,self.clock_range)
            ns_all = len(all_traces[0])
        if(NoiseTypes.GAUSSIAN in self.noises):
            print("Adding Gaus",flush=True)
            all_traces = AisyTools.addGaussianNoise(all_traces,self.gaussian_noise)
            print("Added Gauss !",flush=True)
        #if(self._ns is not None):
         #   self._ns = len(all_traces[0])
        del out_file['Profiling_traces/traces']
        out_file.create_dataset('Profiling_traces/traces', data=np.array(all_traces[:nt_all], dtype=np.int8))
        del out_file['Attack_traces/traces']
        out_file.create_dataset('Attack_traces/traces', data=np.array(all_traces[nt_all:], dtype=np.int8))
        out_file.close()
        return noisy_path_file


    def _prep_data_for_autoencoder(self):
        """Prepares the data for the autoencoder. Makes sure training data and labels have same shapes and are scaled."""
        clean_database = self.databases_path + self.clean_file_name
        if(self.noisy_file_name == self.clean_file_name):
            noisy_traces_file = clean_database
        else:
            noisy_traces_file = self.databases_path + NOISE_DIR +self.noisy_file_name
        train_noisy,attack_noisy = AisyTools.load_only_traces(noisy_traces_file)
        train_clean,attack_clean = AisyTools.load_only_traces(clean_database)
        nt_all = len(train_noisy)  #should use train_noisy as a reference in case of given noisy_file_name
        X_noisy_o =  np.concatenate((train_noisy,attack_noisy), axis=0)
        Y_clean_o =  np.concatenate((train_clean,attack_clean), axis=0)
        #Add padding, to match the NN model needs
        length = max(len(r) for r in X_noisy_o)
        max_len = int(MODEL_CONSTANT * ceil(float(length) / float(MODEL_CONSTANT)))
        X_noisy_o = AisyTools.manual_padding(X_noisy_o,max_len)
        Y_clean_o=AisyTools.manual_padding(Y_clean_o,len(X_noisy_o[0]))
        X_noisy_o = X_noisy_o.reshape(
            (X_noisy_o.shape[0], X_noisy_o.shape[1], 1))
        Y_clean_o = Y_clean_o.reshape(
            (Y_clean_o.shape[0], Y_clean_o.shape[1], 1))      
        X_noisy = AisyTools.scale(X_noisy_o)
        Y_clean = AisyTools.scale(Y_clean_o) 
        return X_noisy,Y_clean,nt_all,Y_clean_o

    def get_res_name(self,leakage_model = None):
        """Returns a string describing the DataManager attributes.     
        
        Parameters
        ----------
        leakage_model : LeakageModel, default : None
                If the DataManager has not been fully initialized, it need a 
                leakage model in order to compute the res name. O/W, 
                you can ommit it.
        
        """
        if(self.noise_state != _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            self._finish_noise_addition()
        if(self.state != _StateMachine.FUNCTION and leakage_model is None):
            raise Error("Can't get res name value before data creation without a leakage_model.")
        elif(self.state != _StateMachine.FUNCTION):
            self._finish_init(leakage_model)
        prepr = ""
        for k,v in self.preprocessings.items():
            if(hasattr(v,'__len__')):
                if(v[0]):
                    prepr += k+"_"
            elif(v):
                prepr += k+"_"
        if(self.imbalance_resolution is not ImbalanceResolution.NONE):
            prepr += self.imbalance_resolution.name+"_"
        return prepr + super().get_res_name(leakage_model)
    def apply_mean_removal(self):
        """Removes the mean of every trace from each trace as done in :footcite:t:`[63]martinasek2013optimization`.
        
        .. footbibliography::
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["SUB-MEAN"] = True
    def use_synthetic_traces(self,noise = 0.0):
        """Function to replace the current power traces by synthetic ones.
        
        Parameters
        ----------
        noise : float, default : 0.0
                How much noise to add in the clean synthetic data.
        
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["SYNTHETIC"] = (True,noise)

    def apply_PCA(self,nb_components=225):
        """Applies a Principal Component Analysis on the data. Computation is done before the attack stage begins.

        Parameters
        ----------
        nb_components : int, default : 225
                How many principal components to retain.
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["PCA"] = (True,nb_components)

    def apply_pearson(self,nb_components=50):
        """Applies a Principal Component Analysis on the data. Computation is done before the attack stage begins.

        Parameters
        ----------
        nb_components : int, default : 50
                How many important samples to retain.
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["pearson"] = (True,nb_components)

    def apply_MA(self,window_size = 20, step_size = 1):
        """Applies a Moving Average filter on the data. Computation is done before the attack stage begins.

        Parameters
        ----------
        window_size : int, default : 20
                Window size of the MA.
        step_size : int, default : 1
                Step size of the MA. Avoid using a step_size > window_size in order to use all sample points in the
                attack.
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["MA"] = (True,window_size,step_size)
    def apply_LDA(self):
        """Applies a LinearDiscriminantAnalysis on the data. Computation is done before the attack stage begins."""
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["LDA"] = True
    def apply_QDA(self):
        """Applies a QuadraticDiscriminantAnalysis on the data. Computation is done before the attack stage begins."""
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["QDA"] = True
    def apply_decimation(self,downsampling_factor = 5):
        """Applies downsampling on the data. Computation is done before the attack stage begins.
        
        Parameters
        ----------
        downsampling_factor : int, default : 5
                The downsampling factor.
        
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["decimation"] = (True,downsampling_factor)
    def append_noise(self,size_ratio = 1.0):
        """Appends noise to the traces, effectively making them size_ratio longer.
        
        Parameters
        ----------
        size_ratio : float, default : 1.0
                Size of the noise to be added, in comparison to the actual power measurement trace.
                E.g. if set to 1.0, the number of samples will double.
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["append_noise"] = (True,size_ratio)
    def apply_messerges(self):
        """Applies a Second Order preprocessing.

        Applies a technique as described in :footcite:t:`[64]10.1007/3-540-44499-8_19`.

        .. footbibliography::
         
        """
        if(self.noise_state is _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("It's too late to add preprocessing.")
        self.preprocessings["MESSERGES"] = True
    
    @staticmethod
    def _moving_average(X, window_size=35,step = 1):
        convolved =  np.array([np.convolve(x, np.ones(window_size), 'valid') / window_size for x in X])
        if(step==1):
            return convolved
        return np.array([c[range(0,len(c),step)] for c in convolved])
    @staticmethod
    def _messerges(X):
        func = np.add
        return np.array([np.concatenate([u[i:] for i,u in enumerate((func(np.abs(np.array([x])).T, - np.abs(np.array([x])))))]) for x in X])


    @staticmethod
    def _get_useless_index(corr_matrix,nb_out):
        nb_in = corr_matrix.shape[0]
        useless_features = set()
        # get te index we would use to sort the pearson correlation matrix, without considering the diagonal of 1s
        idx = ((1-np.eye(corr_matrix.shape[0])) *abs(corr_matrix.to_numpy())).flatten().argsort()[::-1]
        #we discard one of two elements as the matrix is symmetric
        i3 = np.delete(idx, np.arange(0, idx.size, 2))
        #we get the 2D coordinate of the ranking
        i2=np.unravel_index(i3,corr_matrix.shape)
        print("index2 ",i2[0])
        for col in i2[0]:  #use only use one of the dimension of the ranking. We could use the other as well
            useless_features.add(col)  #if not already added, add the index of the column with least importance
            if(nb_in - len(useless_features) <= nb_out):
                return list(useless_features)
      
    def _reduce_pearson(self,X,nb_out,profiling=True):
        # old_useless=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 928, 929, 930, 931, 933, 934, 935, 936, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2011, 2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2103, 2104, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2121, 2122, 2123, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2295, 2296, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2325, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2349, 2350, 2351, 2352, 2354, 2355, 2356, 2358, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2376, 2377, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2484, 2485, 2486, 2487, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2506, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2518, 2519, 2520, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2558, 2559, 2561, 2562, 2564, 2565, 2566, 2567, 2568, 2569, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2603, 2604, 2605, 2606, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2628, 2629, 2630, 2631, 2632, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2703, 2704, 2705, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2722, 2723, 2725, 2726, 2727, 2728, 2730, 2731, 2732, 2733, 2736, 2737, 2738, 2741, 2742, 2743, 2745, 2746, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2889, 2890, 2891, 2892, 2893, 2894, 2896, 2897, 2898, 2901, 2902, 2903, 2904, 2905, 2907, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2942, 2943, 2944, 2946, 2947, 2948, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2960, 2961, 2962, 2964, 2965, 2966, 2967, 2969, 2970, 2971, 2974, 2975, 2976, 2977, 2978, 2979, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2995, 2996, 2997, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3008, 3009, 3010, 3011, 3013, 3014, 3015, 3016, 3017, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3046, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3069, 3070, 3071, 3072, 3073, 3074, 3076, 3077, 3078, 3079, 3081, 3083, 3084, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3117, 3118, 3119, 3120, 3122, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3135, 3136, 3137, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3190, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3217, 3218, 3219, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3242, 3243, 3244, 3245, 3246, 3248, 3249, 3250, 3251, 3252, 3254, 3255, 3257, 3258, 3259, 3261, 3265, 3266, 3267, 3268, 3270, 3271, 3272, 3274, 3275, 3276, 3277, 3278, 3279, 3281, 3282, 3283, 3284, 3287, 3288, 3289, 3290, 3293, 3294, 3295, 3296, 3297, 3299, 3300, 3303, 3305, 3307, 3308, 3309, 3311, 3312, 3315, 3316, 3318, 3319, 3320, 3321, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3338, 3340, 3341, 3342, 3343, 3344, 3346, 3347, 3349, 3350, 3351, 3352, 3353, 3355, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3367, 3369, 3371, 3374, 3376, 3377, 3379, 3380, 3381, 3383, 3384, 3385, 3386, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3410, 3411, 3412, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3426, 3427, 3428, 3429, 3431, 3432, 3433, 3434, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3444, 3445, 3446, 3448, 3449, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3476, 3477, 3478, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3493, 3494, 3495, 3496, 3497, 3498, 3499])
        if(profiling):
            X_prof,X_attack = X
            X_attack = DataFrame(X_attack)#.drop(old_useless,axis=1).to_numpy())
            X_to_be_corred = DataFrame(X_prof)#.drop(old_useless,axis=1).to_numpy())
        else:
            X_to_be_corred = DataFrame(X)
        if(False and "AES_RD" in self.file_name_attack and nb_out == 50):
            print("Re-using Pearson 50 from aes rd: WARNING 50")
            keep50000 = [1928,  2039,  2268,  2557,  2624,  2648,  2661,  2674,  2711,  2720,  2734,  2741,  2765,  2788,  2816,  2828,  2833,  2945,  2989,  2995,  3004,  3018,  3076,  3080,  3091,  3092,  3143,  3211,  3250,  3267,  3294,  3306,  3327,  3339,  3340,  3362,  3385,  3390,  3392,  3395,  3398,  3430,  3448,  3458,  3466,  3469,  3470,  3483,  3488,  3491]
            # keep40000_rd=[1272, 2002, 2241, 2276, 2474, 2513, 2527, 2547, 2643, 2666, 2674, 2720, 2745, 2754, 2755, 2804, 2831, 2875, 2893, 2897, 2963, 3007, 3018, 3048, 3079, 3100, 3105, 3182, 3194, 3199, 3203, 3221, 3226, 3240, 3241, 3242, 3247, 3258, 3265, 3294, 3319, 3365, 3392, 3395, 3400, 3406, 3407, 3435, 3436, 3461]
            indexes = set(range(3500))
            for i in range(3500):
                if i in keep50000:
                    indexes.remove(i)
            useless_features = list(indexes)
        elif(False and "aes_hd" in self.file_name_attack and nb_out == 500):
            print("Re-using Pearson 500 from aes hd: WARNING 500")
            keep500=[1,  2,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  19,  23,  26,  30,  32,  37,  40,  43,  47,  50,  54,  59,  61,  64,  67,  68,  69,  72,  73,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  96,  97,  98,  103,  104,  105,  106,  107,  109,  112,  113,  114,  115,  116,  117,  118,  119,  120,  126,  130,  133,  141,  147,  153,  157,  158,  160,  164,  166,  169,  172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187,  188,  189,  190,  191,  192,  193,  194,  195,  196,  197,  200,  202,  205,  207,  210,  213,  215,  216,  217,  218,  219,  220,  221,  222,  223,  224,  227,  229,  231,  233,  235,  243,  248,  252,  255,  257,  262,  265,  268,  274,  276,  277,  278,  280,  281,  282,  284,  285,  288,  290,  295,  297,  298,  299,  300,  303,  305,  306,  307,  310,  315,  318,  320,  321,  322,  323,  324,  325,  330,  333,  338,  344,  350,  359,  368,  370,  373,  380,  381,  383,  384,  385,  387,  389,  390,  393,  396,  400,  402,  408,  409,  411,  414,  418,  421,  426,  427,  428,  430,  436,  442,  449,  451,  454,  459,  464,  466,  470,  476,  480,  484,  486,  490,  491,  493,  497,  499,  500,  501,  503,  507,  512,  514,  515,  516,  517,  518,  519,  520,  521,  522,  526,  529,  530,  532,  533,  534,  535,  536,  542,  545,  548,  552,  557,  561,  567,  569,  574,  580,  586,  589,  590,  592,  597,  598,  600,  603,  604,  606,  607,  609,  612,  617,  623,  625,  627,  629,  633,  635,  637,  639,  641,  645,  648,  651,  655,  663,  667,  670,  678,  680,  686,  688,  691,  694,  698,  700,  702,  705,  708,  713,  715,  724,  727,  729,  732,  734,  737,  739,  742,  743,  749,  755,  759,  761,  765,  768,  772,  778,  781,  786,  794,  799,  801,  803,  805,  808,  812,  816,  819,  823,  825,  827,  828,  831,  833,  835,  838,  841,  843,  844,  845,  846,  847,  851,  855,  857,  859,  864,  866,  869,  872,  874,  877,  883,  885,  891,  894,  896,  901,  902,  905,  907,  911,  913,  916,  920,  922,  928,  930,  931,  932,  933,  935,  936,  937,  938,  939,  947,  948,  949,  950,  951,  952,  956,  963,  966,  970,  979,  981,  985,  986,  990,  993,  997,  999,  1000,  1001,  1005,  1007,  1010,  1011,  1012,  1014,  1017,  1022,  1029,  1030,  1031,  1032,  1033,  1034,  1035,  1044,  1048,  1052,  1054,  1056,  1057,  1060,  1064,  1066,  1070,  1074,  1079,  1087,  1091,  1095,  1098,  1103,  1107,  1111,  1116,  1118,  1119,  1120,  1121,  1122,  1123,  1124,  1125,  1126,  1127,  1128,  1129,  1130,  1131,  1132,  1136,  1145,  1147,  1148,  1149,  1150,  1151,  1152,  1153,  1154,  1155,  1156,  1157,  1158,  1159,  1160,  1161,  1162,  1163,  1164,  1165,  1166,  1167,  1168,  1169,  1170,  1171,  1172,  1173,  1174,  1175,  1176,  1179,  1181,  1183,  1184,  1185,  1189,  1195,  1198,  1200,  1201,  1204,  1206,  1208,  1210,  1217,  1219,  1223,  1226,  1229,  1234,  1236,  1237,  1238,  1239,  1240,  1241,  1244,  1245,  1247,  1248,  1249]
            indexes = set(range(1250))
            for i in range(1250):
                if i in keep500:
                    indexes.remove(i)
            useless_features = list(indexes)
        else:
            print("Calculating pearson ...")
            corr_matrix = X_to_be_corred.corr()
            useless_features = CustomDataManager._get_useless_index(corr_matrix,nb_out)
            for r in range(corr_matrix.shape[0]):
                if(r not in useless_features):
                    print("Useful feature: ",r) 
        # assert False
        # print("useless",len(useless_features))
        # keep=[1068, 1080, 1330, 1840, 1869, 1956, 2030, 2102, 2191, 2324, 2353, 2375, 2430, 2472, 2483, 2507, 2532, 2560, 2563, 2642, 2694, 2706, 2721, 2744, 2858, 2872, 2941, 2945, 2963, 2998, 3026, 3037, 3085, 3138, 3139, 3205, 3216, 3273, 3285, 3286, 3291, 3302, 3306, 3314, 3322, 3339, 3370, 3378, 3387, 3413]
        if(profiling):
            # return X_to_be_corred[X_to_be_corred.index.isin(keep40000)],X_attack[X_attack.index.isin(keep40000)]
            return X_to_be_corred.drop(useless_features,axis=1).to_numpy(),X_attack.drop(useless_features,axis=1).to_numpy()
        else:
            return X_to_be_corred.drop(useless_features,axis=1).to_numpy()


    def _append_noise_to_x(trace,size_factor,gaussian_std = None):
        if(gaussian_std is None):
            gaussian_std = np.std(trace) /4
            #print("std is ",gaussian_std," mean is ",np.mean(trace))
        gaussian_noise=np.random.normal(loc = 0.,scale = gaussian_std,size=(trace.shape[0],int(trace.shape[1]*size_factor)))#.to(trace.device)
        return np.concatenate((trace,gaussian_noise),-1)
    @staticmethod
    def _create_synthetic_trace(traces,plaintexts,keys,leakage_model):
        ns = len(traces[0])
        synthethic = np.zeros_like(traces)
        for b in range(16):
            labels=leakage_model.get_small_profiling_labels(plaintexts, b, keys,True)
            if(b>=0):
                for i,l in enumerate(labels):
                    synthethic[i][ns//4 + 10*b] = 2* l

        return synthethic
            

    def _finish_init(self, leakage_model):
        super()._finish_init(leakage_model)
        if(not (self.preprocessings["PCA"][0] or self.preprocessings["pearson"][0] or self.preprocessings["LDA"] or self.preprocessings["QDA"]\
             or self.preprocessings["decimation"][0] or self.preprocessings["MA"][0] or self.preprocessings["append_noise"][0]\
                 or self.preprocessings["SUB-MEAN"] or self.preprocessings["MESSERGES"] or self.preprocessings["SYNTHETIC"][0])):
            return
        if(self.nt == 0):
            input_attack_shape = self.x_attack.shape
            compute = np.array(self.x_attack.cpu())
        else:
            input_attack_shape = self.x_attack.shape
            input_profiling_shape = self.x_profiling.shape
            compute = [np.array(self.x_profiling.cpu()),np.array(self.x_attack.cpu())]

        if(self.preprocessings["SYNTHETIC"][0]):
            if(self.profiling):
                compute[0] = CustomDataManager._create_synthetic_trace(compute[0],np.array(self.plaintext_profiling.cpu()),np.array(self.key_profiling.cpu()),leakage_model)
                compute[1] = CustomDataManager._create_synthetic_trace(compute[1],np.array(self.plaintext_attack.cpu()),np.array(self.key_attack.cpu()),leakage_model)
            else:
                compute = CustomDataManager._create_synthetic_trace(compute,np.array(self.plaintext_attack.cpu()),np.array(self.key_attack.cpu()),leakage_model)

        if(self.preprocessings["SUB-MEAN"]):
            if(self.profiling):
                compute= list((lambda x : x - np.mean(x,axis = 0))(c) for c in compute)
            else:
                compute =  (lambda x : x - np.mean(x,axis = 0))(compute)
        if(self.preprocessings["PCA"][0]):
            nb_out = self.preprocessings["PCA"][1]
            if(self.profiling):
                compute =  list(PCA(nb_out).fit_transform(c) for c in compute)
            else:
                compute =  PCA(nb_out).fit_transform(compute)
        if(self.preprocessings["pearson"][0]):
            nb_out = self.preprocessings["pearson"][1]
            compute = self._reduce_pearson(compute,nb_out,self.profiling)
        if(self.preprocessings["LDA"]):
            raise Error("Not implemented")
            self.x_profiling,self.x_attack =  LDA().transform(self.x_profiling),LDA().transform(self.x_attack)
        if(self.preprocessings["QDA"]):
            raise Error("Not implemented")
            #compute =  QDA().fit()
        if(self.preprocessings["decimation"][0]):
            factor = self.preprocessings["decimation"][1]
            if(self.profiling):
                compute = list(decimate(c,factor) for c in compute)
            else:
                compute =  decimate(compute,factor)
        if(self.preprocessings["MA"][0]):
            w = self.preprocessings["MA"][1]
            s = self.preprocessings["MA"][2]
            if(self.profiling):
                compute = list(CustomDataManager._moving_average(X, window_size=w,step = s) for X in compute)
            else:
                compute =  CustomDataManager._moving_average(compute, window_size=w,step = s) 
        if(self.preprocessings["append_noise"][0]):
            size_factor = self.preprocessings["append_noise"][1]
            if(self.profiling):
                compute = list(CustomDataManager._append_noise_to_x(c,size_factor) for c in compute)
            else:
                compute =  CustomDataManager._append_noise_to_x(compute,size_factor)
        if(self.preprocessings["MESSERGES"]):
            if(self.profiling):
                compute = list(CustomDataManager._messerges(c) for c in compute)
            else:
                compute =  CustomDataManager._messerges(compute)
        if(self.nt>0):
            self.x_profiling,self.x_attack = compute
        else:
            self.x_attack = compute
        self.x_attack = torch.from_numpy(np.ascontiguousarray(self.x_attack))  #.to(self.device)  #first turn into np.array for faster execution
        if(len(self.x_attack.shape)>2):
            self.x_attack = self.x_attack.reshape((input_attack_shape[0],self.x_attack.shape[2]))
        if(self.nt > 0):
            self.x_profiling = torch.from_numpy(np.ascontiguousarray(self.x_profiling))  #.to(self.device)
            if(len(self.x_profiling.shape)>2):
                self.x_profiling = self.x_profiling.reshape(input_profiling_shape[0],self.x_profiling.shape[2])
        self._ns=len(self.x_attack[0])
        if(not self.profiling):
            self._temp_x_attack = torch.clone(self.x_attack)
            self._temp_x_attack = self._temp_x_attack.reshape([len(self._temp_x_attack),len(self._temp_x_attack[0]),1])
           # self._temp_x_attack = torch.from_numpy(self._temp_x_attack)
            self._temp_x_attack = torch.transpose(self._temp_x_attack.float(),1,2)
        else:
            self._temp_x_attack=None

    def apply_autoencoder(self,epochs=100,batch_size=128,verbose=1, path_to_reuse_autoencoder = None):
        """Removes the noise using an autoencoder.

        If the function add_noise() was called before, this function will remove the noise added.
        If the function add_noise() was never called, the current function will try to remove
        the noise of the "clean" input database file. (which may increase the SNR by reducing
        some gaussian noise). 
        If path_to_reuse_autoencoder is not None, il will reuse an old autoencoder trained before.
        You may want to use autoencoders tailored to the countermeasures in the traces you have only.
        Inspired from the work done by :footcite:t:`[17]wu2020remove`.

        .. footbibliography::

        Parameters
        ----------
        epochs : int, default : 100
                For how many epochs to train the autoencoder. 100 should be optimal.
        batch_size : int, default : 128
                Batch size used for learning. 128 should be optimal (authors have tried 32, 64, 128, 256).
        verbose : int, default : 1
                Verbose used during computation. Can be between 0 and 2 (inclusive). The higher it is, the
                more is going to be printed.
        path_to_reuse_autoencoder : str, default : None
                If you want to reuse an autoencoder trained before, specify the path to that autoencoder here.
        
        Returns
        -------
        str, str
                The path to the denoised traces and the path to the trained/used autoencoder model.
        
        """
        if(isinstance(epochs,str)):
            raise Error("The path of the autoencoder should be given at last.")
        if(self.noise_state == _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            raise Error("Invalid state. Can't use the autoencoder after attack data creation.")
        if(len(self.noises) == 0 and self.noisy_file_name != None):
            #User given noise file
            copy_from_noisy = True
        else:
            copy_from_noisy = False
        if(self.noisy_file_name == None):
            print("Warning: will denoise the original file.") 
            self.noisy_file_name = self.clean_file_name
        self.noise_state = _NoiseStateMachine.AUTOENCODER_COMPUTATION
        self.autoencoder_output_file_name = "AUTOENCODERED_"+str(epochs)+  self._get_noise_file_name() +self.clean_file_name
        auto_path_file = self.databases_path + NOISE_DIR + self.autoencoder_output_file_name
        if((not self.overwrite) and os.path.exists(auto_path_file) and os.path.getsize(auto_path_file) > 0):
            #output data already computed before
            print("Data is here",auto_path_file)
            return auto_path_file
        elif(not os.path.exists(auto_path_file)):
            #create empty file !
            open(auto_path_file,'a').close()
        if(not copy_from_noisy):
            clean_path_file =  self.databases_path + self.clean_file_name
            shutil.copy(clean_path_file, auto_path_file)
        else:
            noisy_path_file = self.databases_path + self.noisy_file_name
            shutil.copy(noisy_path_file, auto_path_file)

        X_noisy,Y_clean,nt_all,Y_clean_o = self._prep_data_for_autoencoder()
        if(path_to_reuse_autoencoder is None):
            print('train model...')
            autoencoder = old_cnn(0.0001, len(X_noisy[0]))

            if(len(X_noisy)>15000):
                TRAINING_SET_SIZE = 10000 # optimal parameter as taken from [17]wu2020remove, 10000 for training and 
                VALIDATION_SET_SIZE = 5000  # 5000 for validation
            else:
                split_rate=0.95
                TRAINING_SET_SIZE = int(split_rate * len(X_noisy))
                VALIDATION_SET_SIZE = len(X_noisy) - TRAINING_SET_SIZE
            if(len(X_noisy) == len(Y_clean)):
                X_noisy_shuffled,Y_clean_shuffled = DataManager._unison_shuffled_copies(X_noisy,Y_clean)
            else:
                X_noisy_shuffled,Y_clean_shuffled = X_noisy,Y_clean
            print("shapes are ",X_noisy.shape,Y_clean.shape)
            autoencoder.fit(X_noisy_shuffled[:TRAINING_SET_SIZE], Y_clean_shuffled[:TRAINING_SET_SIZE], 
                            validation_data=(X_noisy_shuffled[TRAINING_SET_SIZE:TRAINING_SET_SIZE+VALIDATION_SET_SIZE],\
                                Y_clean_shuffled[TRAINING_SET_SIZE:TRAINING_SET_SIZE+VALIDATION_SET_SIZE]),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=2*verbose % 3)
            model_save_location = os.path.dirname(__file__) + "/"+MODEL_DIR +\
                "AUTOENCODERMODEL_"+str(epochs)+  self._get_noise_file_name() +self.clean_file_name
            autoencoder.save(model_save_location,save_format="h5")
        else:
            print("re use old model")
            model_save_location = path_to_reuse_autoencoder
            autoencoder =  models.load_model(path_to_reuse_autoencoder)

        AisyTools.apply_autoencoder_create_h5(autoencoder,X_noisy,Y_clean_o,nt_all,self.databases_path + NOISE_DIR + self.autoencoder_output_file_name)

        return self.autoencoder_output_file_name,model_save_location



    def _finish_noise_addition(self):
        """Marks the end of the noise management and enter in USAGE in ATTACK mode."""
        if(self.autoencoder_output_file_name != None):
            self.file_name_attack = self.autoencoder_output_file_name
            self.databases_path = self.databases_path + NOISE_DIR
        elif(self.noisy_file_name != None):
            self.file_name_attack = self.noisy_file_name
            self.databases_path = self.databases_path  + NOISE_DIR
        else:
            #print("Warning, no noise has been added or removed.")
            self.file_name = self.clean_file_name
            self.databases_path = self.databases_path
        self.has_countermeasures = self.has_countermeasures or \
                        NoiseTypes.CLOCK_JITTER in self.noises or \
                          NoiseTypes.RANDOM_DELAY in self.noises or \
                            NoiseTypes.SHUFFLING in self.noises or \
                                NoiseTypes.SHUFFLING_VARIANT in self.noises
        self.noise_state = _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK

    def _get_noise_file_name(self):
        """Returns a handy description of the types of noise being used."""
        name = ""
        if(NoiseTypes.GAUSSIAN in self.noises):
            name += "GAUSSIAN-" + str(self.gaussian_noise)+"_"
        if(NoiseTypes.RANDOM_DELAY in self.noises):
            name += "RANDOM-DELAY-" + str(self.delay_amplitude)+"_"
        if(NoiseTypes.CLOCK_JITTER in self.noises):
            name += "CLOCK-JITTER-" + str(self.clock_range)+"_"
        if(NoiseTypes.SHUFFLING in self.noises):
            name += "SHUFFLING-" + str(self.shuffling_level)+"_"
        if(NoiseTypes.SHUFFLING_VARIANT in self.noises):
            name += "SHUFFLING-VARIANT" 
        if(NoiseTypes.MA in self.noises):
            name += "MA-4-4_"
        return name

    def _balance(self,X,y,leakage_model):
        
        shape_orig = X.shape
        a = np.array(y.cpu())
        unique, counts = np.unique(a, return_counts=True)
        if(self.sampling_strategy == "two minorities"):
            for idx in leakage_model.get_imbalanced_classes():
                counts[idx] = max(counts)
            # counts[8] = max(counts)
            strategy = dict(zip(unique, counts))
        else:
            strategy = self.sampling_strategy
        print("Before balancing:",dict(zip(unique, counts)))
        X=X.reshape([len(X),X.shape[2]])

        if(self.imbalance_resolution in [ImbalanceResolution.ONESIDED_SELECTION,ImbalanceResolution.OSS_SMOTE]):
            method = OneSidedSelection(sampling_strategy = strategy)
        elif(self.imbalance_resolution is ImbalanceResolution.SMOTE):
            # for idx in leakage_model.get_imbalanced_classes():
            #     print("Counts",counts," on id ",idx, "with max count",max(counts))
            #     counts[idx] = max(counts)
            method = SMOTE(sampling_strategy=strategy)
        elif(self.imbalance_resolution is ImbalanceResolution.SMOTE_TOMEK):
            #n = max(counts)//3
            method = SMOTETomek()#tomek =TomekLinks(sampling_strategy = lambda x : {0:n,1:n,2:n,3:n,4:n,5:n,6:n,7:n,8:n}))
        elif(self.imbalance_resolution in [ImbalanceResolution.NEIGHBOURHOOD_CLEANING_RULE,ImbalanceResolution.NCR_SMOTE]):
            method = NeighbourhoodCleaningRule(sampling_strategy = strategy)  #"majority"
        elif(self.imbalance_resolution is ImbalanceResolution.RANDOM_UNDERSAMPLER):
            #m = max(counts)//3
           # print("Counts is ",counts)
            # counts[2] = counts[2]//2.8
            # counts[3] = counts[3] // 5.6
            # counts[4] = counts[4] // 7
            # counts[5] = counts[5] // 5.6
            # counts[6] = counts[6] // 2.8
            #n = m
            method = RandomUnderSampler()#sampling_strategy=dict(zip(unique, counts)))
        print("X is ",X)
        X,y = method.fit_resample(X.cpu(),y.cpu())

        if(self.imbalance_resolution in [ImbalanceResolution.NCR_SMOTE,ImbalanceResolution.OSS_SMOTE]):
            a = np.array(y)
            unique, counts = np.unique(a, return_counts=True)
            print("After downsampling only:",dict(zip(unique, counts)))
            min_neighbors = min(min(counts),6) - 1
            method2 = SMOTE(k_neighbors=min_neighbors)
            X,y = method2.fit_resample(X,y)

        a = np.array(y)
        unique, counts = np.unique(a, return_counts=True)
        print("After balancing:",dict(zip(unique, counts)))
        X = torch.from_numpy(X).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        X = X.reshape([len(X),shape_orig[1],shape_orig[2]])
        return X,y
    def prepare_unprofiled_data(self, byte, dk, key_guess,leakage_model, split_val = False,split_rate = 0.8,min_id=None,max_id=None):
        '''Prepares the data for each key guess in a non profiled attack.

        Loads, standardizes, transforms to binary-categorical labels. It also reshapes the 
        data to allow usage on the models. The min_id/max_id are used in incremental learning
        to find the minimum number of attack traces needed to reach GE=1 for example.
        Additionally, this function will perform any balancing technique if requested by the
        user beforehand.

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
        if(self.noise_state != _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            self._finish_noise_addition()
        res = super().prepare_unprofiled_data(byte, dk, key_guess,leakage_model,split_val,split_rate, min_id, max_id)
        if(len(res)==7):
            x_attack_res,label_attack,plaintext_attack,key_attack,x_val,label_val,plaintext_val = res
        else:
            x_attack_res,label_attack,plaintext_attack,key_attack,x_attack_accumulated,x_val,label_val,plaintext_val = res
        if(self.imbalance_resolution is ImbalanceResolution.NONE):
            return res
        else:
            X,y = self._balance(x_attack_res,label_attack,leakage_model)
        print("Real NA was",x_attack_res.shape[0],"and now is",len(X))
        if(len(res)==7):
            return X,y,plaintext_attack,key_attack,x_val,label_val,plaintext_val
        else:
            return X,y,plaintext_attack,key_attack,x_attack_accumulated,x_val,label_val,plaintext_val
    def prepare_profiled_data(self, byte, leakage_model,use_attack_as_val=True, split_rate=0.95, dk=False):
        '''Prepares the data for profiled attacks.
        
        This function may perform any balancing technique if requested by the
        user beforehand.

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
        if(self.noise_state != _NoiseStateMachine.NOISE_FINISH_USAGE_IN_ATTACK):
            self._finish_noise_addition()
        res =  super().prepare_profiled_data(byte,leakage_model, use_attack_as_val, split_rate, dk)
        if(self.imbalance_resolution is ImbalanceResolution.NONE):
            return res
        x_profiling_res,label_profiling,x_val,label_val,plaintext_val,mask_val,x_attack_res,label_attack,plaintext_attack,key_attack,mask_attack = res #self.ptx / self.key
        X,y = self._balance(x_profiling_res,label_profiling,leakage_model)
        print("Real NT was",x_profiling_res.shape[0],"and now is",len(X))
        return X,y,x_val,label_val,plaintext_val,mask_val,x_attack_res,label_attack,plaintext_attack,key_attack,mask_attack


