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
# brief     : ML attack on AES
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 



'''
This script runs a ML attack on AES.

'''
import os
from Attacks.attack import TrainingMethods
from Attacks.blind_unprofiled import BlindUnProfiled

from Attacks.unprofiled import UnProfiled
from Attacks.profiled import Profiled
from Ciphers.AES_DPA_contest import DPAContestLeakage
from Ciphers.AES_leakage import AESLeakageModel
from Data.custom_manager import ImbalanceResolution, NoiseTypes, CustomDataManager
from Data.data_manager import PreProcessing
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #discard tensorflow warnings
import inspect
import sys
from time import gmtime, strftime, time
from error import Error

#warnings.filterwarnings('ignore')

redraw = None
def _launch_attack(attacker,byte_range,min_for_ge,guess_range,attack_key,get_fastGE,record_stuff):
    """Launches the attack according to the given arguments."""
    record_file_name,record_trials,info_for_plot,record_axis = record_stuff
    # print("USING CUSTOM REC")
    if(min_for_ge):
        attacker.attack_bytes_with_min_traces(byte_range,guess_range)
    else:
        if(isinstance(attacker,UnProfiled)):
            if(record_file_name is not False):
                raise Error("Non profiled records not implemented")
            if(attack_key):
                attacker.threshold = 0.45
                attacker.attack_key(timeout = 4 * 60 * 60)
            else:
                attacker.attack_bytes(byte_range,guess_range)
        else:
            if(attack_key):
                if(record_file_name is not False):
                    raise Error("Non profiled records not implemented")
                attacker.attack_key(timeout = 4 * 60 * 60)#
            else:
                if(record_file_name is not False):
                    if(isinstance(byte_range,int)):
                        byte_range=[byte_range]
                    res = []
                    for byte in byte_range:
                        attacker._printd(f"Attacking byte {byte}...")
                        if("logit" in record_axis):
                            record_axis="epoch"
                            logit=True
                        else:
                            logit=False
                        res.append(attacker.record_attack(byte=byte,number_of_trials = record_trials,\
                            info_for_filename = record_file_name,info_for_plot = info_for_plot,x_axis=record_axis,redraw_this=redraw\
                                ,target_model_substring = "",logit=logit))
                    return res
                else:#,kwargs = {"fast_GE":get_fastGE}
                    attacker.attack_bytes(byte_range,get_fast_GE = get_fastGE)

def script_parser(user_input):
    """Parses the use input arguments and create attack classes."""
    user_arguments={}
    byte_range=0
    next_input=None
    remove_mask=False
    LeakageModel = AESLeakageModel
    user_arguments["noise_types"] = list()
    i=0
    t0 = time()
    try:
        while(i<len(user_input)):
            if(user_input[i] in ["-a","--na","--attack","-na","na"]):
                if(user_input[i+1] in ["min","all"]):
                    user_arguments["na"] = user_input[i+1]
                else:
                    user_arguments["na"] = int(user_input[i+1])
                i+=2 
            elif(user_input[i] in ["--append-noise","-append-noise"]):
                user_arguments["append-noise"] = float(user_input[i+1])
                i+=2 
            elif(user_input[i] in ["-b","--byte","--byte-number"]):                                             
                if(user_input[i+1]=="all"):
                    byte_range=range(16)
                elif("," in user_input[i+1]):
                    byte_range = [int(x) for x in user_input[i+1].split(",")]
                elif("-" in user_input[i+1]):
                    inputs=user_input[i+1].split("-")
                    byte_range=range(int(inputs[0]),int(inputs[1]))
                else:
                    byte_range=int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["--ba","-ba","--batch_size","--batch","-batch","-batch_size","-batch-size","--batch-size"]):
                user_arguments["batch_size"] = int(user_input[i+1])
                i+= 2
            elif(user_input[i] in ["--blind","-blind","--real"]):
                user_arguments["blind"] = True
                i+=1
            elif(user_input[i] in ["--cpu","-cpu"]):
                user_arguments["force_cpu"] = True
                i+=1
            elif(user_input[i] in ["--decimation","--dec","-decimation"]):
                user_arguments["decimation"] = True
                i+=1
            elif(user_input[i] in ["-d","--dim","-dim"]):
                user_arguments["dim"] = int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-dk","--dk","--DK","-DK"]):
                user_arguments["dk"]=True
                i+=1
            elif(user_input[i] in ["-e","--epochs","--epoch"]):
                user_arguments["epochs"]=int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["--fast","--fast","-fast"]):
                user_arguments["fast"] = True
                i+=1
            elif(user_input[i] in ["-f","--file","--file-name","-fa","--fa","--file-name-attack"]):
                user_arguments["file_name_attack"] = user_input[i+1]
                if(not "." in user_arguments["file_name_attack"]):
                    user_arguments["file_name_attack"] += ".h5"
                i+=2
            elif(user_input[i] in ["-fp","--filep","--file-namep","-fp","--fp","--file-name-profiling"]):
                user_arguments["file_name_profiling"]=user_input[i+1]
                if(not "." in user_arguments["file_name_profiling"]):
                    user_arguments["file_name_profiling"] += ".h5"
                i+=2  
            elif(user_input[i] in ["-fr","--file-r","--file-res","--fr","--file-result","--results-path"]):
                user_arguments["results_sub_path"]=user_input[i+1]
                i+=2
            elif(user_input[i] in ["--fs","-fs","--first-sample"]):
                user_arguments["fs"] = int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-h","--help","-man"]):
                with open("help.txt") as f:
                    print(f.read())
                return -1
            elif(user_input[i] in ["-i","--info","--i"]):
                user_arguments["info"] = str(user_input[i+1])
                i+=2
            elif(user_input[i] in ["--imb", "--bal", "--imbalance","-bal","-imb","-imbalance","--imbalance-resolution"]):
                s =  str(user_input[i+1]).upper()
                if("NON" in s):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.NONE
                elif(s=="SMOTE"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.SMOTE
                elif(s == "OSS"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.ONESIDED_SELECTION
                elif(s == "NCR"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.NEIGHBOURHOOD_CLEANING_RULE
                elif(s=="OSS-SMOTE"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.OSS_SMOTE
                elif(s=="NCR-SMOTE"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.NCR_SMOTE
                elif(s=="SMOTE-TOMEK"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.SMOTE_TOMEK
                elif(s == "RANDOM"):
                    user_arguments["imbalance-resolution"] = ImbalanceResolution.RANDOM_UNDERSAMPLER
                else:
                    print("Warning: discarding the balancing resolution because of invalid specifier.")
                i+=2
            elif(user_input[i] in ["-k","--k","-key","--key","--attack-key"]):
                user_arguments["key"]=True
                i+=1
            elif(user_input[i] in ["--lm","--leakage_model","-lm"]):
                user_arguments["leakage_model"]=(user_input[i+1])
                i+=2
            elif(user_input[i] in ["--LTH", "--pruning","-lth","-LTH","--lth","-pruning","-prune","--prune"  ]):
                user_arguments["training"] = TrainingMethods.PRUNING_LOCAL
                try:
                    user_arguments["LTH"] = float(user_input[i+1])
                    if(user_arguments["LTH"]>1):
                        user_arguments["LTH"]/=100
                    i+=1
                except:
                    pass
                i+=1
            elif(user_input[i] in ["--LTHH", "--prune-half-half","-lthh","-LTHH","--lthh"]):
                user_arguments["training"] = TrainingMethods.PRUNING_HALF_EPOCHS_LOCAL
                try:
                    user_arguments["LTH"] = float(user_input[i+1])
                    if(user_arguments["LTH"]>1):
                        user_arguments["LTH"]/=100
                    i+=1
                except:
                    pass
                i+=1
            elif(user_input[i] in ["-lc","--loc","--leakage_location","-loc","--lc"]):
                user_arguments["leakage_location"]=(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-lrs","--lr-scheduler","--lrs","--learning-rate-scheduler"]):
                user_arguments["lr_schedule"]=True#[int(user_input[i+1]),float(user_input[i+2])]
                i+=1
            elif(user_input[i] in ["-lr","--lr","--learning-rate"]):
                user_arguments["learning_rate"]=float(user_input[i+1]) #[int(user_input[i+1]),float(user_input[i+2])]
                i+=2
            elif(user_input[i] in ["-L","--loss"]):
                user_arguments["loss"]=(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-m","--model"]):
                user_arguments["model_name"]=(user_input[i+1])
                i+=2 
            elif(user_input[i] in ["--mess","--messerges","-mes","-mess","-messerges","--MESS","-MESS"]):
                user_arguments["messerges"]=True
                i+=1 
            elif(user_input[i] in ["--MA","-MA","-ma","-MA"]):
                user_arguments["MA"]=True
                i+=1
            elif(user_input[i] in ["-n","--noise","-noise"]):
                user_arguments["noise"]=True
                i+=1 
            elif(user_input[i] in ["--noise-types","-noise-types","--noise-type","-noise-type","-GAUSS"]):
                n = user_input[i+1].upper()
                if("GAUSSIAN" in n or n == "-GAUSS"):
                    user_arguments["noise_types"].append(NoiseTypes.GAUSSIAN)
                if("RANDOM_DELAY" in n):
                    user_arguments["noise_types"].append(NoiseTypes.RANDOM_DELAY)
                if("CLOCK_JITTER" in n):
                    user_arguments["noise_types"].append(NoiseTypes.CLOCK_JITTER)
                if("SHUFFLING" in n and (not ("SHUFFLING_VARIANT" in n) or n.count("SHUFFLING")>1)):
                    user_arguments["noise_types"].append(NoiseTypes.SHUFFLING)
                if("SHUFFLING_VARIANT" in n):
                    user_arguments["noise_types"].append(NoiseTypes.SHUFFLING_VARIANT)
                i+=2 
            elif(user_input[i] in ["--no-regularization"]):
                user_arguments["lambdas"]=None
                i+=1
            elif(user_input[i] in ["-p","--np","--profiling","-t","--nt","-np","-nt"]):
                if(user_input[i+1]=="all"):
                    user_arguments["nt"]=user_input[i+1]
                else:
                    user_arguments["nt"]=int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["--path-database","-pd","--pd","-path-database","--databases-path", "-databases-path","--dp","-dp"]):
                user_arguments["databases_path"]=user_input[i+1]
                i+=2
            elif(user_input[i] in ["--path-results","-pr","--pr","-path-results","--results-path", "-results-path"]):
                user_arguments["results_path"]=user_input[i+1]
                i+=2
            elif(user_input[i] in ["--pre-processing","--preprocessing","--pre","-pre","--prep", "-prep"]):
                # prep = user_input[i+1]
                if(len(user_input[i+1])==1):
                    user_arguments["pre_processing"]=PreProcessing(int(user_input[i+1]))
                else:
                    user_arguments["pre_processing"]=PreProcessing[user_input[i+1]]
                i+=2
            elif(user_input[i] in ["python","main.py","--python","-python","--and-then-execute"] or "python" in user_input[i]):
                if(user_input[i+1] in ["python","main.py","--python","-python","--and-then-execute"]):
                    next_input = user_input[i+2:]
                else:
                    next_input = user_input[i+1:]
                break
            elif(user_input[i] in ["--PCA","-PCA","-pca","-pca"]):
                user_arguments["PCA"]=True
                i+=1
            elif(user_input[i] in ["--PEARSON","--pearson","-Pearson","--Pearson","-pearson","-PEARSON"]):
                user_arguments["pearson"]=True
                nb_out_pearson = int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-r","--ra","--range"]):
                #inputs=user_input[i+1].split("-")
                if("," in user_input[i+1]):
                    user_arguments["guess_range"] = [int(x) for x in user_input[i+1].split(",")]
                elif("-" in user_input[i+1]):
                    inputs=user_input[i+1].split("-")
                    user_arguments["guess_range"]=range(int(inputs[0]),int(inputs[1]))
                else:
                    user_arguments["guess_range"]=range(int(user_input[i+1]))
                i+=2
            elif(user_input[i] in ["-record","--rec","--record","-rec"]):
                user_arguments["record"]=(user_input[i+1])
                i+=2 #--record-trials            
            elif(user_input[i] in ["-record-trials","--rec-trials","--record-trials","-rec-trials"]):
                user_arguments["record-trials"]=int(user_input[i+1])
                i+=2 #--record-trials --record-axis
            elif(user_input[i] in ["-record-axis","--rec-axis","--record-axis","-rec-axis"]):
                user_arguments["record-axis"]=user_input[i+1]
                i+=2 
            elif(user_input[i] in ["-o","--o","--optimizer"]):
                user_arguments["optimizer"]=(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-s","--ns","--samples","-ns","ns"]):
                user_arguments["ns"]=int(user_input[i+1])
                i+=2#--SUB-MEAN, --SUB
            elif(user_input[i] in ["--SUB","-SUB","-sub","--SUB-MEAN","--sub-mean","-sub-mean","--SUB_MEAN"]):
                user_arguments["SUB_MEAN"]=True
                i+=1
            elif(user_input[i] in ["--SYN","-SYN","--SYNTHETIC"]):
                user_arguments["SYNTHETIC"]=True
                i+=1
            elif(user_input[i] in ["--seed","-seed"]):
                if(user_input[i+1][0] in ["r","random","None","none","n","N","no","No","NO"]):
                    print("Will not use seed")
                    user_arguments["seed"] = None #int(time()*1000) % 2**31
                else:
                    user_arguments["seed"]=int(user_input[i+1])
                i+=2
            elif(user_input[i] in ["-v","--verbose"]):
                user_arguments["verbose"]=int(user_input[i+1])
                i+=2
                
            elif(user_input[i] in ["--with-regularization"]):
                user_arguments["lambdas"]=[0.003,0.003,0.001,0.001]
                i+=1
            elif(time()>(t0 + 20)):
                raise Error("too long to parse the args. Args must be wrong.")
            else:
                print("Invalid parameter, discarding ",(user_input[i]))
                i+=1

    except :
        print("Invalid arguments.")
        with open("help.txt") as f:
            print(f.read())
        return -1

    if('nt' in user_arguments and user_arguments["nt"] == 0):
        if("na" in user_arguments and user_arguments["na"] in ["min",0]):
            min_for_ge = True
        else:
            min_for_ge=False
        if(user_arguments.get("blind",False)):
            attack_class = BlindUnProfiled
        else:
            attack_class = UnProfiled
    else:
        min_for_ge=False
        attack_class=Profiled
    user_arguments["has_countermeasures"] = "desync" in user_arguments.get("file_name_attack","blob")  or\
                                                "Set4"in user_arguments.get("file_name_attack","blob") or\
                                                    "Set5" in user_arguments.get("file_name_attack","blob")
    init_arguments=inspect.getfullargspec(attack_class.__init__)
    user_arguments["leakage_model"] = LeakageModel(user_arguments.get("leakage_model",\
                                            init_arguments.defaults[init_arguments.args.index('leakage_model')-1].leakage_model),\
                                                user_arguments.get("leakage_location",\
                                            init_arguments.defaults[init_arguments.args.index('leakage_model')-1].leakage_location))
#na,ns,nt,file_name,seed = 5437,databases_path="/path/to/Databases/",has_countermeasures=False
    default_data_manager = init_arguments.defaults[init_arguments.args.index('data_manager')-1]

    overwrite = False
    user_arguments["data_manager"] = CustomDataManager(user_arguments.get("na",default_data_manager.na),\
                                            user_arguments.get("ns",default_data_manager._ns),\
                                            user_arguments.get("nt",default_data_manager.nt),\
                                            user_arguments.get("fs",default_data_manager.fs),\
                                            user_arguments.get("file_name_attack",default_data_manager.file_name_attack),\
                                            user_arguments.get("file_name_profiling",default_data_manager.file_name_profiling),\
                                            user_arguments.get("databases_path",default_data_manager.databases_path),\
                                            user_arguments.get("has_countermeasures",default_data_manager.has_countermeasures),\
                                            user_arguments.get("blind",default_data_manager.blind),\
                                            user_arguments.get("force_cpu",default_data_manager.force_cpu),\
                                            overwrite,\
                                            user_arguments.get("imbalance-resolution",ImbalanceResolution.NONE),\
                                            sampling_strategy = 'auto',\
                                            noisy_file_name = None ,\
                                            remove_mask=remove_mask
                                            )
    if(user_arguments.get("SUB_MEAN",False)):
        user_arguments["data_manager"].apply_mean_removal()
    if(user_arguments.get("MA",False)):
        user_arguments["data_manager"].apply_MA(window_size=2,step_size=2)  #Some FPGA traces need a FS = 3
    if(user_arguments.get("PCA",False)):
        user_arguments["data_manager"].apply_PCA()
    if(user_arguments.get("pearson",False)):
        user_arguments["data_manager"].apply_pearson(nb_out_pearson)
    if(user_arguments.get("decimation",False)):
        user_arguments["data_manager"].apply_decimation(downsampling_factor=2)
    if(user_arguments.get("messerges",False)):
        user_arguments["data_manager"].apply_messerges()
    if(user_arguments.get("SYNTHETIC",False)):
        user_arguments["data_manager"].use_synthetic_traces(0.0)
    if(user_arguments.get("append-noise",0.0)>0):
        user_arguments["data_manager"].append_noise(user_arguments["append-noise"])
    # user_arguments["data_manager"].append_noise(0.2)
    if(len(user_arguments["noise_types"])>0):
        if(user_arguments.get("verbose",1)>=2):
            print("Adding ",user_arguments["noise_types"])
        user_arguments["data_manager"].add_noise(user_arguments["noise_types"])
    # user_arguments["data_manager"].add_noise(list([NoiseTypes.GAUSSIAN]))#,NoiseTypes.SHUFFLING_VARIANT]))
    #user_arguments["data_manager"].apply_autoencoder(path_to_reuse_autoencoder="MLSCAlib/Data/ModelCache/AUTOENCODERMODEL_100NOISY_MA-4-4_Set2_sync.h5")
#    user_arguments["data_manager"].apply_autoencoder(path_to_reuse_autoencoder="MLSCAlib/Data/ModelCache/AUTOENCODERMODEL_100Set3_saved_set5.h5")


    # user_arguments["lambdas"] = None # [0.0003,0.0003,0.000008,0.000008]

    final_arguments=[]

    if(not ("verbose" in user_arguments)):
        if("record" in user_arguments):
            user_arguments["verbose"] = 0
    for arg,default in zip(init_arguments.args[1:],init_arguments.defaults): 
        if(arg=="results_path" and ( arg in user_arguments or "results_sub_path" in user_arguments)):
            if(arg in user_arguments):
                default=user_arguments["results_path"]
                if(default[-1] != "/"):
                    default += "/"
            if("results_sub_path" in user_arguments):
                fr = user_arguments["results_sub_path"]
                fr.replace("\\","/")
                if("/" == fr[0]):
                    fr=fr[1:]
                if(fr[-1] != "/"):
                    fr += "/"
                if(len(fr)<2):
                    raise Error("Results frath too short. Avoid slashes in the path name.")
            else:
                fr=""
            if(not os.path.exists(default + fr )):
                os.mkdir(default + fr )
            final_arguments.append(default + fr)
        else:
            final_arguments.append(user_arguments.get(arg,default))
    attacker=attack_class(*final_arguments)
    #print("has ",str(attacker))
    record_stuff = [user_arguments.get("record",False),user_arguments.get("record-trials",10),user_arguments.get("info",""),user_arguments.get("record-axis","epoch")]
    return attacker,byte_range,min_for_ge,user_arguments.get("guess_range",range(256)),user_arguments.get("key",False),user_arguments.get("fast",False),record_stuff,next_input

if __name__ == "__main__":
    next_input = sys.argv[1:]
    while(next_input is not None):
        state=script_parser(next_input)
        if(state!=-1):
            _launch_attack(*state[:-1])
            next_input = state[-1]
        else:
            next_input=None