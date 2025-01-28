import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))


from Ciphers.AES_leakage import AESLeakageModel
from Ciphers.leakage_model import LeakageModel
from Data.data_manager import DataManager


d = DataManager(force_cpu=True,file_name_attack="ASCAD.h5")
l = AESLeakageModel(leakage_location="SBox",leakage_model="ID")
data = d.prepare_profiled_data(3,l)
p,c=d.get_check_data()
print("c,p",c,p,data[-1])
print(l.check_key(p,c,data[-1][0]))