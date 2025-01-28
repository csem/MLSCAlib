import sys
import os

import numpy as np
sys.path.append(os.path.abspath(os.getcwd()))
from Attacks.attack import TrainingMethods
from Attacks.unprofiled import UnProfiled
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #discard tensorflow warnings
import sys
import warnings

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    byte = 0
    range_ = 100
    res_path = "/path/to/PlotsResults/dim_contest/"
    GEs_dim_0 = np.empty((10,4,2,2))
    GEs_dim_1 = np.empty((10,4,2,2))
    for seed_ in range(42,52):
        print("################## New seed ####################",seed_)
        for i_b,batch in enumerate([9,50,100,500]):
            print("---------------- New batch ----------------",batch)
            for i_e,epochs in enumerate([20,40]):
                for i_p,pruner in enumerate([TrainingMethods.PRUNING_LOCAL]):#,TrainingMethods.PRUNING_HALF_EPOCHS]):
                    print(f"Seed {seed_}, batch {batch}, epochs {epochs}")
                    attacker = UnProfiled(epochs=epochs,seed=seed_,batch_size=batch,training=pruner,dim=0,results_path=res_path,verbose=1)
                    ge = attacker.attack_byte(byte,guess_list=range(range_))#,get_stats=True)
                    GEs_dim_0[seed_ - 42][i_b][i_e][i_p] = ge

                    attacker = UnProfiled(epochs=epochs,seed=seed_,batch_size=batch,training=pruner,dim=1,results_path=res_path,verbose=1)
                    ge = attacker.attack_byte(byte,guess_list=range(range_))#,get_stats=True)
                    GEs_dim_1[seed_ - 42][i_b][i_e][i_p] = ge
                    
                
            
        
    np.save(res_path+"GEs_0.npy",GEs_dim_0)
    np.save(res_path+"GEs_1.npy",GEs_dim_1)