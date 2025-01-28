import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
import numpy as np
import heapq
from Attacks.attack import TrainingMethods
from Attacks.unprofiled import UnProfiled
from Attacks.profiled import Profiled


if __name__=="__main__":
    byte = 3
    profiled = Profiled(epochs=10)
    ge,min_ge_for_1,output_probas_on_keys = profiled.attack_byte(byte,get_output_probas=True)
    if(ge > 1):
        probas = np.sum((output_probas_on_keys), axis=0) #removed np.log
        guesses = heapq.nlargest(20, range(len(probas)), probas.take)
        non_profiled = UnProfiled(epochs=16,training=TrainingMethods.PRUNING_LOCAL)
        ge_final = non_profiled.attack_byte(byte,guesses)  #run non profiled on 20 best key guesses
