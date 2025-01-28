import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from Attacks.unprofiled import UnProfiled


if __name__ == "__main__":
    attacker = UnProfiled(epochs=10)
    ge = attacker.attack_byte(3,guess_list=range(20))
    if(ge==1):
        attacker.attack_byte_with_min_traces(3,guess_list=range(20))