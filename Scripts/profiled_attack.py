import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from Attacks.profiled import Profiled


if __name__ == "__main__":
    attacker = Profiled(epochs=10,seed=None,batch_size=100)
    attacker.attack_byte(3)