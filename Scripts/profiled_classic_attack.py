import sys
import os
sys.path.append(os.path.abspath(os.getcwd()))
from Attacks.profiled_classic import ProfiledClassic


if __name__ == "__main__":
    attacker = ProfiledClassic(verbose=2)
    attacker.attack_byte(3)