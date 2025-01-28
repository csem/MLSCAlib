
import itertools
from time import gmtime, strftime, time
from Data.custom_manager import NoiseTypes, CustomDataManager



def all():
    noises=[NoiseTypes.GAUSSIAN,NoiseTypes.CLOCK_JITTER,NoiseTypes.RANDOM_DELAY,NoiseTypes.SHUFFLING]
    t1=time()
    t2=None
    for r in range(1,5):
        for comb in itertools.combinations(noises,r):
            print("Comb is ",comb)
            noise1 = CustomDataManager('all','all','all',overwrite=False)
            noise1.add_noise(comb)
            noise1.remove_noise()
            if(t2==None):
                t2=time()
                print("First round finished in ",str(int(t2-t1)),"s. Will finish at: ",\
                    strftime('%D::%H:%M:%S', gmtime( 60*60 * 2 +t2 + (15-1)*(t2-t1))))

def once():
    noises=[NoiseTypes.RANDOM_DELAY,NoiseTypes.GAUSSIAN,NoiseTypes.CLOCK_JITTER,NoiseTypes.SHUFFLING]
    t1=time()
    t2=None
    for noise in noises:
        print("Noise is ",noise)
        noise1 = CustomDataManager('all','all','all',overwrite=False)
        noise1.add_noise(noise)
        noise1.remove_noise()
        if(t2==None):
            t2=time()
            print("First round finished in ",str(int(t2-t1)),"s. Will finish at: ",\
                strftime('%D::%H:%M:%S', gmtime( 60*60 * 2 +t2 + (len(noises)-1)*(t2-t1))))

if __name__=="__main__":
    print("Hello world")
    once()
    