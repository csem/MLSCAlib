# MIT License

# Copyright (c) 2020 AISyLab @ TU Delft
# copyright : publicly available on https://github.com/AISyLab/Denoising-autoencoder
# modified for our needs.
import h5py
import numpy as np
import os,sys
from error import Error
import random

class AisyTools(object):
    def __init__():
        pass
    @staticmethod
    def apply_autoencoder_create_h5(autoencoder,X_noisy,Y_clean_o,nt_all,labeled_traces_file):
        """Effectively removes the noise from the noisy traces using the autoencoder model and stores the result."""
        print('generate traces, may take some minutes...')

        # Open the output labeled file for writing
        try:
            out_file = h5py.File(labeled_traces_file, "r+")
        except:
            print("Error: can't open HDF5 file '%s' for writing ..." % labeled_traces_file)
            sys.exit(-1)

        # denoise and unscale the traces
        all_traces = AisyTools._unscale(AisyTools._noiseFilter(X_noisy, 0, autoencoder)[:,:,0], Y_clean_o)
        all_traces = all_traces.reshape((all_traces.shape[0], all_traces.shape[1]))
        print("Processing profiling traces...")
        profiling_traces = all_traces[:nt_all]
        print("Profiling: after")
        del out_file['Profiling_traces/traces']
        out_file.create_dataset('Profiling_traces/traces', data=np.array(profiling_traces, dtype=np.int8))

        print("Processing attack traces...")
        attack_traces = all_traces[nt_all:]
        print("Attack: after")
        del out_file['Attack_traces/traces']
        out_file.create_dataset('Attack_traces/traces',  data=np.array(attack_traces, dtype=np.int8))

        out_file.close()

        print("File integrity checking...")
        out_file = h5py.File(labeled_traces_file, 'r')
        profiling = out_file['Profiling_traces/traces']
        print(np.shape(profiling))
        print(np.allclose(profiling[()], profiling_traces))
        attack = out_file['Attack_traces/traces']
        print(np.allclose(attack[()], attack_traces))
        out_file.close()
        print("Done!") 
    @staticmethod
    def addShuffling_variant(traces):
        """Adds a shuffling noise variant.

        The shuffling operation can be resumed by:
        Each round is divided in 4 parts (1 per column), and those 4 parts are shuffled.
        When capturing traces every part has 8 samples (CW clock is 4 times the FPGA,
        and clock generation make processing 1 part in 2 cycles, this is why the traces
        have to be the result of a MA with a window of 4, step 4).Information about the 
        first column, originally in cycle X, is distributed in X+i*8, (i = 0, 1 ,2, 3).
        
        Parameters
        ----------
        traces : List[List[float]]
                The traces. Each sample should be half an atomic operation cycle.
        """
        print('Add shuffling variant...')
        output_traces = np.zeros(np.shape(traces))
        print(np.shape(output_traces))
        for idx in range(len(traces)):
            if(idx % 2000 == 0):
                print(str(idx) + '/' + str(len(traces)))
            trace = traces[idx]
            for i in range(0,len(trace) - (len(trace)%8),8):
                perm = np.random.permutation(4)*2
                perm2 = perm + 1
                res_perm = np.zeros(shape = (len(perm)*2,),dtype=np.int64)
                res_perm[::2] = perm
                res_perm[1::2] = perm2
                output_traces[idx][i:i+8] = trace[i:i+8][res_perm]
        return output_traces


    @staticmethod
    def addShuffling(traces,shuffling_level):
        """Adds a shuffling noise !"""
        print('Add shuffling...')
        output_traces = np.zeros(np.shape(traces))
        print(np.shape(output_traces))
        for idx in range(len(traces)):
            if(idx % 2000 == 0):
                print(str(idx) + '/' + str(len(traces)))
            trace = traces[idx]
            for iter in range(shuffling_level):
                rand = np.random.randint(low=0, high=len(traces[0]), size=(2,))
                temp = trace[rand[0]]
                trace[rand[0]] = trace[rand[1]]
                trace[rand[1]] = temp
            output_traces[idx] = trace
        return output_traces

    @staticmethod
    def addClockJitter(traces,clock_range):
        """Adds a clock jitter noise !"""
        print('Add clock jitters...')
        output_traces = []
        min_trace_length = 100000
        for trace_idx in range(len(traces)):
            if(trace_idx % 2000 == 0):
                print(str(trace_idx) + '/' + str(len(traces)))
            trace = traces[trace_idx]
            point = 0
            new_trace = []
            while point < len(trace)-1:
                new_trace.append(int(trace[point]))
                # generate a random number
                r = random.randint(-clock_range, clock_range)
                # if r < 0, delete r point afterward
                if r <= 0:
                    point += abs(r)
                # if r > 0, add r point afterward
                else:
                    avg_point = int((trace[point] + trace[point+1])/2)
                    for _ in range(r):
                        new_trace.append(avg_point)
                point += 1
            output_traces.append(new_trace)
        return AisyTools.regulateMatrix(output_traces)

    @staticmethod
    def addRandomDelay(traces,delay_amplitude):
        """Adds some random delay noise !"""
        print('Add random delays...')
        output_traces = []
        min_trace_length = 100000
        random_delay_start = 5
        random_delay_end = 3
        for trace_idx in range(len(traces)):
            if(trace_idx % 2000 == 0):
                print(str(trace_idx) + '/' + str(len(traces)))
            trace = traces[trace_idx]
            point = 0
            new_trace = []
            while point < len(trace)-1:
                new_trace.append(int(trace[point]))
                # generate a random number
                r = random.randint(0, 10)
                # 10% probability of adding random delay
                if r > 5:
                    m = random.randint(0, random_delay_start-random_delay_end)
                    num = random.randint(m, m+random_delay_end)
                    #num has distribution: p(num=0 or 5)=1/12 ; P(num=1 or 4)=1/6 ; P(num=2 or 3)=1/4
                    if num > 0:
                        for _ in range(num):
                            new_trace.append(int(trace[point]))
                            new_trace.append(int(trace[point]+delay_amplitude))
                            new_trace.append(int(trace[point+1]))
                point += 1
                # if len(new_trace) > trace_length:
                #    break
            output_traces.append(new_trace)

        return AisyTools.regulateMatrix(output_traces)

    @staticmethod
    def addGaussianNoise(traces,gaussian_noise):
        """Adds some gaussian noise !"""
        print('Add Gaussian noise...',flush=True)
        if (gaussian_noise == 0):
            return traces
        else:
            output_traces = np.zeros(np.shape(traces))
            print(np.shape(output_traces))
            for trace in range(len(traces)):
                if(trace % 5000 == 0):
                    print(str(trace) + '/' + str(len(traces)))
                profile_trace = traces[trace]
                noise = np.random.normal(
                    0, gaussian_noise, size=np.shape(profile_trace))
                output_traces[trace] = profile_trace + noise
            return output_traces

    @staticmethod
    def regulateMatrix(M):
        """A function to make sure the noisy traces has same length (padding zeros)."""
        maxlen = max(len(r) for r in M)
        Z = np.zeros((len(M), maxlen))
        for enu, row in enumerate(M):
            if len(row) <= maxlen:
                Z[enu, :len(row)] += row
            else:
                Z[enu, :] += row[:maxlen]
        return Z
    @staticmethod
    def manual_padding(Y,size):
        """A function to add a padding to each trace in Y."""
        #all traces in Y should be of same length !
        pad_size = size-len(Y[0])
        if(pad_size <0):
            raise Error("Negative pad size. Did the countermeasures decrease the trace sizes ?")
        return np.pad(Y,(0,pad_size),constant_values=(0))[:len(Y)]

    @staticmethod
    def scale(v):
        return (v - v.min()) / (v.max() - v.min())
    @staticmethod
    def _unscale(o, v):  
        return o * (v.max() - v.min()) + v.min()
    @staticmethod
    def _noiseFilter(input, order, model):
        """Applies the autoencoder to the traces to remove noise."""
        filtered_imgs = model.predict(input)
        for i in range(order):
            filtered_imgs = model.predict(filtered_imgs)
        return np.array(filtered_imgs)

    @staticmethod
    def load_only_traces(database_file):
        """Will load only the traces (no metadata) from the data file.

        Parameters
        ----------
        databases_file : str
                The path to the data file.
        
        Returns
        -------
        List[List[float]],List[List[float]]
                The profiling traces and the attack traces.

        """
        if(not os.path.exists(database_file)):
            raise Error(f"File {database_file} not found.")
        try:
            in_file = h5py.File(database_file, "r")
        except:
            print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." %
                database_file)
            sys.exit(-1)
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
        print("X stats ",X_profiling[0])
        return (X_profiling, X_attack)

