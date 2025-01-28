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
# brief     : Chipwhisperer tools
# author    :  Lucas D. Meier
# date      : 2023
# copyright :  CSEM

# 
import os
import pickle
from time import sleep
import h5py
import numpy as np
import chipwhisperer as cw


def _initialize_h5(chip_trace,scale,nt,na,ns,fs,info,datasets_root_folder,profiling_key_type,ceiling=True):
    """ Uses the (first) trace from chip_trace to set up a h5 file. 
    
    Parameters
    ----------
    chip_trace : chipwhisperer.Traces
            The chipwhisperer traces to be saved in .h5 (often accessed via proj.traces), containing the wave, plaintexts
            (ciphertexts) and keys.
    scale : bool
            Whether to multiply each measurements by 100.
    nt : int
            How many profiling traces to use to initialize the dataset. Should be left equal to one.
    na : int, default : 0
            How many attack traces to use to initialize the dataset. Should be set to zero unless you know what you are doing.
    ns : int
            How many samples per trace to save.
    fs : int
            Offset, index of first sample in each trace to consider.
    info : str
            Information to add to the dataset filename.
    datasets_root_folder : str
            The path where to save the data.
    profiling_key_type : str
            Can be for example fresh or fixed. To be used in the filename description. Asserts whether the profiling
            traces use random(fresh) or a fixed key(s) for each Trace.
    ceiling : bool, default : True
            Whether to ceil the result to an integer.

    Returns
    -------
    str,str
                The file path / filename.
    """
    assert nt==1 and na == 0 
    raw_data = np.array(chip_trace[0],dtype=object).T
    ptx= list(raw_data[1]) #tranform data representation from cwbytes to integers
    ctx=list(raw_data[2])
    key=list(raw_data[3])
    names= ['plaintext','ciphertext','key']
    np_dict = np.rec.fromarrays([ptx,ctx,key],names=names)
    np_dict=np.array([np_dict])
    trace=raw_data[0][fs:fs+ns]
    if(scale):
        trace=trace*100
    if(ceiling):
        trace=np.ceil(trace*10)
        trace=trace/10
    traces=np.array([trace])
    filename ='chipData_'+profiling_key_type+'_key_'+info+'.h5'
    f = h5py.File(datasets_root_folder+filename, 'w')
    prof = f.create_group("Profiling_traces")
    prof_traces = prof.create_dataset("traces", data=traces[:nt],dtype=np.float64,chunks=True,maxshape=(None,ns))#,dtype=np.float64)
    prof_meta=prof.create_dataset("metadata",data=np_dict,chunks=True,maxshape=(None,16))
    att = f.create_group("Attack_traces")
    att_traces=att.create_dataset("traces",data=traces[nt:nt+na],dtype=np.float64,chunks=True,maxshape=(None,ns))
    att_meta=att.create_dataset("metadata",data=np_dict[nt:nt+na],chunks=True,maxshape=(None,16))
    f.close()
    return datasets_root_folder,filename
def _save_last_trace(chip_trace,file,scale,profiling, ns,fs,ceiling=True,i=0):
    """Saves the last - i trace in the given .h5 file.
    Parameters
    ----------
    chip_trace : chipwhisperer.Traces
            The chipwhisperer traces to be saved in .h5 (often accessed via proj.traces), containing the wave, plaintexts
            (ciphertexts) and keys.
    file : str
            The data file, accessed e.g. via h5py.File(datasets_root_folder+filename, 'a').
    scale : bool
            Whether to multiply each measurements by 100.
    profiling : bool
            Whether to save this trace in the *'Profiling_traces/'* folder or not.
    ns : int
            How many samples per trace to save.
    fs : int
            Offset, index of first sample in each trace to consider.
    ceiling : bool, default : True
            Whether to ceil the result to an integer.
    i : int, default : 0
            Offset to save a specific trace (and not the last one).
    """
    raw_data = np.array(chip_trace[-1-i],dtype=object).T
    
    ptx= list(raw_data[1]) #tranform data representation from cwbytes to integers
    ctx=list(raw_data[2])
    key=list(raw_data[3])
    names= ['plaintext','ciphertext','key']
    np_dict = np.rec.fromarrays([ptx,ctx,key],names=names)
    np_dict=np.array([np_dict])
    
    if(scale):
        trace=raw_data[0]*100
    else:
        trace = raw_data[0]
    if(ceiling):
        trace=np.ceil(trace*10)
        trace=trace/10
    trace=np.array([trace[fs:fs+ns]])
    
    if(profiling):
        path="Profiling_traces/"
    else:
        path="Attack_traces/"
        
    file[path+'traces'].resize((file[path+'traces'].shape[0] + trace.shape[0]),axis=0)
    file[path+'traces'][-trace.shape[0]:] = trace
    file[path+'metadata'].resize((file[path+'metadata'].shape[0]+np_dict.shape[0]),axis=0)
    file[path+'metadata'][-np_dict.shape[0]:] = np_dict

def save_chip_trace_to_h5(proj,nt,na,ns=14000,fs=0,info="",datasets_root_folder="/path/to/Databases/",profiling_key_type="fresh",scale=True,ceiling=True):
    """Saves a chipwhisperer's project traces in a .h5 file. 

    The chipwhisperer's project traces have to contain nt profiling traces followed by na attack traces.
    The operation is done incrementally to avoid high CPU/RAM usage.
    Output file structure:

    * *[Profiling_traces/traces]*
    * *[Profiling_traces/metadata][plaintext]*
    * *[Profiling_traces/metadata][key]*
    * *[Profiling_traces/metadata][ciphertext]* (if exists)
    * *[Attack_traces/traces]*
    * *[Attack_traces/metadata][plaintext]*
    * *[Attack_traces/metadata][key]* (if exists)
    * *[Attack_traces/metadata][ciphertext]* (if exists)

    * *[Profiling_traces/metadata][masks]* (if exists)
    * *[Profiling_traces/metadata][masks]* (if exists)

    Parameters
    ----------
    proj : chipwhisperer.Project
            The project.
    nt : int
            Number of profiling traces.
    na : int
            Number of attack traces.
    ns : int, default : 14000
            Number of samples per trace. 
    fs : int, default : 0
            Offset, index of first sample in each trace to consider.
    info : str, default : ""
            Information string to add to the file.
    datasets_root_folder : str, default : "/path/to/Databases/"
            The path where to save the data.
    profiling_key_type : str, default : fresh
            Can be for example fresh or fixed. To be used in the filename description. Asserts whether the profiling
            traces use random(fresh) or a fixed key(s) for each Trace.
    scale : bool
            Whether to multiply each measurements by 100.
    ceiling : bool, default : True
            Whether to ceil the result to an integer.
    
    """
#     resync_traces = cw.analyzer.preprocessing.ResyncSAD(proj)
#     resync_traces.ref_trace = 0
#     resync_traces.target_window = (50, 150)
#     resync_traces.max_shift = 10
#     resync_analyzer = resync_traces.preprocess()
#     proj = resync_analyzer
    traces = np.array(proj.traces,dtype=object)  #this line speeds up the computation
    path,name = _initialize_h5(traces,info=str(na+nt)+"_trace_"+str(ns)+"_"+info,nt=1,na=0,ns=ns,fs=fs,datasets_root_folder=datasets_root_folder,profiling_key_type=profiling_key_type,scale=scale,ceiling=ceiling)
    sleep(1)
    try:
        file=h5py.File(path+name, 'a')
        if(nt==0):
            k=0
            profiling=False
        else:
            k=1  # 1 trace already stored in .h5 at his creation
            profiling=True
        for i in range(len(traces[k:])):
            if(i + k >nt):
                profiling=False
            _save_last_trace(traces[k:],file=file,profiling=profiling,ns=ns,fs=fs,ceiling=ceiling,i=i,scale=scale)
        file.close()
    except:
        if(os.path.exists(path+name)):
            os.remove(path+name)
        print("Error while saving traces to h5 format.")
    return name

class MockProj:
    def __init__(self,pickle_path_file):
        with open(pickle_path_file, 'rb') as pickle_file:
            self.traces = pickle.load(pickle_file)

class MockProj2:
    def __init__(self,traces):
        self.traces = np.array(traces)


if __name__ == "__main__":
    # Example usage: (may take HOURS to complete)
    path = "/path/"
    project_name = "project.cwp"
    proj = cw.open_project(path + project_name)
    # mock_proj = MockProj2(proj.traces)
    save_chip_trace_to_h5(proj,nt = 50000,na = 0,ns=372,fs=0,info="set1",\
                    datasets_root_folder="/path/to/Databases/",\
                    profiling_key_type="fresh")
