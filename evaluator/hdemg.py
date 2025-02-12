import scipy.io as sio
import numpy as np
import torch
import tkinter as tk
import os
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

class hdEMG:
    def __init__(self, matfilePaths:list=None, mode="1D"):
        if matfilePaths is None:
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory(title="Select the folder containing the .mat files")
            matfilePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')]
        else:
            pass

        self.matfiles = matfilePaths
        self.index = 0
        if mode == "1D":
            self.mode = 1
        elif mode == "3D":
            self.mode = 3
        else:
            raise ValueError("Invalid mode")
        
        self.EMGs, self.raw_emg_data = self._emg_processor()
        self.frameCnt = self.EMGs.shape[0]
        
    def _select_mat_file_gui(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select the .mat file", filetypes=[("MATLAB data files", "*.mat")])
        return file_path

    def _emg_processor(self):
        emg_data = []
        raw_emg_data = []
        for matfile in self.matfiles:
            mat = sio.loadmat(matfile)
            rawEMG = mat['EMGs']
            raw_emg_data.append(rawEMG)
            if self.mode == 1:
                emgReshaped = rawEMG.transpose(0, 2, 1)
                EMGs = emgReshaped.astype(np.float32)
                emg_data.append(EMGs)
            elif self.mode == 3:
                emgPadded = np.pad(rawEMG, ((0, 0), (0, 0), (1, 0)), mode='constant')
                emg_reshaped = emgPadded.reshape(emgPadded.shape[0], emgPadded.shape[1], 13, 5)
                emgExpanded = np.expand_dims(emg_reshaped, axis=1)
                emgTransposed = emgExpanded.transpose(0, 1, 2, 3, 4)
                EMGs = emgTransposed.astype(np.float32)
                emg_data.append(EMGs)
        EMGs = np.concatenate(emg_data, axis=0)
        raw_emg_data = np.concatenate(raw_emg_data, axis=0)
        return EMGs, raw_emg_data
            
    def reset_frame(self):
        self.index = 0
    
    def get_frame(self, index=None):
        if self.mode == 1:
            if index is None:
                EMG = self.EMGs[self.index:self.index+1, :, :]
                self.index += 1
            else:
                self.index = index
                EMG = self.EMGs[index, :]
        elif self.mode == 3:
            if index is None:
                EMG = self.EMGs[self.index:self.index+1, :, :, :]
                self.index += 1
            else:
                self.index = index
                EMG = self.EMGs[index, :, :]
                self.index += 1
        EMG = torch.from_numpy(EMG)
        return EMG
    
    def get_frame_cnt(self):
        return self.frameCnt
    
    def get_mode(self):
        return self.mode
    
    def get_raw_EMG(self):
        return self.raw_emg_data
    
    def get_matfile(self):
        return self.matfile

if __name__ == "__main__":
    hdEMG = hdEMG(mode="3D")
    print(hdEMG.EMGs.shape)
    print(hdEMG.get_frame_cnt())
    print(hdEMG.get_mode())
    print(hdEMG.get_raw_EMG().shape)
    print(hdEMG.get_frame().shape)
    print(type(hdEMG.get_frame(10)))


    
