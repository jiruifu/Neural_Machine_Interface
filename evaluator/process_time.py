"""
The script to compute the real-time processing time of the CNN models.
How to use?
1. Import the correct model class from models.py
"""
import os
import torch
import pickle
from evaluator.dependancy.models import NeuralInterface_1D, NeuralInterface_3D
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import re
import shutil
import time
import numpy as np
import scipy.io as sio
import joblib

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

def extract_info_from_dirname(dir_name):
    """
    Extract the subject info., intensity, muscle, window size, stride, and dimension from the directory name.
    Using the python package re to extract the information.
    """
    pattern = r'uc(\d+)_(\d+)_(\w+)_WS(\d+)-ST(\d+)-(\d+D)'
    match = re.match(pattern, dir_name)
    result = {}

    if match:
        uc_num = match.group(1) 
        result["subject"] = f"uc{uc_num}"     # '1'
        mvc = match.group(2)
        result["intensity"] = mvc         # '10'
        muscle = match.group(3)
        result["muscle"] = muscle      # 'GM'
        window_size = match.group(4)
        result["window_size"] = window_size # '20'
        stride = match.group(5)
        result["step_size"] = stride
        dimension = match.group(6)
        result["dimension"] = dimension
        
        print(f"UC: uc{uc_num}")
        print(f"MVC: {mvc}")
        print(f"Muscle: {muscle}")
        print(f"Window: {window_size}")
        print(f"Stride: {stride}")
        print(f"Dimension: {dimension}")
        return result
    else:
        raise ValueError(f"The directory name {dir_name} does not match the expected pattern")
        
def ask_dir(title):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

def ask_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def copy_best_model(source_dir, target_dir, model_name, param_name):
    """
    Copy best model files to target directory and rename only the copied files.
    Source files remain unchanged.
    
    Args:
        source_dir: Source directory containing the best model files
        target_dir: Target directory to copy files to
        model_name: New name for the copied model file (.pth)
        param_name: New name for the copied parameter file (.pkl)
    """
    if source_dir is None or target_dir is None:
        raise ValueError("The source directory or target directory is not provided")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Keep track of whether we found and copied each type of file
    found_model = False
    found_param = False
    
    for filename in os.listdir(source_dir):
        # Check if the file starts with 'best'
        if filename.startswith('best'):
            source_path = os.path.join(source_dir, filename)
            
            # Handle .pth file (model)
            if filename.endswith('.pth') and not found_model:
                target_path = os.path.join(target_dir, model_name)
                shutil.copy2(source_path, target_path)
                print(f"Copied {filename} to {model_name}")
                found_model = True
                
            # Handle .pkl file (parameters)
            elif filename.endswith('.pkl') and not found_param:
                target_path = os.path.join(target_dir, param_name)
                shutil.copy2(source_path, target_path)
                print(f"Copied {filename} to {param_name}")
                found_param = True
                
    # Optional: Add warning if files weren't found
    if not found_model:
        print("Warning: No best model (.pth) file found")
    if not found_param:
        print("Warning: No best parameter (.pkl) file found")

def organize_files(resultDir=None, dataDir=None, targetDir=None):
    """
    Copy the model and parameter files
    """
    output = dict()

    if resultDir is None:
        resultDir = ask_dir("Please select the result directory")
        resultDirName = os.path.split(resultDir)[-1]
        print(f"Selected result directory: {resultDirName}")
    if dataDir is None:
        dataDir = ask_dir("Please select the directory of the .mat files")
        dataDirName = os.path.split(dataDir)[-1]
        print(f"Selected data directory: {dataDirName}")
    if targetDir is None:
        targetDir = ask_dir("Please select the target directory")
        targetDirName = os.path.split(targetDir)[-1]
        print(f"Selected target directory: {targetDirName}")
    
    info = extract_info_from_dirname(resultDirName)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # rebuilt filename for model and parameter files
    model_name = f"{info['subject']}_{info['intensity']}_{info['muscle']}_WS{info['window_size']}-ST{info['step_size']}-{info['dimension']}_model.pth"
    param_name = f"{info['subject']}_{info['intensity']}_{info['muscle']}_WS{info['window_size']}-ST{info['step_size']}-{info['dimension']}_param.pkl"

    model_dir = os.path.join(resultDir, "models")
    # copy the best model and parameter files to the target directory
    copy_best_model(model_dir, targetDir, model_name, param_name)

    # copy the mat file to the target directory
    matSubDir = f"{info["subject"]}_{info["intensity"]}MVC"
    matFullDir = os.path.join(dataDir, matSubDir)
    matfiles = []
    for i in range(1, 4):
        SG = f"SG{i}"
        mat_name = f"{info['subject']}_{info['intensity']}_{info['muscle']}-{SG}-WS{info['window_size']}-ST{info['step_size']}.mat"
        print(f"mat_name: {mat_name}")
        matSourcePath = os.path.join(matFullDir, mat_name)
        matTargetPath = os.path.join(targetDir, mat_name)
        matfiles.append(matTargetPath)
        if os.path.exists(matSourcePath):
            shutil.copy2(matSourcePath, matTargetPath)
            print(f"Copied {mat_name} to {targetDir}")
        else:
            print(f"The mat file {mat_name} does not exist in {matFullDir}")
    
    output["models"] = {"name": model_name, "path": os.path.join(targetDir, model_name)}
    output["params"] = {"name": param_name, "path": os.path.join(targetDir, param_name)}
    output["matfiles"] = matfiles
    output["targetDir"] = targetDir
    return output, info["dimension"]

def process_time(frames=500):
    """
    Compute the processing time of the model using n frames of the HD-EMG segments.
    Args:
        modelFile: the path of the model file which is a .pth file
        paramFile: the path of the parameter file which is a .pkl file
        matFile: the path of the mat file which is a .mat file
        frames: the number of frames of HD-EMG signal to compute the processing time
    Returns:
        mean_time: the mean processing time of the model
        tHist: the histogram of the processing time
    """
    # # 1-Copy the model, parameter and the mat file to the target directory
    result, mode = organize_files()
    model_path = result["models"]["path"]
    model_name = result["models"]["name"]
    param_path = result["params"]["path"]
    param_name = result["params"]["name"]
    matfiles = result["matfiles"]
    print(f"mode: {mode}")
    print(type(mode))
    for i, matfile in enumerate(matfiles):
        print(f"matfile {i+1}: {matfile}")
    
    # 2-Load the model and parameter
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = torch.load(model_path, weights_only=False)
    model.load_state_dict(torch.load(param_path))
    model.to(device)

    # 3-Load the mat file
    hdemg = hdEMG(matfilePaths=matfiles, mode=mode)

    # 4-Compute the processing time
    if frames == 0:
        frames = hdemg.get_frame_cnt()
    
    tHist = []
    hdemg.reset_frame()
    with torch.no_grad():
        for i in range(frames):
            frame = hdemg.get_frame()
            frame.to(device)
            start_time = time.time()
            _ = model(frame)
            end_time = time.time()
            tHist.append(end_time - start_time)
    shutil.rmtree(result["targetDir"])
    print("---%s seconds ---"% np.mean(tHist))
    return np.mean(tHist), tHist


if __name__ == "__main__":
    mean_time, tHist = process_time()
    print(f"mean_time: {mean_time}")
    print(f"Length of tHist: {len(tHist)}")


