import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def CST_plotter(data, ref, step_size, window_size, label_ref:str, lebel_data:str,  title:str=None,desired_window_size=400, 
                sampling_rate=2048, xlabel:str=None, ylabel:str=None, dpi:int=300,legend:bool=False, Outputformat:str="png", 
                fname:str="CST_Plot", save_dir:str="figures", size:tuple=(3, 2)):
    
    
    def calculate_raw_duration(n_samples, window_size, step_size, sampling_freq):
        """
        Calculate the duration of the original raw data from segmented data parameters.
        
        Parameters:
        -----------
        n_samples : int
            Number of segments/windows in the data
        window_size : int
            Size of each window in samples
        step_size : int
            Number of samples between the start of consecutive windows
        sampling_freq : float
            Sampling frequency in Hz
        
        Returns:
        --------
        duration_seconds : float
            Duration of the original data in seconds
        """
        # Calculate number of samples in raw data
        # For n windows with step_size, we need: (n-1)*step_size + window_size samples
        raw_samples = (n_samples - 1) * step_size + window_size
        
        # Calculate duration in seconds
        duration_seconds = raw_samples / sampling_freq
        
        return duration_seconds
    def smooth_signal(x, desired_window_size, sampling_rate, step_size, window_size):
        """
        Smooth a signal using a Hanning window.
        
        Parameters:
        x : array_like
            Input signal
        window_size : int
            Size of the smoothing window (in ms)
            
        Returns:
        smoothed : ndarray
            Smoothed signal
        """
        def get_samples_for_period_ms(time_period_ms, sampling_rate, step_size, window_size):
            """
            Calculate number of samples needed for a specific time period in milliseconds in a segmented signal.
            
            Parameters:
            -----------
            time_period_ms : float
                Desired time period in milliseconds
            sampling_rate : float
                Sampling rate in Hz
            step_size : int
                Step size between windows in samples
            window_size : int
                Size of the analysis window in samples
                
            Returns:
            --------
            n_samples : int
                Number of samples needed (rounded down to nearest integer)
            actual_time_ms : float
                Actual time period that will be covered by the returned number of samples in milliseconds
            """
            # Convert time period from milliseconds to seconds
            time_period = time_period_ms / 1000.0
            
            # Calculate how many samples we need for the desired time period
            n_samples_float = (time_period * sampling_rate - window_size/2) / step_size + 1
            
            # Round down to nearest integer
            n_samples = int(np.floor(n_samples_float))
            
            # Calculate actual time period that will be covered in milliseconds
            actual_time = ((n_samples - 1) * step_size + window_size/2) / sampling_rate
            actual_time_ms = actual_time * 1000.0
            
            return n_samples, actual_time_ms
        # Make sure window size is odd
        if desired_window_size % 2 == 0:
            desired_window_size += 1
        
        actual_window, actual_duration = get_samples_for_period_ms(desired_window_size, sampling_rate=sampling_rate, step_size=step_size, window_size=window_size)
        
        # Create Hanning window
        window = np.hanning(actual_window)
        
        # Normalize window
        window = window / window.sum()
        
        # Convolve signal with window
        smoothed = np.convolve(x, window, mode='valid')
        
        # Handle edges (optional)
        # Pad the result to match input size
        pad_size = (len(x) - len(smoothed)) // 2
        smoothed = np.pad(smoothed, (pad_size, pad_size), mode='edge')
        index = np.arange(len(smoothed))
        
        return smoothed, index
    
    
    smoothed_data, idx = smooth_signal(data, desired_window_size=desired_window_size, sampling_rate=sampling_rate, step_size=step_size, window_size=window_size)
    smoothed_ref, idx_ref = smooth_signal(ref, desired_window_size=desired_window_size, sampling_rate=sampling_rate, step_size=step_size, window_size=window_size)
    
    # Calculate time points for x-axis
    n_samples = len(idx)  # number of points in the smoothed signal
    duration = calculate_raw_duration(n_samples, window_size, step_size, sampling_rate)
    time_points = np.linspace(0, duration, len(idx))  # create evenly spaced time points
    
    sns.set_style("ticks")
    sns.set_context("paper")

    plt.figure(figsize=size)
    sns.lineplot(x=time_points, y=smoothed_ref, label=label_ref)
    sns.lineplot(x=time_points, y=smoothed_data, label=lebel_data)
    
    if legend:
        plt.legend(labels=[label_ref, lebel_data])
    else:
        plt.legend([],[], frameon=False)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("Time (s)")  # Default to showing time in seconds
    
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("")
    
    if title is not None:
        plt.title(title)
    
    fName = f"{fname}.{Outputformat}"
    plt.tight_layout()
    save_path = os.path.join(save_dir, fName)
    plt.savefig(save_path, dpi=dpi)
    plt.show()
    plt.close()

if __name__ == "__main__":
    result = joblib.load("test_result.pkl")
    print(result['meta_result'].keys())
    cst_cnn = result['meta_result']['CST_of_best_model']
    cst_label = result['meta_result']['CST_labels_of_best_model']
    ws = 20
    st = 10
    fs = 2048
    CST_plotter(data=cst_cnn, 
                ref=cst_label, 
                window_size=ws, 
                step_size=st, 
                sampling_rate=fs,
                desired_window_size=400,
                label_ref="Label",
                lebel_data="CNN",
                title="CST_Plot",
                fname="CST_Plot",
                save_dir="figures",
                size=(3, 2))
