import os
import numpy as np
import matlab.engine  # MATLAB Engine for Python
from utils_v2 import save_ome_tiff
from joblib import Parallel, delayed
import time

# Define parameters  
froot = r'S:\Shukran\GH_NanoParticle_092424\logs'
savefolder = r'S:\Shukran\GH_NanoParticle_092424\DPCImages'
fstart = 'QPM10x_092424_'
Angles = matlab.double([90, 0, 180, 270])  # MATLAB-compatible array for angles

startPos = 27
endPos = 216
startFrame = 1
endFrame = 72
image_size = (1200, 1920)  # height x width
compression = 'zstd'
numFrames = endFrame - startFrame + 1

def process_position(pos):
    """ Function to process a single position. """
    # Start a new MATLAB engine inside the worker process
    eng = matlab.engine.start_matlab()
    eng.addpath(r'C:\Users\Hassan\Desktop\Shukran\DPCMakerWithAbsorption')
    
    phase_stack = np.zeros((numFrames, image_size[0], image_size[1]), dtype=np.int16)
    absorption_stack = np.zeros((numFrames, image_size[0], image_size[1]), dtype=np.int16)
    time_stack = np.zeros(numFrames)

    print(f"Processing position {pos}...")

    for ii in range(startFrame, endFrame + 1):
        frame = ii
        
        t0=time.time()
        
        folder_path = os.path.join(froot, f'pos{pos}', f'frame{frame}\\')
        
        if os.path.exists(folder_path):
            # Call the MATLAB function trickingParfor_v6 for each frame
            Phase, Absorption, timestamp = eng.trickingParfor_v6(folder_path, Angles, nargout=3)

            # Convert the MATLAB arrays to numpy arrays and transpose them to swap rows and columns
            Phase = np.array(Phase._data).reshape(image_size[::-1]).T.astype(np.int16)
            Absorption = np.array(Absorption._data).reshape(image_size[::-1]).T.astype(np.int16)
            timestamp = float(timestamp)
            
            # Store the results in the numpy stacks
            phase_stack[ii - startFrame, :, :] = Phase
            absorption_stack[ii - startFrame, :, :] = Absorption
            time_stack[ii - startFrame] = timestamp
            
        t1=time.time()
        print(f"Computed pos {pos}, frame {frame} in {t1-t0} sec")
        
    # Define metadata dictionary
    metadata = {
        'time_stack': list(time_stack),
        'axes': 'TYX'
    }

    # File names
    phase_filename = os.path.join(savefolder, f'{fstart}ph_loc_{pos}.ome.tiff')
    absorption_filename = os.path.join(savefolder, f'{fstart}abs_loc_{pos}.ome.tiff')

    # Save phase image stack using the Python function save_ome_tiff
    save_ome_tiff(phase_filename, phase_stack, compression=compression, metadata=metadata)
    save_ome_tiff(absorption_filename, absorption_stack, compression=compression, metadata=metadata)

    print(f"Finished processing position {pos}")
    eng.quit()

# Joblib to parallelize the processing of positions
# n_jobs=-1 means using all available processors
tst=time.time()
Parallel(n_jobs=16)(delayed(process_position)(pos) for pos in range(startPos, endPos + 1))
tend=time.time()

print(f"All positions processed in {tend-tst}")
