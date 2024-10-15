import os
import numpy as np
import time
import matlab.engine  # MATLAB Engine for Python
from utils_v2 import save_ome_tiff

# Start the MATLAB engine
eng = matlab.engine.start_matlab()
# Add the folder containing trickingParfor_v6 to the MATLAB path
eng.addpath(r'C:\Users\Hassan\Desktop\Shukran\DPCMakerWithAbsorption')

# Define parameters
froot = r'S:\Shukran\GH_NanoParticle_092424\logs'
savefolder = r'S:\Shukran\GH_NanoParticle_092424\DPCImages'
Angles = matlab.double([90, 0, 180, 270])  # MATLAB-compatible array for angles

startPos = 41
endPos = 41
startFrame = 1
endFrame = 72
image_size = (1200, 1920)
compression = 'none'
numFrames = endFrame - startFrame + 1

# Loop through positions
for pos in range(startPos, endPos + 1):
    phase_stack = np.zeros((numFrames, image_size[0], image_size[1]), dtype=np.int16)
    absorption_stack = np.zeros((numFrames, image_size[0], image_size[1]), dtype=np.int16)
    time_stack = np.zeros(numFrames)

    print(f"Processing position {pos}...")

    # Loop through frames
    for ii in range(startFrame, endFrame + 1):
        frame = ii
        
        t0=time.perf_counter()
        
        folder_path = os.path.join(froot, f'pos{pos}', f'frame{frame}\\')
        
        if os.path.exists(folder_path):
            # Call the MATLAB function trickingParfor_v6
            Phase_m, Absorption_m, timestamp = eng.trickingParfor_v6(folder_path, Angles, nargout=3)

            # Convert the MATLAB arrays to numpy arrays
            Phase = np.array(Phase_m._data).reshape(image_size[::-1]).T.astype(np.int16)
            Absorption = np.array(Absorption_m._data).reshape(image_size[::-1]).T.astype(np.int16)
            timestamp = float(timestamp)  # Convert timestamp to Python float
            
            # Store the results in the numpy stacks
            phase_stack[ii - startFrame, :, :] = Phase
            absorption_stack[ii - startFrame, :, :] = Absorption
            time_stack[ii - startFrame] = timestamp
        t1=time.perf_counter()
        print(f"Computed pos {pos}, frame {frame} in {t1-t0} sec")
        
    # Define metadata dictionary
    metadata = {
        'time_stack': list(time_stack),
        'axes': 'TYX'
    }

    # File names
    phase_filename = os.path.join(savefolder, f'python_phase_loc_{pos}.ome.tiff')
    absorption_filename = os.path.join(savefolder, f'python_abs_loc_{pos}.ome.tiff')

    save_ome_tiff(phase_filename, phase_stack, compression=compression, metadata=metadata)
    save_ome_tiff(absorption_filename, absorption_stack, compression=compression, metadata=metadata)
    

# Stop the MATLAB engine when done
eng.quit()

print("All positions processed!")
