import concurrent.futures
import nest_asyncio
import asyncio
import numpy as np
import matlab.engine  # MATLAB Engine for Python
import os
import tifffile as tf
import time

# Parameters
froot = r'S:\Shukran\GH_NanoParticle_092424\logs_test'
savefolder = r'S:\Shukran\GH_NanoParticle_092424\DPCImages'
fstart = 'QPM10x_092424_'
Angles = matlab.double([90, 0, 180, 270])
startPos = 27
endPos = 30
startFrame = 1
endFrame = 3
image_size = (1200, 1920)  # height x width
compression = 'zstd'  # Compression method
numFrames = endFrame - startFrame + 1

MAX_CONCURRENT_TASKS = 2  # Limit the number of concurrent positions being processed per frame
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Thread pool executor for CPU-bound tasks (like MATLAB processing)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Adjust number of threads

# Start the MATLAB engine once and reuse it
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\Hassan\Desktop\Shukran\DPCMakerWithAbsorption')

# Preallocate the OME-TIFF file with placeholder planes (zeros)
def preallocate_ome_tiff(filename, num_planes, shape, dtype=np.uint16):
    options = {
        'metadata': {'axes': 'TYX'},  # T for planes, Y for height, X for width
    }

    # Create an empty (zero) array of the desired shape (num_planes, height, width)
    empty_image = np.zeros((num_planes, *shape), dtype=dtype)

    # Write the empty planes into the OME-TIFF file
    with tf.TiffWriter(filename, append=False, bigtiff=True) as tif:
        tif.write(empty_image, **options)

# Function to overwrite a specific plane in the OME-TIFF file with actual data
def overwrite_plane(filename, plane_index, new_image):
    # Open the OME-TIFF file with memory-mapping for efficient plane modification
    tif_memmap = tf.memmap(filename, mode='r+')

    # Overwrite the specific plane (T index)
    tif_memmap[plane_index] = new_image

# Function to handle MATLAB image processing (reuse MATLAB engine)
def matlab_image_processing(folder_path):
    # Call the MATLAB function
    Phase, Absorption, timestamp = eng.trickingParfor_v6(folder_path, Angles, nargout=3)

    # Convert MATLAB output to numpy arrays
    Phase = np.array(Phase._data).reshape(image_size[::-1]).T.astype(np.int16)
    Absorption = np.array(Absorption._data).reshape(image_size[::-1]).T.astype(np.int16)
    
    return Phase, Absorption

async def process_frame_for_position(pos, frame):
    """Process a single frame for a given position and overwrite it in the TIFF."""
    async with semaphore:
        folder_path = os.path.join(froot, f'pos{pos}', f'frame{frame}\\')

        # Polling mechanism: wait until the frame folder exists
        while not os.path.exists(folder_path):
            print(f"Waiting for pos {pos}, frame {frame} to appear...")
            await asyncio.sleep(2)  # Wait for 2 seconds before checking again

        print(f"Processing pos {pos}, frame {frame}...")

        # Offload MATLAB processing to a separate thread but reuse the MATLAB engine
        loop = asyncio.get_event_loop()
        Phase, Absorption = await loop.run_in_executor(executor, matlab_image_processing, folder_path)

        # Overwrite the processed frame in the OME-TIFF files
        phase_filename = os.path.join(savefolder, f'{fstart}ph_loc_{pos}.ome.tiff')
        absorption_filename = os.path.join(savefolder, f'{fstart}abs_loc_{pos}.ome.tiff')

        # Overwrite the specific plane for the phase and absorption images
        overwrite_plane(phase_filename, frame - 1, Phase)  # frame-1 because index is 0-based
        overwrite_plane(absorption_filename, frame - 1, Absorption)

        print(f"Overwritten pos {pos}, frame {frame} in OME-TIFF")

async def process_frame_across_positions(frame):
    """Process the same frame across all positions concurrently."""
    tasks = []
    
    # Process frame for all positions
    for pos in range(startPos, endPos + 1):
        # Process the current frame for this position
        task = process_frame_for_position(pos, frame)
        tasks.append(task)
    
    # Wait for all positions to finish processing the current frame
    await asyncio.gather(*tasks)

async def main():
    """Process all frames sequentially across all positions."""
    # Ensure that all positions have OME-TIFFs created initially with preallocated planes
    for pos in range(startPos, endPos + 1):
        phase_filename = os.path.join(savefolder, f'{fstart}ph_loc_{pos}.ome.tiff')
        absorption_filename = os.path.join(savefolder, f'{fstart}abs_loc_{pos}.ome.tiff')

        # Ensure that the save folder exists
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        # Preallocate OME-TIFFs if they don't exist
        if not os.path.exists(phase_filename):
            preallocate_ome_tiff(phase_filename, numFrames, image_size, dtype=np.int16)
        if not os.path.exists(absorption_filename):
            preallocate_ome_tiff(absorption_filename, numFrames, image_size, dtype=np.int16)

    # Process frames across all positions, frame by frame
    for frame in range(startFrame, endFrame + 1):
        print(f"Processing frame {frame} across all positions...")
        await process_frame_across_positions(frame)

if __name__ == "__main__":
    nest_asyncio.apply()  # To allow asyncio.run in environments like Spyder
    asyncio.run(main())
