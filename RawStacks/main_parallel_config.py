from process_location_config import process_location
import time
from joblib import Parallel, delayed
from process_data import process_data
####################################################
# This is for raw phase stacks
####################################################

def main():
    
    start_loc = 80
    end_loc = 216
    locations = range(start_loc, end_loc + 1)

    # Parallel processing using joblib
    Parallel(n_jobs=24)(delayed(process_location)(loc) for loc in locations)

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f'Total processing time: {t1 - t0}')

    # The remaining part of processing (SGR, Normalized Mass, etc.)
    process_data()
    