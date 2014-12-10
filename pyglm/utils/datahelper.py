import cPickle
import os
import numpy as np

def create_unique_results_folder(results_dir):
    """ Create a unique results folder for results.
        The folder will be named based on the current date and time
    """
    from datetime import datetime 
    d = datetime.now()
    unique_name = d.strftime('%Y_%m_%d-%H_%M')
    full_res_dir = os.path.join(results_dir, unique_name)

    # Create the results directory
    if not os.path.exists(full_res_dir):
        print "Creating results directory: %s" % full_res_dir
        os.makedirs(full_res_dir)

    return full_res_dir

def load_data(dataFile):
    """ Load data from the specified file or generate synthetic data if necessary.
    """
    # Load data
    if dataFile is not None:
        if dataFile.endswith('.mat'):
            print "Loading data from %s" % dataFile
            print "WARNING: true parameters for synthetic data " \
                  "will not be loaded properly"

            import scipy.io
            data = scipy.io.loadmat(dataFile,
                                    squeeze_me=True)

            # Scipy IO is a bit weird... ints like 'N' are saved as arrays
            # and the dictionary of parameters doesn't get reloaded as a dictionary
            # but rather as a record array. Do some cleanup here.
            data['N'] = int(data['N'])
            data['T'] = np.float(data['T'])
            
        elif dataFile.endswith('.pkl'):
            print "Loading data from %s" % dataFile
            with open(dataFile,'r') as f:
                data = cPickle.load(f)

                # Print data stats
                N = data['N']
                Ns = np.sum(data['S'])
                T = data['S'].shape[0]
                fr = 1.0/data['dt']
                print "Data has %d neurons, %d spikes, " \
                      "and %d time bins at %.3fHz sample rate" % \
                      (N,Ns,T,fr)

        else:
            raise Exception("Unrecognized file type: %s" % dataFile)

    else:
        raise Exception("Path to data file (.mat or .pkl) must be specified with the -d switch. "
                         "To generate synthetic data, run the test.generate_synth_data script.")
    
    return data

def segment_data(data, (T_start, T_stop)):
    """ Extract a segment of the data
    """
    import copy
    new_data = copy.deepcopy(data)
    
    # Check that T_start and T_stop are within the range of the data
    assert T_start >= 0 and T_start <= data['T']
    assert T_stop >= 0 and T_stop <= data['T']
    assert T_start < T_stop

    # Set the new T's
    new_data['T'] = T_stop - T_start

    # Get indices for start and stop of spike train
    i_start = T_start // data['dt']
    i_stop = T_stop // data['dt']
    new_data['S'] = new_data['S'][i_start:i_stop, :]
    
    # Get indices for start and stop of stim
    i_start = T_start // data['dt_stim']
    i_stop = T_stop // data['dt_stim']
    new_data['stim'] = new_data['stim'][i_start:i_stop, :]
    
    return new_data
