import sys
from os import chdir, mkdir, system, getcwd
from os.path import dirname, exists, join, basename, abspath
from shutil import copy
import configparser


def train_unet(conf_dict):

    config_path, unet_src = conf_dict['config_path'], conf_dict['unet_src']

    # Config file
    config = configparser.RawConfigParser()
    config.readfp(open(r'{}'.format(config_path)))

    conf_dir = dirname(config_path)

    # Get name of the experiment
    name_experiment = config.get('experiment name', 'name')
    nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

    run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

    #create a folder for the results
    result_dir = join(conf_dir, name_experiment)
    print("\n1. Create directory for the results (if not already existing)")
    if exists(result_dir):
        print("Dir already existing")
    else:
        mkdir(result_dir)

    print(result_dir)

    print("copy the configuration file in the results folder")
    copy(config_path, join(result_dir, name_experiment + '_configuration.txt'))

    # Run the experiment
    cwd = getcwd()
    chdir(unet_src)
    
    # Get absolute paths to config path and log file
    abs_config = abspath(join(cwd, config_path))
    print(abs_config)

    log_path = join(result_dir, name_experiment + '_training.nohup')
    abs_log = abspath(join(cwd, log_path))
    print(abs_log)
    exit()    

    cmd = run_GPU + 'nohup python -u ./src/retinaNN_training.py {} > '.format(config_path) + join(result_dir, name_experiment + '_training.nohup')

    if nohup:
        print("\n2. Run the training on GPU with nohup")
        # system(run_GPU +' nohup python -u ./src/retinaNN_training.py ' + config_path + '> ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
        # system(cmd)
	print(cmd)
    else:
        #print "\n2. Run the training on GPU (no nohup)"
        #system(run_GPU +' python ./src/retinaNN_training.py ' + config_path)
	pass
	
    chdir(cwd)
