# global preprocess settings
DEBUG = False
EPS = 0.00000001
FEATURES_NUM = 2 #<arg>
RANGE_N = 2000 #<arg> #-1
DEBUG_PLOT_RANGE = 2000
USE_RATIO=True
TEST_NUM = 1000 #<arg>
SPLIT_RATIO = 0.8 #0.7 #0.5 #<arg>
DISTURB_SCALE = 1.0 #255 #0.25 #<arg>
DISTURB_PROBABILITY = 0.01 #0.01 #0.05 0.02 #<arg> 0.01
DISTURB_N_THRESHOLD_MIN = 0.3 #0.2 <arg>
DISTURB_N_THRESHOLD_MAX = 1.0 #0.8 #<arg>
EXCHANGE_NUM = 10 #<arg>100
ERROR_SPLIT_PROBABLITY = 0.5 #<arg>
RANDOM_SEED=42 #<arg>
SCALER_SUFFIX = ['', '_log2', '_sin', '_std', '_std_scaled', '_std_log2', '_std_sin', '_std_log2_sin', '_l1', '_l2']
SCALED_DATA = ['xc_scaled', 'xc_log2', 'xc_sin', 'xc_std', 'xc_std_scaled', 'xc_std_log2', 'xc_std_sin','xc_std_log2_sin', 'xc_l1', 'xc_l2']

# main settings
EPOCHS = 20#5
FILE_SUFFIX = "" #"_std_log2_sin"
# USAD settings
USAD_ALPHA = 0.1#0.6 * l(ae1s, data) + 0.4 * l(ae2ae1s, data)#0.8 * l(ae1s, data) + 0.2 * l(ae2ae1s, data)#0.4 * l(ae1s, data) + 0.6 * l(ae2ae1s, data) #0.1 0.9 #<arg>
USAD_BETA = 0.9
# USAD_BiLSTM_VAE settings
USAD_BLV_ALPHA = 0.1
USAD_BLV_BETA = 0.9