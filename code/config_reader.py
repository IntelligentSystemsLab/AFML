import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default=r'./experiment.conf',
                    type=str, help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config_path)
data_config = config["Data"]
model_config = config["Model"]
test_config = config["Test"]

TRAINING_PERCENT = float(data_config["training_percent"])
VALIDATION_PERCENT = float(data_config["validation_percent"])
SUPPORT_PERCENT = float(data_config["support_percent"])
CITY = data_config["city"]

LEARNING_RATE = float(model_config["LEARNING_RATE"])
META_LEARNING_RATE = float(model_config["META_LEARNING_RATE"])
REP_LEARNING_RATE = float(model_config["REP_LEARNING_RATE"])
LEN_INPUT = int(model_config["LEN_INPUT"])
BATCH_SIZE = int(model_config["BATCH_SIZE"])
TEST_UPDATE_STEP = int(model_config["TEST_UPDATE_STEP"])
REPTILE_INNER_STEP = int(model_config["reptile_inner_step"])
INPUT_SIZE = int(model_config["input_size"])
HIDDEN_SIZE = int(model_config["hidden_size"])
OUTPUT_SIZE = int(model_config["output_size"])
NUM_LAYERS_GRU = int(model_config["num_layers_gru"])
NUM_LAYERS_LSTM = int(model_config["num_layers_lstm"])
NUM_LAYERS_RNN = int(model_config["num_layers_rnn"])
EPSILON = float(model_config["epsilon"])
BETA = float(model_config["beta"])

RESULT_SAVE_NUMBER_START = int(test_config["result_save_number_start"])
RESULT_SAVE_NUMBER_END = int(test_config["result_save_number_end"])
TEST_FREQUENCY = int(test_config["test_frequency"])
TRAINING_EPOCHS = int(test_config["training_epochs"])
BATCH_INDEX = int(test_config["batch_index"])
REFRESH_FREQUENCY = int(test_config["refresh_frequency"])
TEST_BATCH_INDEX = int(test_config["test_batch_index"])
TEST_BATCH_SIZE = int(test_config['test_batch_size'])
TRAIN_BATCH_SIZE = int(test_config["train_batch_size"])