import numpy as np

# training process config
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.
EVAL_BATCH_SIZE = 5
BATCH_SIZE = 10

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2033
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 109
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

MAX_STEPS = 20000

# image config
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 400
IMAGE_DEPTH = 3

# prediction layout config
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Other = [256,256,256]

# labeling config
label = {
    'num_classes' : 2,
    'label_colours' : np.array([Other, Road]),
    'loss_weight' : np.array([
        1.0,
        7.0
        ])
    }


