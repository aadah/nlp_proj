import numpy as np
import sys
import math
import config

# Loads the training data, randomly splits it into train and test, and resaves it.

all_data_with_labels = np.load(config.TWO_INSTANCE_DATA)
np.random.shuffle(all_data_with_labels)

train_frac = float(sys.argv[1])
if not (train_frac <= 1 and train_frac > 0):
    raise ValueError('Training set fraction must be greater than 0 and less than or equal to 1.')

train_size = math.ceil(all_data_with_labels.shape[0]*train_frac)

print "Train size:", train_size
print "Test size:", all_data_with_labels.shape[0]-train_size


train = all_data_with_labels[:train_size,:]
test = all_data_with_labels[train_size:,:]

train_filename = '%s/split_train.npy' % config.RESOURCES
test_filename = '%s/split_test.npy' % config.RESOURCES

np.save(train_filename, train)
np.save(test_filename, test)

