import numpy as np

import nn
import config


class DataManager:
    def __init__(self, train_filename, test_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename

        self.X_original = None
        self.X = None
        self.Y = None


    def load_data(self, dset='train'):
        if dset == 'train':
            filename = self.train_filename
        elif dset == 'test':
            filename = self.test_filename

        data = np.load(filename)
        X, Y = data[:,:-1], data[:,-1:]
        
        self.X_original = X
        self.Y = Y


    def transform(self, typ): # 'subtract' or 'autoencode'
        if typ == 'subtract':
            self._transform_substact()
        elif typ == 'autoencode':
            self._transform_autoencode()


    def _transform_substact(self):
        X = self.X_original

        _, D = X.shape

        X1 = X[:,:D/2]
        X2 = X[:,D/2:]

        Xa = X1[:,:D/4]
        Xb = X1[:,D/4:]
        Xc = X2[:,:D/4]
        Xd = X2[:,D/4:]
        new_X1 = Xb - Xa
        new_X2 = Xd - Xc

        new_X = np.hstack([new_X1,new_X2])

        self.X = new_X


    def _transform_autoencode(self):
        X = self.X_original

        _, D = X.shape

        ae = nn.AutoEncoderRelations(D/2)

        X1 = X[:,:D/2]
        X2 = X[:,D/2:]

        new_X1 = ae.autoencode(X1)
        new_X2 = ae.autoencode(X2)
        ae.close()

        new_X = np.hstack([new_X1,new_X2])

        self.X = new_X


    def save_to(self, filename):
        data = np.hstack([self.X, self.Y])

        print 'saving to', filename
        print 'dim is', data.shape

        np.save(filename, data)
        self.X = None


def main():
    dm = DataManager(config.DATA_TRAIN, config.DATA_TEST)

    dm.load_data('train')

    dm.transform('subtract')
    dm.save_to(config.SUBTRACT_DATA_TRAIN)
    dm.transform('autoencode')
    dm.save_to(config.AUTOENCODE_DATA_TRAIN)

    dm.load_data('test')

    dm.transform('subtract')
    dm.save_to(config.SUBTRACT_DATA_TEST)
    dm.transform('autoencode')
    dm.save_to(config.AUTOENCODE_DATA_TEST)


if __name__ == '__main__':
    main()
