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


    def load_data(self, dset='train', num_col=1):
        if dset == 'train':
            filename = self.train_filename
        elif dset == 'test':
            filename = self.test_filename

        data = np.load(filename)
        X, Y = data[:,:-num_col], data[:,-num_col:]
        
        self.X_original = X
        self.Y = Y


    def transform_single(self, typ): # 'subtract' or 'autoencode'
        if typ == 'subtract':
            self._transform_substact_single()
        elif typ == 'autoencode':
            self._transform_autoencode_single()


    def _transform_substact_single(self):
        X = self.X_original

        _, D = X.shape

        Xa = X[:,:D/2]
        Xb = X[:,D/2:]

        new_X = Xb - Xa

        self.X = new_X


    def _transform_autoencode_single(self):
        X = self.X_original

        _, D = X.shape

        ae = nn.AutoEncoderRelations(D)
        ae.compile()

        new_X = ae.autoencode(X)
        ae.close()

        self.X = new_X


    def transform(self, typ, simple_model=False): # 'subtract' or 'autoencode'
        if typ == 'subtract':
            self._transform_substact()
        elif typ == 'autoencode':
            self._transform_autoencode(simple_model=simple_model)


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


    def _transform_autoencode(self, simple_model=False):
        X = self.X_original

        _, D = X.shape

        ae = nn.AutoEncoderRelations(D/2, simple_model=simple_model)
        ae.compile()

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


def main2():
    dm = DataManager(config.SINGLE_INSTANCE_TRAIN, config.SINGLE_INSTANCE_TEST)

    dm.load_data('train')

    dm.transform_single('subtract')
    dm.save_to(config.SUBTRACT_SINGLE_INSTANCE_TRAIN)
    dm.transform_single('autoencode')
    dm.save_to(config.AUTOENCODE_SINGLE_INSTANCE_TRAIN)

    dm.load_data('test')

    dm.transform_single('subtract')
    dm.save_to(config.SUBTRACT_SINGLE_INSTANCE_TEST)
    dm.transform_single('autoencode')
    dm.save_to(config.AUTOENCODE_SINGLE_INSTANCE_TEST)


def main3():
    dm = DataManager(config.NEW_DATA_TRAIN_NPY, config.NEW_DATA_TEST_NPY)

    simple_model = True

    dm.load_data('train')
    dm.transform('subtract')
    dm.save_to(config.NEW_DATA_TRAIN_SUBTRACT_NPY)
    #dm.transform('autoencode', simple_model=simple_model)
    #dm.save_to(config.NEW_DATA_TRAIN_AUTOENCODE_NPY)

    dm.load_data('test')
    dm.transform('subtract')
    dm.save_to(config.NEW_DATA_TEST_SUBTRACT_NPY)
    #dm.transform('autoencode', simple_model=simple_model)
    #dm.save_to(config.NEW_DATA_TEST_AUTOENCODE_NPY)


if __name__ == '__main__':
    #main()
    #main2()
    main3()
