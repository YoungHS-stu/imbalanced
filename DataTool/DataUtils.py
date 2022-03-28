import numpy as np


class DataUtils:
    def __init__(self):
        pass

    def shuffle_X_y(self, X, y):
        """
        Shuffle data
        """
        data = np.concatenate((
            X,
            np.reshape(y, (y.shape[0], -1))
        ), axis=1)
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1]
        y = y.astype(int)
        return X, y
