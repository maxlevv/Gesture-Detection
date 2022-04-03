import numpy as np


class StandardScaler():
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's omit that:
        self.std[self.std == 0] = 0.00001

    def transform(self, X: np.array):
        return (X - self.mean) / self.std

    def inverse_transform(self, X_scaled: np.array):
        return X_scaled * self.std + self.mean

    def to_dict(self):
        # check if scaler has be used
        if 'mean' in self.__dict__.keys():
            return {
                'mean': list(self.mean),
                'std': list(self.std)
            }
        else:
            return {}

    @classmethod
    def from_dict(cls, scaler_dict: dict) -> 'StandardScaler':
        # check if scaler was used in the run
        if scaler_dict:
            scaler = cls()
            scaler.mean = scaler_dict['mean']
            scaler.std = scaler_dict['std']
            return scaler
        else:
            return cls()
