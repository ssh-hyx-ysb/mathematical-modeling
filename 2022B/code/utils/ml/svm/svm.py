from matplotlib import scale
from sklearn.svm import SVC


class svm:
    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)
        self.x_train
        self.y_train
        pass

    def fit(self):
        self.model(self.x_train, self.y_train)
        pass

    def set_datasets():
        pass
