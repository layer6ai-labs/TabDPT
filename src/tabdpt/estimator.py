import torch
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .model import TabDPTModel
from .utils import convert_to_torch_tensor, FAISS, download_model


class TabDPTEstimator(BaseEstimator):
    def __init__(self, path: str = '', mode: str = "cls", inf_batch_size: int = 512, device: str = 'cuda:0', use_flash: bool = True, compile: bool = True):
        self.mode = mode
        self.inf_batch_size = inf_batch_size
        self.device = device
        # automatically download model weight if path is empty
        if path == '':
            path = download_model()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['cfg']['env']['device'] = self.device
        self.model = TabDPTModel.load(model_state=checkpoint['model'], config=checkpoint['cfg'], use_flash=use_flash)
        self.model.eval()
        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile
        assert self.mode in ['cls', 'reg'], "mode must be 'cls' or 'reg'"

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"

        self.imputer = SimpleImputer(strategy='mean')
        X = self.imputer.fit_transform(X)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        self.faiss_knn = FAISS(X)
        self.n_instances, self.n_features = X.shape
        self.X_train = X
        self.y_train = y
        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)
        
    def _prepare_prediction(self, X: np.ndarray):
        check_is_fitted(self)
        self.X_test = self.imputer.transform(X)
        self.X_test = self.scaler.transform(self.X_test)
        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA optionally to reduce the number of features
        if self.n_features > self.max_features:
            U, S, self.V = torch.pca_lowrank(train_x, q=self.max_features)
            train_x = train_x @ self.V
        else:
            self.V = None
        
        test_x = test_x @ self.V if self.V is not None else test_x
        return train_x, train_y, test_x
