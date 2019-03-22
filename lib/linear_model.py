import numpy as np


class LinearModel():
    def __init__(self, alpha=0.01, n_features=3, action=False):
        self.n_features = n_features
        self.weights = np.zeros(n_features * (1 + int(action)))
        self.alpha = alpha

    def feature_extractor(self, state, action=None):
        """
        This is a sample feature extractor.
        It is simplified as our state is simply represented by an int.
        In practice, we could extract features from an image or implement alphas here for financial data.
        Input: integer state
        Output: array feature vector of dimension 
        """
        res = [state ** k for k in range(self.n_features)]
        if action != None:
            res += [(state - action) ** k for k in range(self.n_features)]
        return np.array(res)

    def predict(self, state, action=None):
        """
        Return predicted value for this model of our state
        Input: integer tate
        Output: float v_hat for the state
        """
        x_sa = self.feature_extractor(state, action)
        return np.dot(x_sa, self.weights)

    def update(self, state, target, action=None):
        """
        Update the weight vector using gradient descent towards the target
        """
        delta = self.alpha * (target - self.predict(state, action)) * self.feature_extractor(state, action)
        self.weights += delta
