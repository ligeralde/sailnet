from . import dictlearner

import numpy as np
from scipy.optimize import fmin_cg
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Can't import matplotlib.")

class Sparsenet(dictlearner.DictLearner):
    """A sparse dictionary learner based on (Olshausen and Field, 1996)."""

    def __init__(self,
                 data,
                 datatype="image",
                 stimshape=None,
                 batch_size=100,
                 store_every=1000,
                 niter=150,
                 nunits=256,
                 gamma=0.01,
                 lamb=0.3,
                 eta=0.1,
                 adapt=0.96,
                 eta_gd=0.01,
                 cg_maxiter=20,
                 cg_epsilon=1e-8,
                 cg_gtol=1e-4
                 # measure='abs',
                 # infrate=1,,

                 # var_goal=0.1,
                 # sigma=.316,
                 # gain_rate=0.02,
                 # var_eta=0.1,
                 **kwargs):

        #niter: number of inference time steps
        #lamb: sparseness hyperparameter (1/noise_var)
        #infrate: scaling constant for inference
        #measure: sparseness function (logarithmic, bell/exponential, absolute value)
        #var_goal:
        #sigma: scaling constant for activities
        #gains:
        #variances:
        #var_eta:
        #gain_rate:


        self.niter=niter
        self.lamb = lamb
        self.infrate=infrate
        self.measure = measure
        self.var_goal = var_goal
        self.sigma = sigma
        self.gains = np.ones(nunits)
        self.variances = self.var_goal*np.ones(nunits)
        self.var_eta = var_eta
        self.gain_rate = gain_rate
        dictlearner.DictLearner.__init__(self, data, eta, nunits, **kwargs)

    def dSda(self, acts):
        """Returns the derivative of the activity-measuring function."""
        if self.measure == 'log':
            # TODO: Why doesn't this work well? The activities in the denominator may need to be scaled?
            return acts*(1/(1+acts*acts))
        elif self.measure == 'abs':
            return np.sign(acts)
        elif self.measure == 'bell':
            return 2*acts*np.exp(-acts**2)

    def objective(self, acts, X):
        X_hat = self.Q.T.dot(acts)
        error = 0.5*np.linalg.norm(X-X_hat(acts))**2
        if self.measure == 'log':
            return np.sum(np.log(1+(acts/self.sigma)**2)) + error
        elif self.measure == 'abs':
            return np.sum(np.abs(acts/self.sigma)) + error
        elif self.measure == 'bell':
            return np.sum(-np.exp(-(acts/self.sigma)**2)) + error

    def gradient(self, acts, X):
        QX = self.Q.dot(X)
        gramian = self.Q.dot(self.Q.T)
        return QX - gramian.dot(acts) - self.lamb/self.sigma*self.dSda(acts/self.sigma)

    def infer(self, X, infplot=False):
        X_list = [X[:,i] for i in range(np.shape(X)[1])]
        # acts = np.zeros((self.nunits,X.shape[1]))
        # if infplot:
            # costY1 = np.zeros(self.niter)
        # objective = self.objective(X)
        # gradient = self.gradient(X)
        dictnorms = np.sum(self.Q**2,axis=1)
        acts0 = lambda x : self.Q.dot(x)/dictnorms
        acts_final = [fmin_cg(self.objective(acts0(x), x), acts0(x), fprime=self.gradient, args=x, maxiter=self.cg_maxiter) for x in X_list]
        # phi_sq = self.Q.dot(self.Q.T)
        # QX = self.Q.dot(X)
        # for k in range(self.niter):
            # da_dt = QX - phi_sq.dot(acts) - self.lamb/self.sigma*self.dSda(acts/self.sigma)
            # acts = acts+self.infrate*(da_dt)
            # if infplot:
                # costY1[k]=np.mean((X.T-np.dot(acts.T,self.Q))**2)
        # if infplot:
            # plt.plot(costY1)
        # return acts, None, None
        return np.array(acts_final).T

    def learn(self, data, coeffs, normalize=False):
        mse = dictlearner.DictLearner.learn(self, data, coeffs, normalize)
        variances = np.var(coeffs, axis=1)
        self.variances = (1-self.var_eta)*self.variances + self.var_eta*variances
        newgains = self.variances/self.var_goal
        self.gains = self.gains*newgains**self.gain_rate
        normvec = np.sqrt(np.sum(self.Q*self.Q, axis=1))[:,np.newaxis]
        self.Q = self.gains[:,np.newaxis]*self.Q/normvec
        return mse/np.mean(data**2)

    def sort(self, usages, sorter, plot=False, savestr=None):
        self.gains = self.gains[sorter]
        self.variances = self.variances[sorter]
        dictlearner.DictLearner.sort(self, usages, sorter, plot, savestr)

    def set_params(self, params):
        (self.learnrate, self.infrate, self.niter, self.lamb,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate) = params

    def get_param_list(self):
        return (self.learnrate, self.infrate, self.niter, self.lamb,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate)
