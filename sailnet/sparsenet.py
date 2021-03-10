from . import dictlearner

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Can't import matplotlib.")

class Sparsenet(dictlearner.DictLearner):
    """A sparse dictionary learner based on (Olshausen and Field, 1996)."""

    def __init__(self, data, nunits, eta=0.01, measure='abs', infrate=1,
                 niter=200, lamb=0.02, var_goal=0.1, sigma=.3, gain_rate=0.02,
                 var_eta=0.1, **kwargs):

        #niter: number of inference time steps
        #lamb: sparseness hyperparameter
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


    def infer(self, X, infplot=False):
        acts = np.zeros((self.nunits,X.shape[1]))
        if infplot:
            costY1 = np.zeros(self.niter)
        phi_sq = self.Q.dot(self.Q.T)
        QX = self.Q.dot(X)
        for k in range(self.niter):
            da_dt = QX - phi_sq.dot(acts) - self.lamb/self.sigma*self.dSda(acts/self.sigma)
            acts = acts+self.infrate*(da_dt)

            if infplot:
                costY1[k]=np.mean((X.T-np.dot(acts.T,self.Q))**2)
        if infplot:
            plt.plot(costY1)
        return acts, None, None

    def learn(self, data, coeffs, normalize=False):
        mse = dictlearner.DictLearner.learn(self, data, coeffs, normalize)
        variances = np.diag(coeffs.dot(coeffs.T))/self.batch_size
        self.variances = (1-self.var_eta)*self.variances + self.var_eta*variances
        newgains = self.variances/self.var_goal
        self.gains = self.gains*newgains**self.gain_rate
        normQ = np.linalg.norm(self.Q)
        self.Q = self.gains[:,np.newaxis]*self.Q/normQ
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
