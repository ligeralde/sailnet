"""
Created on Thu Aug 20 15:48:46 2015
@author: Eric Dodds
(Inference method adapted from code by Jesse Livez ey)
Dictionary learner that uses LCA for inference and gradient descent for learning.
(Intended for static inputs)
"""
import numpy as np
import matplotlib.pyplot as plt
from . import dictlearner
import pickle
try:
    from . import lcaGPU
except ImportError:
    print("Unable to load GPU implementation. Only CPU inference available.")


class LCALearner(dictlearner.DictLearner):

    def __init__(self,
                 data,
                 datatype = "image",
                 stimshape = None,
                 batch_size = 100,
                 store_every = 50,
                 niter=50,
                 nunits=256,
                 learnrate=.001,
                 infrate=.01,
                 theta = 0.022,
                 min_thresh=0.4,
                 adapt=0.95,
                 tolerance = .01,
                 max_iter=4,
                 softthresh = True, #L1 sparsity
                 moving_avg_rate=.001,
                 pca = None,
                 paramfile = None,
                 gpu=False):
        """
        An LCALearner is a dictionary learner (DictLearner) that uses a Locally Competitive Algorithm (LCA) for inference.
        By default the LCALearner optimizes for sparsity as measured by the L0 pseudo-norm of the activities of the units
        (i.e. the usages of the dictionary elements).

        Args:
            data: data presented to LCALearner for estimating with LCA
            nunits: number of units in thresholding circuit = number dictionary elements
            learnrate: rate for mean-squared error part of learning rule
            theta: rate for orthogonality constraint part of learning rule
            batch_size: number of data presented for inference per learning step
            infrate: rate for evolving the dynamical equation in inference (size of each step)
            niter: number of steps in inference (if tolerance is small, chunks of this many iterations are repeated until tolerance is satisfied)
            min_thresh: thresholds are reduced during inference no lower than this value. sometimes called lambda, multiplies sparsity constraint in objective function
            adapt: factor by which thresholds are multipled after each inference step
            tolerance: inference ceases after mean-squared error falls below tolerance
            max_iter: maximum number of chunks of inference (each chunk having niter iterations)
            softthresh: if True, optimize for L1-sparsity
            datatype: image or spectro
            pca: pca object for inverse-transforming data if used in PC representation
            stimshape: original shape of data (e.g., before unrolling and PCA)
            paramfile: a pickle file with dictionary and error history is stored here
            gpu: whether or not to use the GPU implementation of
        """

        learnrate = learnrate or 1./batch_size

        self.infrate = infrate
        self.niter = niter
        self.min_thresh = min_thresh
        self.adapt = adapt
        self.softthresh = softthresh
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.gpu = gpu
        self.meanacts = np.zeros(nunits)
        self.stimshape = stimshape

        super().__init__(data, learnrate, nunits, paramfile = paramfile, theta=theta, moving_avg_rate=moving_avg_rate,
                            stimshape=stimshape, datatype=datatype, batch_size=batch_size, pca=pca, store_every=store_every)


    def show_oriented_dict(self, batch_size=None, *args, **kwargs):
        """Display tiled dictionary as in DictLearn.show_dict(), but with elements inverted
        if their activities tend to be negative."""
        if batch_size is None:
            means = self.meanacts
        else:
            if batch_size == 'all':
                X = self.stims.data.T
            else:
                X = self.stims.rand_stim(batch_size)
            means = np.mean(self.infer(X)[0],axis=1)
        toflip = means < 0
        realQ = self.Q
        self.Q[toflip] = -self.Q[toflip]
        result = self.show_dict(*args, **kwargs)
        self.Q = realQ
        return result

    def infer_cpu(self, X, infplot=False, tolerance=None, max_iter = None):
        """Infer sparse approximation to given data X using this LCALearner's
        current dictionary. Returns coefficients of sparse approximation.
        Optionally plot reconstruction error vs iteration number.
        The instance variable niter determines for how many iterations to evaluate
        the dynamical equations. Repeat this many iterations until the mean-squared error
        is less than the given tolerance or until max_iter repeats."""
        tolerance = tolerance or self.tolerance
        max_iter = max_iter or self.max_iter
        ndict = self.Q.shape[0]

        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)

        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T)
        for i in range(c.shape[0]):
            c[i,i] = 0

        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T

        # initialize threshold values, one for each stimulus, based on average response magnitude
        thresh = np.absolute(b).mean(1)
        thresh = np.array([np.max([th, self.min_thresh]) for th in thresh])

        if infplot:
            errors = np.zeros(self.niter)
            allerrors = np.array([])

        error = tolerance+1
        outer_k = 0
        while(error>tolerance and ((max_iter is None) or outer_k<max_iter)):
            for kk in range(self.niter):
                # ci is the competition term in the dynamical equation
                ci[:] = s.dot(c)
                u[:] = self.infrate*(b-ci) + (1.-self.infrate)*u
                if np.max(np.isnan(u)):
                    raise ValueError("Internal variable blew up at iteration " + str(kk))
                if self.softthresh:
                    s[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis])
                else:
                    s[:] = u
                    s[np.absolute(s) < thresh[:,np.newaxis]] = 0

                if infplot:
                    errors[kk] = np.mean(self.compute_errors(s.T,X))

                thresh = self.adapt*thresh
                thresh[thresh<self.min_thresh] = self.min_thresh

            error = np.mean((X.T - s.dot(self.Q))**2)
            outer_k = outer_k+1
            if infplot:
                allerrors = np.concatenate((allerrors,errors))

        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(allerrors)
            return s.T, errors
        return s.T, u.T, thresh

    def infer(self, X, infplot=False, tolerance=None, max_iter = None):
        if self.gpu:
            # right now there is no support for multiple blocks of iterations, stopping after error crosses threshold, or plots monitoring inference
            return LCAonGPU.infer(self, X.T)
        else:
            return self.infer_cpu(X, infplot, tolerance, max_iter)

    def test_inference(self, niter=None):
        temp = self.niter
        self.niter = niter or self.niter
        X = self.stims.rand_stim()
        s = self.infer(X, infplot=True)[0]
        self.niter = temp
        print("Final SNR: " + str(self.snr(X,s)))
        return s

    def adjust_rates(self, factor):
        """Multiply the learning rate by the given factor."""
        self.learnrate = factor*self.learnrate
        #self.infrate = self.infrate*factor # this is bad, but NC seems to have done it

    def set_params(self, params):
        (self.learnrate, self.theta, self.min_thresh, self.infrate, self.nunits,
                self.niter, self.adapt, self.max_iter, self.tolerance) = params

    def get_param_list(self):
        return {'learnrate': self.learnrate,
                'theta': self.theta,
                'min_thresh': self.min_thresh,
                'infrate': self.infrate,
                'niter': self.niter,
                'adapt': self.adapt,
                'nunits': self.nunits,
                'max_iter': self.max_iter,
                'tolerance': self.tolerance,
                }
                # 'Q0': self.Q0,
                # 'Q0norm': self.Q0norm}
    def set_histories(self, histories):
        super().set_histories(histories)

    def _old_load(self, filename=None):
        """Load parameters (e.g., weights) from a previous run from pickle file.
        This pickle file is then associated with this instance of SAILnet."""
        if filename is None:
            filename = self.paramfile
        with open(filename, 'rb') as f: self.Q, param_dict, stat_dict = pickle.load(f) #self.theta, rates, histories = pickle.load(f)
        self.theta = param_dict['theta']
        # self.W = param_dict['W']
        # self.alpha = param_dict['alpha']
        # self.beta = param_dict['beta']
        # self.gamma = param_dict['gamma']
        self.infrate = param_dict['infrate']
        self.nunits = param_dict['nunits']
        # self.p = param_dict['p']
        self.L0acts = stat_dict['L0acts']
        self.L1acts = stat_dict['L1acts']
        self.L2acts = stat_dict['L2acts']
        self.L0hist = stat_dict['L0hist']
        self.L1hist = stat_dict['L1hist']
        self.L2hist = stat_dict['L2hist']
        # self.corrmatrix_ave = stat_dict['corrmatrix_ave']
        self.errorhist = stat_dict['errorhist']
        # self.objhistory = stat_dict['objhistory']
        # self.actshistory = stat_dict['actshistory']
        self.Qhistory = stat_dict['Qhistory']
        self.dQhistory = stat_dict['dQhistory']
        self.Qoverlaphistory = stat_dict['Qoverlaphistory']
        self.dQtotalhistory = stat_dict['dQtotalhistory']
        self.Qtotaloverlaphistory = stat_dict['Qtotaloverlaphistory']
        self.Qsmoothnesshistory = stat_dict['Qsmoothnesshistory']
        self.L1usagehistory = stat_dict['L1usagehistory']
        # self.rfWcorrhistory = stat_dict['rfWcorrhistory']
        # self.Whistory = stat_dict['Whistory']
        # self.rfoverlaphistory = stat_dict['rfoverlaphistory']
        # self.datahistory= stat_dict['datahistory']

    def load(self, filename=None):
        """Loads the parameters that were saved. For older files when I saved less, loads what I saved then."""
        self.paramfile = filename
        try:
            super().load(filename)
            return
        except:
            # This is all for backwards-compatibility with files I saved before I started saving as many statistics
            try:
                with open(filename, 'rb') as f:
                    self.Q, params, histories = pickle.load(f)
                self.learnrate, self.theta, self.min_thresh, self.infrate, self.niter, self.adapt, self.max_iter, self.tolerance = params
                try:
                    self.errorhist, self.L0acts, self.L0hist, self.L1acts, self.L1hist, self.corrmatrix_ave = histories
                except ValueError:
                    print('Loading old file. Correlation matrix not available.')
                    try:
                        self.errorhist, self.L0acts, self.L0hist, self.L1acts, self.L1hist = histories
                    except ValueError:
                        print('Loading old file. Activity histories not available.')
                        try:
                            self.errorhist, self.L0acts, self.L1acts = histories
                        except ValueError:
                            print("Loading old file. Moving average activities not available.")
                            self.errorhist = histories
            except ValueError:
                print("Loading very old file. Only dictionary and error history available.")
                with open(filename, 'rb') as f:
                    self.Q, self.errorhist = pickle.load(f)
