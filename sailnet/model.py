# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:00:18 2015

@author: Eric Dodds

original MATLAB code by:
SAILnet: Sparse And Independent Local network _/)
Joel Zylberberg
UC Berkeley Redwood Center for Theoretical Neuroscience
joelz@berkeley.edu
Dec 2010

for work stemming from use of this code, please cite
Zylberberg, Murphy & DeWeese (2011) "A sparse coding model with synaptically
local plasticity and spiking neurons can account for the diverse shapes of V1
simple cell receptive fields", PLoS Computational Biology 7(10).
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import pickle

from . import dictlearner
from . import plotting


class SAILnet(dictlearner.DictLearner):
    """
    Runs SAILnet: Sparse And Independent Local Network
    Currently supports images and spectrograms.
    """

    def __init__(self,
                 data=None,
                 datatype="image",
                 stimshape=None,
                 batch_size=100,
                 niter=50,
                 delay=0,
                 buffer=20,
                 ninput=256,
                 nunits=256,
                 p=0.05,
                 alpha=.1,
                 beta=0.001,
                 gamma=0.01,
                 theta0=0.5,
                 infrate=0.1,
                 moving_avg_rate=0.001,
                 paramfile='SAILnetparams.pickle',
                 pca=None,
                 store_every=1,
                 rfwplots=False,
                 rfw_store_factor=1
                 ):
        """
        Create SAILnet object with given parameters.
        Defaults are as used in Zylberberg et al.

        Args:
        data:               numpy array of data for analysis
        datatype:           (str) type of data
        stimshape:          (array) hape of images/spectrograms (16,16) default
        batch_size:         (int) size of each data batch
        niter:              (int) number of inference time steps
        delay:              (int) spikes before this time step don't count
        buffer:             (int) buffer on image edges
        ninput:             (int) number of inputs, e.g., pixels
        nunits:             (int) number of SAILnet units / 'neurons'
        p:                  (float) target firing rate
        alpha:              (float) learning rate for inhibitory weights
        beta:               (float) learning rate for feedforward weights
        gamma:              (float) learning rate for thresholds
        theta0:             (float) initial value of thresholds
        infrate:            (float) rate parameter for inference
        moving_avg_rate:    (float) rate for averaging stats
        paramfile:          (str) filename for saving parameters
        pca:                (pca) PCA object used to create vector inputs
        store_every:        (int) how many batches between storing stats

        Raises:
        ValueError when datatype is not one of the supported options.
        """

        # Store instance variables
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.infrate = infrate
        self.moving_avg_rate = moving_avg_rate
        self.p = p
        self.batch_size = batch_size
        self.niter = niter
        self.delay = delay
        self.nunits = nunits  # M in original MATLAB code
        self.paramfile = paramfile
        self.pca = pca
        self.plotter = plotting.Plotter(self)
        self.ninput = ninput  # N in original MATLAB code
        self.stimshape = stimshape
        self.store_every = store_every
        self.rfwplots = rfwplots
        self.rfw_store_factor = rfw_store_factor

        self._load_stims(data, datatype, self.stimshape, self.pca)

        self.initialize(theta0)

    def initialize(self, theta0=0.5):
        """Initialize or reset weights, averages, histories."""
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = self.rand_dict()
        self.W = np.zeros((self.nunits, self.nunits))
        self.theta = theta0*np.ones(self.nunits)
        if len(self.errorhist) == 0:
            self.Q0 = self.Q
            self.Q0norm = np.linalg.norm(self.Q0, axis=1)
        # initialize average activity stats
        self.initialize_stats()
        self.corrmatrix_ave = self.p**2
        self.objhistory = []
        self.actshistory = []
        self.dQhistory = []
        self.Qoverlaphistory = []
        self.dQtotalhistory = []
        self.Qtotaloverlaphistory = []
        self.Qsmoothnesshistory = []
        self.L1usagehistory = []
        self.rfWcorrhistory = []
        self.Whistory = []
        self.rfoverlaphistory = []
        self.datahistory = []


    def infer(self, X, infplot=False, savestr=None):
        """
        Simulate LIF neurons to get spike counts.
        Optionally plot mean square reconstruction error vs time.
        X:        input array
        Q:        feedforward weights
        W:        horizontal weights
        theta:    thresholds
        y:        outputs
        """

        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        # initialize values. Note that I've renamed some variables compared to
        # Zylberberg's code. My variable names more closely follow the paper
        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):
            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - 2*self.W.dot(y))

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)])
            y = y.T

            # add spikes to counts (after delay if applicable)
            if t >= self.delay:
                acts = acts + y

            if infplot:
                errors[t] = np.mean(self.compute_errors(acts, X))
                yhist[t] = np.mean(y)

            # reset the internal variables of the spiking units
            u = u*(1-y)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        return acts

    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update lateral weights with Foldiak's rule
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        self.W[self.W < 0] = 0  # force weights to be inhibitory

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

    def run(self, ntrials=25000, datatracking=False, Wtracking=False, rate_decay=1):
        """
        Run SAILnet for ntrials: for each trial, create a random set of image
        patches, present each to the network, and update the network weights
        after each set of batch_size presentations.
        The learning rates are multiplied by rate_decay after each trial.
        """
        # self.actshistory_slice = np.zeros((self.nunits, ntrials//self.store_every))
        # self.dQhistory_slice = np.zeros((self.nunits, ntrials//self.store_every))
        # self.Whistory = np.zeros((self.nunits*(self.nunits-1)/2, ntrials//self.store_every))
        # self.rfWcorrhistory_slice = np.zeros(ntrials//self.store_every)
        # self.datahistory_slice = np.zeros((self.batch_size, ntrials))
        for t in range(ntrials):
            if datatracking == True:
                X, idxs = self.stims.rand_stim(track=datatracking) #(256, 100) matrix, each column a ravelled patch
                self.datahistory.append(idxs)
            else:
                X = self.stims.rand_stim(track=datatracking)

            acts = self.infer(X) #(1536, 100) matrix, each column the activities for every unit

            errors = np.mean(self.compute_errors(acts, X)) #(256, 100) matrix = X - Q^T * acts
            if t % self.store_every == 0:
                corrmatrix = self.store_statistics(acts, errors) #for storing and computing corrmatrix
                self.objhistory.append(self.compute_objective(acts, X))
                self.actshistory.append(np.mean(acts, axis=1))
                if Wtracking == True:
                    mask = self.W[np.triu_indices(self.nunits,k=1)] > 0
                    W = self.W[np.triu_indices(self.nunits,k=1)][mask]
                    rfoverlaps = self.Q.dot(self.Q.T)[np.triu_indices(self.nunits,k=1)][mask]
                if t % self.store_every*self.rfw_store_factor == 0:
                    if Wtracking == True:
                        self.Whistory.append(W)
                        self.rfoverlaphistory.append(rfoverlaps)
                        if len(W) == 0:
                            self.rfWcorrhistory.append(0)
                        else:
                            W = W - W.mean()
                            rfoverlaps - rfoverlaps.mean()
                            rfWcorr = np.dot(W, rfoverlaps)/(np.sqrt(np.sum(W**2))*np.sqrt(np.sum(rfoverlaps**2)))
                            self.rfWcorrhistory.append(rfWcorr)

                #track 'displacement' of each RF from initialization
                self.dQtotalhistory.append(np.linalg.norm(self.Q-self.Q0,axis=1))
                #track overlap of each RF with its initialization
                self.Qtotaloverlaphistory.append(np.einsum('ij,ij->i', self.Q, self.Q0)/self.Q0norm/np.linalg.norm(self.Q, axis=1))
                #track mean smoothness of each RF
                grads = [np.gradient(self.Q[i,:].reshape(self.stimshape)) for i in range(self.nunits)]
                smoothness = []
                for grad_pair in grads:
                    smoothness.append(np.sqrt(grad_pair[0]**2+grad_pair[1]**2).mean())
                self.Qsmoothnesshistory.append(np.array(smoothness))
                #track L1 usage
                L1usage = np.linalg.norm(acts, ord=1, axis=1)
                self.L1usagehistory.append(L1usage)

            else:
                corrmatrix = self.compute_corrmatrix(acts, errors, acts.mean(1)) #computing corrmatrix, no storing

            #save current Q value for dQ tracking
            oldQ = self.Q

            self.learn(X, acts, corrmatrix)

            if t % self.store_every == 0:
                self.dQhistory.append(np.linalg.norm(self.Q-oldQ,axis=1))
                oldQnorm = np.linalg.norm(oldQ, axis=1)
                newQnorm = np.linalg.norm(self.Q, axis=1)
                self.Qoverlaphistory.append(np.einsum('ij,ij->i', self.Q, oldQ)/oldQnorm/newQnorm)



            if t % 50 == 0:
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save()
                    print("Done. Continuing to run...")

            self.adjust_rates(rate_decay)

        self.save()

    def set_dot_inhib(self):
        """Sets each inhibitory weight to the dot product of the corresponding
        units' feedforward weights."""
        self.W = self.Q.dot(self.Q.T)
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        self.W[self.W < 0] = 0  # force weights to be inhibitory

    def visualize(self, cmap='gray'):
        """Display visualizations of network parameters."""
        self.plotter.visualize(cmap=cmap)

    def compute_errors(self, acts, X):
        """Given a batch of data and activities, compute the squared error between
        the generative model and the original data.
        Returns vector of mean squared errors."""
        diffs = X - self.generate_model(acts)
        return np.mean(diffs**2, axis=0)/np.mean(X**2, axis=0)

    def compute_objective(self, acts, X):
        """Compute value of objective/Lagrangian averaged over batch."""
        errorterm = np.mean(self.compute_errors(acts, X))
        rateterm = np.mean((acts-self.p)*self.theta[:, np.newaxis])
        corrWmatrix = (acts-self.p).T.dot(self.W).dot(acts-self.p)
        corrterm = (1/acts.shape[1]**2)*np.trace(corrWmatrix)
        return (errorterm*self.beta/2 + rateterm*self.gamma +
                corrterm*self.alpha)

    def adjust_rates(self, factor):
        """Multiply all the learning rates (alpha, beta, gamma) by factor."""
        self.alpha = factor*self.alpha
        self.beta = factor*self.beta
        self.gamma = factor*self.gamma
        # self.objhistory = factor*self.objhistory

    def sort_dict(self, batch_size=None, allstims=False, plot=True,
                  savestr=None):
        """Sorts the feedforward RFs in order by their usage on a batch.
        By default the batch used for this computation is 10x the usual one,
        since otherwise many dictionary elements tend to have zero activity."""
        batch_size = batch_size or 10*self.batch_size
        if allstims:
            testX = self.stims.data.T
        else:
            testX = self.stims.rand_stim(batch_size)
        # meanacts = np.mean(self.infer(testX),axis=1)
        usage = np.mean(self.infer(testX) != 0, axis=1)
        sorter = np.argsort(usage)
        self.sort(usage, sorter, plot, savestr)
        self.L0acts = usage[sorter]
        return self.L0acts

    def sort(self, usage, sorter, plot=False, savestr=None):
        super().sort(usage, sorter, plot, savestr)
        self.W = self.W[sorter]
        self.W = self.W.T[sorter].T
        self.theta = self.theta[sorter]

    def test_inference(self, infplot=True):
        X = self.stims.rand_stim()
        s = self.infer(X, infplot=infplot)
        print("Final SNR: " + str(self.snr(X, s)))
        return s

    def pairwisedot(self, acts=None):
        pairdots = []
        if acts is None:
            for i in range(self.nunits):
                for j in range(i, self.nunits):
                    pairdots.append(self.Q[i].dot(self.Q[j]))
            return pairdots
        # if acts provided, only return pairwise dot products for
        # coactive units in a batch
        for it in range(self.batch_size):
            itacts = acts[:, it]
            for unit in range(self.nunits):
                if itacts[unit] > 0:
                    for other in range(unit, self.nunits):
                        if itacts[other] > 0:
                            pairdots.append(self.Q[unit].dot(self.Q[other]))
        return pairdots

    def get_param_list(self):
        return {'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'W': self.W,
                'theta': self.theta,
                'p': self.p,
                'nunits': self.nunits,
                'batch_size': self.batch_size,
                'paramfile': self.paramfile,
                'niter': self.niter,
                'infrate': self.infrate}

    def get_histories(self):
        histories = super().get_histories()
        histories['objhistory'] = self.objhistory
        histories['actshistory'] = self.actshistory
        histories['dQhistory'] = self.dQhistory
        histories['Qoverlaphistory'] = self.Qoverlaphistory
        histories['dQtotalhistory'] = self.dQtotalhistory
        histories['Qtotaloverlaphistory'] = self.Qtotaloverlaphistory
        histories['Qsmoothnesshistory'] = self.Qsmoothnesshistory
        histories['L1usagehistory'] = self.L1usagehistory
        histories['rfWcorrhistory'] = self.rfWcorrhistory
        histories['Whistory'] = self.Whistory
        histories['rfoverlaphistory'] = self.rfoverlaphistory
        histories['datahistory'] = self.datahistory
        return histories

    def set_histories(self, histories):
        super().set_histories(histories)
        self.objhistory = histories['objhistory']

    # legacy code for convenience loading old pickle files
    def _old_save(self, filename=None):
        """
        Save parameters to a pickle file, to be picked up later. By default
        we save to the file name stored with the SAILnet instance, but a
        different file can be passed in as the string filename.
        This filename is then saved.
        """
        if filename is None:
            filename = self.paramfile
        histories = (self.L0acts, self.L1acts, self.L2acts, self.L0hist,
                     self.L1hist, self.L2hist, self.corrmatrix_ave,
                     self.errorhist, self.objhistory)
        rates = (self.alpha, self.beta, self.gamma)
        with open(filename, 'wb') as f:
            pickle.dump([self.Q, self.W, self.theta, rates, histories], f)
        self.paramfile = filename

    def _old_load(self, filename=None):
        """Load parameters (e.g., weights) from a previous run from pickle file.
        This pickle file is then associated with this instance of SAILnet."""
        if filename is None:
            filename = self.paramfile
        with open(filename, 'rb') as f: self.Q, param_dict, stat_dict = pickle.load(f) #self.theta, rates, histories = pickle.load(f)
        self.theta = param_dict['theta']
        self.W = param_dict['W']
        self.alpha = param_dict['alpha']
        self.beta = param_dict['beta']
        self.gamma = param_dict['gamma']
        self.infrate = param_dict['infrate']
        self.nunits = param_dict['nunits']
        self.p = param_dict['p']
        self.L0acts = stat_dict['L0acts']
        self.L1acts = stat_dict['L1acts']
        self.L2acts = stat_dict['L2acts']
        self.L0hist = stat_dict['L0hist']
        self.L1hist = stat_dict['L1hist']
        self.L2hist = stat_dict['L2hist']
        self.corrmatrix_ave = stat_dict['corrmatrix_ave']
        self.errorhist = stat_dict['errorhist']
        self.objhistory = stat_dict['objhistory']
        self.actshistory = stat_dict['actshistory']
        self.dQhistory = stat_dict['dQhistory']
        self.Qoverlaphistory = stat_dict['Qoverlaphistory']
        self.dQtotalhistory = stat_dict['dQtotalhistory']
        self.Qtotaloverlaphistory = stat_dict['Qtotaloverlaphistory']
        self.Qsmoothnesshistory = stat_dict['Qsmoothnesshistory']
        self.L1usagehistory = stat_dict['L1usagehistory']
        self.rfWcorrhistory = stat_dict['rfWcorrhistory']
        self.Whistory = stat_dict['Whistory']
        self.rfoverlaphistory = stat_dict['rfoverlaphistory']
        self.datahistory= stat_dict['datahistory']
        # with open(filename, 'rb') as f:
        #     self.Q, self.W, self.theta, rates, histories = pickle.load(f)
        # try:
        #     (self.L0acts, self.L1acts, self.L2acts, self.L0hist,
        #      self.L1hist, self.L2hist, self.corrmatrix_ave,
        #      self.errorhist, self.objhistory) = histories
        # except:
        #     # older files don't have as many statistics saved
        #     try:
        #         try:
        #             self.L0acts, self.L1acts, self.L2acts, self.corrmatrix_ave, self.errorhist, self.objhistory = histories
        #         except ValueError:
        #             print("Loading old file. Some statistics unavailable.")
        #             try:
        #                 self.L0acts, self.L1acts, self.corrmatrix_ave, self.errorhist, self.objhistory = histories
        #                 assert len(self.L1acts.shape) < 2
        #             except AssertionError:
        #                 self.L1acts, self.corrmatrix_ave, self.errorhist, self.objhistory, usage = histories
        #                 self.L0acts = usage
        #     except ValueError:
        #         self.L1acts, self.corrmatrix_ave, self.errorhist, self.objhistory = histories
        # self.alpha, self.beta, self.gamma = rates
        # self.paramfile = filename

