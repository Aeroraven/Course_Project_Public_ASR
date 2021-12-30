# Work 2 For Acoustic Signal Recognition


import asr_mfcc as mfcc  # MFCC from Homework 1
import os
import random
import tqdm  # Progress Bars
import sys
import warnings
import time
import scipy.io as sio
import python_speech_features as psf
import scipy.io.wavfile as wav
import multiprocessing as mpr

# This suppresses warnings from Numpy which disallows log(0)
# log(0) will return -Inf in the program
# warnings.filterwarnings("ignore")

# Feature Parameters
HMM_INITIAL_VALUE = 0.4  # Constant used for initializing transitional probabilities
MFCC_DIMENSIONS = 39  # Feature dimensions used to calculate MFCCs
MEL_FILTERS = 26  # Mel filters used to generate features
TRAINING_ITERATIONS = 1  # Default internal training iterations (Do not change this constant)
PI = 3.141592653589793  # Just PI
LOG_2_PI = 1.83787706640934548356  # Constant to reduce redundant LOG(2*PI)
LOG2_2_PI = 2.651496129472319

VAL_RATIO = 0.95
TRAINING_EXT_ITERATIONS = 20 # Training iteration (This is configurable)
MODEL_CHECKPOINT = "."

# Computation Parameters
# GPU overhead consumes more time if the the matrix operations are tiny
# CuPy are required to enable the option USING_CUDA
USING_CUDA = False  # Use GPU-accelerated computation, defaulting to False
USING_MULTITHREAD = True  # Use thread pool to accelerate training
USING_MULTIPROCESS = True
USING_MFCC_LIBRARY = False  # Use library to calculate MFCC
USING_CACHED_MFCC = False # Use cached MFCC file

# Running Test
USING_TEST_MODE = True
USING_TEST_ITERATION = 20
USING_TEST_PATH = "./"

# Importing computation framework
if USING_CUDA:
    import cupy as np  # Replace Numpy by CuPy
else:
    import numpy as np  # Use CPU computation
import numpy

if USING_MULTITHREAD:
    from concurrent.futures import ThreadPoolExecutor, as_completed

# Global Variables
train_data_loaders = []
models = []
iter = 0
cur_pbar = None
result_a = []
cur_prob = []


class Misc:
    """Miscellaneous Class
    """

    @staticmethod
    def print(stuff):
        """
        Print with an immediate flush
        :param stuff:
        """
        print(stuff, flush=True)

    @staticmethod
    def to_tensor(numpy_tensor):
        """
        Convert a numpy ndarray to a CuPy array
        :param numpy_tensor: (ndarray) Numpy array to be converted
        :return:(cuarray) CuPy array
        """
        if USING_CUDA:
            return np.asarray(numpy_tensor)
        else:
            return numpy_tensor

    @staticmethod
    def to_numpy(cuda_tensor):
        """
        Convert a CuPy tensor to NumPy array
        :param cuda_tensor: (cuarray) tensor to be converted
        :return: (ndarray) NumPy array
        """
        if USING_CUDA:
            return np.asnumpy(cuda_tensor)
        else:
            return cuda_tensor

    @staticmethod
    def check_nan(tensor):
        if numpy.isnan(tensor).any() or numpy.isinf(tensor).any():
            Misc.print("NaN or +Inf !!!")
            raise Exception("NaN or +Inf encountered")

    @staticmethod
    def check_min(matrix, minval):
        if numpy.min(matrix) < minval:
            Misc.print("Min < " + str(minval))
            raise Exception("Value too small")

    @staticmethod
    def check_zero_division(value):
        if np.min(np.abs(value)) < Math.eps():
            Misc.print("Divide by zero!!!")
            raise Exception("Divide by zero")

    @staticmethod
    def check_log_prob(value):
        if np.min(value) > 0.:
            Misc.print("Invalid Prob")
            raise Exception("Invalid Prob" + str(np.min(value)))


class Math:
    """Miscellaneous class for math calculation
    """

    @staticmethod
    def eps():
        """
        Get the epsilon of double scalars
        :return:
        """
        return np.finfo('float').eps

    @staticmethod
    def log_normal_distribution(observed, average, standard_variance):
        """
        Logarithmic probability of the normal distribution
        :param observed: (float) Observed value
        :param average: (float) Central value of the distribution (average)
        :param standard_variance: (float) Standard deviation of the distribution
        :return: (float) Logarithmic probability
        """
        return -1 / 2 * Math.log(2 * PI * standard_variance * standard_variance) + \
               Math.log((observed - average) ** 2 / 2 * (standard_variance ** 2))

    @staticmethod
    def log_normal_distribution_seq(observed, average, standard_variance, dimension):
        """
        Logarithmic probability of the normal distribution for a observation sequence
        :param observed: (ndarray) Observed sequence
        :param average: (float) Central value of the distribution (average)
        :param standard_variance: (float) Standard deviation of the distribution
        :return: (ndarray) Array of element-wise logarithmic probability
        """
        var = Math.pow(standard_variance, 2)
        Misc.check_zero_division(var)
        return -1 / 2 * (Math.scalar_log(2 * PI) * dimension + 2 * np.sum(Math.log(standard_variance)) +
                         np.sum(np.true_divide(Math.pow(observed - average, 2), var)))

    @staticmethod
    def log_normal_distribution_seq_var(observed, average, variance, dimension):
        """
        Logarithmic probability of the normal distribution for a observation sequence
        :param observed: (ndarray) Observed sequence
        :param average: (float) Central value of the distribution (average)
        :param variance: (float) Variance deviation of the distribution
        :return: (ndarray) Array of element-wise logarithmic probability
        """
        t = np.sum(Math.log(variance))
        x = np.sum(np.true_divide(Math.pow(observed - average, 2), variance))
        v = -1 / 2 * (LOG_2_PI * dimension + t + x )
        return v

    @staticmethod
    def pow(base, exp):
        """
        Power of a figure or a sequence. (Element-wise)
        :param base: (ndarray) Base
        :param exp: (int/flat) exponent
        :return: (ndarray) result
        """
        return np.power(base, exp)

    @staticmethod
    def inf():
        """
        Return infinity
        :return: infinity
        """
        if USING_CUDA:
            return np.inf
        else:
            return numpy.inf

    @staticmethod
    def scalar_log(scalar):
        """
        Logarithmic calculation for scalars
        :param scalar: (int/float) Input
        :return: (int/float) Logarithmic input
        """
        return Math.log(scalar)

    @staticmethod
    def log(num):
        with np.errstate(divide='ignore'):
            return np.log(num)

    @staticmethod
    def exp(num):
        return np.exp(num)


class MFCCProcessing:
    """ Utility of MFCC processing
    """

    def __init__(self):
        self.cache = dict()
        self.square_cache = dict()

    def get_squared_feature(self, filename, mel_filters=MEL_FILTERS):
        """
        Get the squared MFCC features. The function is used for reducing redundant calculations
        :param filename: wave file or HTK MFCC file path
        :param mel_filters: number of Mel filters
        :return: (ndarray) squared features
        """
        if self.cache.get(filename) is None:
            self.get_feature(filename, mel_filters)
        return self.square_cache[filename]

    def get_feature(self, filename, mel_filters=MEL_FILTERS):
        """
        Get the MFCC features. Features will be loaded from cache if it's loaded before
        :param filename: wave file or HTK MFCC file path
        :param mel_filters: number of Mel filters
        :return: (ndarray) features
        """
        if self.cache.get(filename) is not None:
            return self.cache[filename]
        if USING_CACHED_MFCC:
            with open(filename, 'rb') as fid:
                nsample = np.fromfile(fid, dtype=np.dtype('>i4'), count=1)
                samp_period = np.fromfile(fid, dtype=np.dtype('>i4'), count=1) * 1e-7
                samp_size = np.fromfile(fid, dtype=np.dtype('>i2'), count=1)
                dm = 0.25 * samp_size
                param_kind = np.fromfile(fid, dtype=np.dtype('>i2'), count=1)
                feature = np.fromfile(fid, dtype=np.dtype('>f')).reshape((nsample[0], int(dm[0])))
                self.cache[filename] = feature
                self.square_cache[filename] = Math.pow(feature, 2)
                return feature
        if not USING_MFCC_LIBRARY:
            sampling_freq, audio = mfcc.audio_read(filename, True)
            audio, audio_original = mfcc.audio_preemphasis(audio)
            frame, frame_count, frame_size = mfcc.audio_windowing(audio, sampling_freq,0)
            frame_stft, fft_bins = mfcc.audio_fft(frame, frame_count,512)
            mel_filter_banks = mfcc.get_mel_filter_banks(sampling_freq, fft_bins, mel_filters, 0, sampling_freq // 2)
            feat, energy = mfcc.get_feat_and_energy(frame_stft, mel_filter_banks)
            log_feat = Misc.to_numpy(np.log(Misc.to_tensor(feat)))
            dct_result = mfcc.audio_dct(log_feat)
            dct_result = mfcc.audio_lifter(dct_result)
            dct_result[:, 0] = energy
        else:
            (rate, sig) = wav.read(filename)
            dct_result = psf.mfcc(sig, rate)

        feat_order_1 = mfcc.calculate_delta_feature(dct_result, 1)
        feat_order_2 = mfcc.calculate_delta_feature(feat_order_1, 1)
        feature_vector = mfcc.feature_concatenate_normalisation((dct_result, feat_order_1, feat_order_2))
        self.cache[filename] = feature_vector
        self.square_cache[filename] = Math.pow(feature_vector, 2)
        return feature_vector


class MFCCDataSetDivider:
    """ Read audio dataset and split the dataset into training set and test set
    """

    def __init__(self, directory, label, val_split=0.5):
        self.part = os.listdir(directory)
        # random.shuffle(self.part)
        self.train_set = self.part[int(len(self.part) * val_split):]
        self.val_set = self.part[:int(len(self.part) * val_split)]
        self.train_data = []
        self.val_data = []
        self.label = label
        for i in self.train_set:
            self.__load_data__(directory, i, self.train_data)
        for i in self.val_set:
            self.__load_data__(directory, i, self.val_data)

    def __load_data__(self, directory, label, recipient):
        """
        Load dataset configurations
        :param directory: root directory of the dataset
        :param label: sub directory of the dataset
        :param recipient: (ndarray) reference to the recipient array
        """
        filelist = os.listdir(directory + "/" + label)
        for i in filelist:
            if i.startswith(self.label):
                sdata = dict()
                sdata['label'] = self.label
                sdata['path'] = directory + '/' + label + '/' + i
                sdata['mfcc'] = 'mfcc' + '/' + label + '/' + i[:-3] + 'mfc'
                recipient.append(sdata)

    def get_train_data_loader(self):
        """
        Get the training data loader
        :return: (MFCCDataLoader) training data loader
        """
        return MFCCDataLoader(self.train_data)

    def get_val_data_loader(self):
        """
        Get the validation data loader
        :return: (MFCCDataLoader) validation data loader
        """
        return MFCCDataLoader(self.val_data)


class DataLoader:
    """ Base class for data loader
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class MFCCDataLoader(DataLoader, MFCCProcessing):
    """ Data loader for loading MFCC features
    """

    def __init__(self, data):
        self.cache = dict()
        self.square_cache = dict()
        self.use_mfcc = USING_CACHED_MFCC
        super().__init__(data)

    def __getitem__(self, item):
        return self.get_feature(item)

    def get_label(self, item):
        """
        Get the label associated with the data
        :param item: index of the data entry
        :return: data label
        """
        return self.data[item]['label']

    def get_feature(self, index, mel_filters=26):
        """
        Get the feature associated with the data entry
        :param index:  index of the data entry
        :param mel_filters: number of MFCC features to be used
        :return: MFCC feature
        """
        if self.use_mfcc:
            return super().get_feature(self.data[index]['mfcc'])
        return super().get_feature(self.data[index]['path'])

    def get_squared_feature(self, index, mel_filters=26):
        """
        Get the squared feature associated with the data entry
        :param index:  index of the data entry
        :param mel_filters: number of MFCC features to be used
        :return: squared MFCC feature
        """
        if self.use_mfcc:
            return super().get_squared_feature(self.data[index]['mfcc'])
        return super().get_squared_feature(self.data[index]['path'])

    def __len__(self):
        return len(self.data)

    def preload(self):
        """
        Preload the dataset
        """
        for i in tqdm.tqdm(range(len(self)), desc="Preloading Dataset", file=sys.stdout):
            self.get_feature(i)

    def join(self, data_loader):
        """
        Add data entries from another data loader
        :param data_loader: source data loader
        """
        self.data.extend(data_loader.data)


class HiddenMarkovModelCheckpoint:
    """ Checkpoint for a trained HMM model
    """

    def __init__(self, avg, var, mat):
        self.avg = avg.tolist()
        self.var = var.tolist()
        self.mat = mat.tolist()

    def save(self, path):
        """
        Save the model
        :param path: checkpoint file
        """
        save_data = dict()
        save_data['avg'] = self.avg
        save_data['var'] = self.var
        save_data['mat'] = self.mat
        sio.savemat(path, save_data)


class HiddenMarkovModel:
    def __init__(self, mid_states, name, order, feature_dimension=MFCC_DIMENSIONS):
        self.a = np.zeros((mid_states + 2, mid_states + 2))
        self.avg = np.zeros((feature_dimension, mid_states + 2))
        self.var = np.zeros((feature_dimension, mid_states + 2))
        self.states = mid_states + 2
        self.mfccs = feature_dimension
        self.name = name
        self.order = order
        self.multithread = USING_MULTITHREAD

    def __model_param_initialization__(self, feature_sum, feature_square_sum, feature_size, init_val=0.4):
        """
        Subprocess for parameter initialization
        :param feature_sum: sum of features
        :param feature_square_sum: sum of squared features
        :param feature_size: number of features
        :param init_val: factor for setting initial transitional probability
        """
        for i in range(1, self.states - 1):
            self.avg[:, i] = feature_sum / feature_size
            self.var[:, i] = feature_square_sum / feature_size - Math.pow(self.avg[:, i], 2)
            Misc.check_min(self.var[:, i], 0.)
        for i in range(1, self.states - 1):
            self.a[i, i + 1] = init_val
            self.a[i, i] = 1 - self.a[i, i + 1]
        self.a[0, 1] = 1

    def log_observation_prob(self, feature, state):
        """
        Returns the observation probability at current state B(x_i) or P(x_i|q=state)
        :param feature: feature to be observed
        :param state: current state
        :return: the probability of observing the given feature at the current state
        """
        return Math.log_normal_distribution_seq_var(feature, self.avg[:, state], self.var[:, state], self.mfccs)

    def transitional_prob(self, cur_state, next_state):
        """
        Returns the transition probability from current state to the destined state A(i,j) or P(q=j|q=i)
        :param cur_state: Current state
        :param next_state:  Next state
        :return: Transitional probability
        """
        return self.a[cur_state, next_state]

    def model_initialization(self, data_loader, dimension=MFCC_DIMENSIONS):
        """
        Initialize the model
        :param data_loader: source of data
        :param dimension: number of MFCC dimension
        """
        feature_sum = np.zeros((dimension, 1)).squeeze()
        feature_square_sum = np.zeros((dimension, 1)).squeeze()
        feature_size = 0
        for i in tqdm.tqdm(range(len(data_loader)), desc="Initializing Model", file=sys.stdout):
            feature = Misc.to_tensor(data_loader[i])
            feature_sum += np.sum(feature, 0)
            feature_square_sum += np.sum(Math.pow(feature, 2), 0)
            feature_size += feature.shape[0]
        self.__model_param_initialization__(feature_sum, feature_square_sum, feature_size)

    def __log_sum_alpha__(self, log_alpha_mat, t, k, log_a):
        """
        Subprocess used for calculating alpha in the forwarding process
        :param log_alpha_mat: matrix of log(alpha)
        :param t: observation timestamp
        :param k: transitional state
        :param log_a: matrix of log(a)
        :return: sum used for calculating alpha
        """
        y = np.ones(log_alpha_mat.shape[0]) * (-Math.inf())
        ymax = -Math.inf()
        xlen = self.states
        for i in range(1, xlen - 1):
            y[i] = log_alpha_mat[t - 1, i] + log_a[i, k]
            if y[i] > ymax:
                ymax = y[i]
        if ymax == Math.inf():
            return Math.inf()
        else:
            ret = 0
            for i in range(1, xlen - 1):
                if ymax == -Math.inf() and y[i] == -Math.inf():
                    ret += 1
                else:
                    ret += Math.exp(y[i] - ymax)
            return ymax + Math.scalar_log(ret)

    def __log_sum_beta__(self, log_beta_mat, t, j, log_obs, log_a):
        """
        Subprocess used for calculating beta in the forwarding process
        :param log_beta_mat: matrix of log(beta)
        :param t: observation timestamp
        :param j: transitional probability
        :param log_obs: matrix of log(observation_probability)
        :param log_a: matrix of log(a)
        :return: sum used for calculating beta
        """
        y = np.ones(self.states) * (-Math.inf())
        ymax = -Math.inf()
        xlen = self.states
        for i in range(1, xlen - 1):
            y[i] = log_a[j, i] + log_obs[t, i] + log_beta_mat[t, i]
            if y[i] > ymax:
                ymax = y[i]
        if ymax == Math.inf():
            return Math.inf()
        else:
            ret = 0
            for i in range(1, xlen - 1):
                if ymax == -Math.inf() and y[i] == -Math.inf():
                    ret += 1
                else:
                    ret += Math.exp(y[i] - ymax)
            return ymax + Math.scalar_log(ret)

    def __model_forwarding_and_backwarding__(self, feature, squared_feature, feature_length, reevaluated_a_num,
                                             denominator, avg_num,
                                             var_num, log_a, likelihood):
        """
        Subprocess used for parameter updating using expectation-maximization algorithm
        :param feature: mfcc features
        :param squared_feature: squared mfcc features
        :param feature_length: length of the input feature
        :param reevaluated_a_num: reference to the matrix used to update the numerators of transitional matrix
        :param denominator: reference to the matrix used to update the denominators
        :param avg_num: reference to the matrix used to update the numerators of mean values in the observation distributions
        :param var_num: reference to the matrix used to update the numerators of deviations in the  observation distributions
        :param log_a: matrix of log(a) (namely, log(matrix of the transitional probability))
        :param likelihood: likelihood
        :return: likelihood
        """
        log_alpha = np.ones((feature_length, self.states)) * (-Math.inf())
        log_beta = np.ones((feature_length, self.states)) * (-Math.inf())
        log_obs = np.ones((feature_length, self.states)) * np.nan

        # Initialization & Reduction to Redundant Calculation
        # This optimization largely reduce the calculation overhead
        for i in range(feature_length):
            for j in range(1, self.states - 1):
                log_obs[i, j] = self.log_observation_prob(feature[i], j)

        # Alpha Initialization: log a(1,i)= log(p(xi|q1=j))+log(p(q1=j))
        for i in range(self.states):
            log_alpha[0, i] = log_a[0, i] + log_obs[0, i]

        # Alpha Iteration
        for t in range(1, feature_length):
            for i in range(1, self.states - 1):
                log_alpha[t, i] = self.__log_sum_alpha__(log_alpha, t, i, log_a) + log_obs[t, i]
        la_weight = self.__log_sum_alpha__(log_alpha, feature_length, self.states - 1, log_a)

        # Beta Initialization:
        log_beta[feature_length - 1, :] = log_a[:, self.states - 1]

        # Beta Iteration:
        for t in range(feature_length - 1, 0, -1):
            for j in range(1, self.states - 1):
                log_beta[t - 1, j] = self.__log_sum_beta__(log_beta, t, j, log_obs, log_a)
        log_beta[0, self.states - 1] = self.__log_sum_beta__(log_beta, 0, 0, log_obs, log_a)

        # Gamma Iteration:
        log_gamma = np.ones((feature_length, self.states)) * (-Math.inf())
        for t in range(feature_length):
            for i in range(1, self.states - 1):
                log_gamma[t, i] = log_alpha[t, i] + log_beta[t, i] - la_weight
        gamma = Math.exp(log_gamma)
        # Xi Iteration
        log_xi = np.ones((feature_length, self.states, self.states)) * (-Math.inf())
        for t in range(feature_length - 1):
            for j in range(1, self.states - 1):
                for i in range(1, self.states - 1):
                    log_xi[t, i, j] = log_alpha[t, i] + log_a[i, j] + log_obs[t + 1, j] + log_beta[t + 1, j] - la_weight
        for i in range(self.states):
            log_xi[feature_length - 1, i, self.states - 1] = log_alpha[feature_length - 1, i] + log_a[
                i, self.states - 1] - la_weight
        xi = Math.exp(log_xi)
        # Re-estimate transitional probability
        for i in range(1, self.states - 1):
            for j in range(1, self.states - 1):
                for t in range(feature_length):
                    reevaluated_a_num[i, j] += xi[t, i, j]
        for i in range(1, self.states - 1):
            for t in range(feature_length):
                denominator[i] += gamma[t, i]
                avg_num[:, i] += gamma[t, i] * feature[t]
                var_num[:, i] += gamma[t, i] * squared_feature[t]
        likelihood += la_weight
        return likelihood

    def model_training_single_iteration(self, data_loader, broadcast_progbar=None):
        """
        Parameter updating using expectation-maximization algorithm
        :param data_loader: source of data
        :param broadcast_progbar: tqdm progress bar. (should be none when the train runs in the single-worker mode)
        :return:
        """

        sum_a_numo = np.zeros_like(self.a)
        sum_avg_numo = np.zeros((self.mfccs, self.states))
        sum_var_numo = np.zeros((self.mfccs, self.states))
        sum_deno = np.zeros((self.states, 1)).squeeze()
        sum_likelihood = 0

        # forwarding & backwarding
        self.a[self.states - 1, self.states - 1] = 1
        log_a = Math.log(self.a)
        if broadcast_progbar is None or (isinstance(broadcast_progbar,int) and broadcast_progbar == 0):
            # training in traditional mode or multiprocessing mode
            desc = "Training "
            if isinstance(broadcast_progbar,int) and broadcast_progbar == 0:
                desc += self.name
            pb = tqdm.trange(len(data_loader), desc=desc, file=sys.stdout, position=self.order + 1, leave=not (isinstance(broadcast_progbar, int) and broadcast_progbar == 0))
            for i in pb:
                feature = Misc.to_tensor(data_loader[i])
                sq_feature = Misc.to_tensor(data_loader.get_squared_feature(i))
                sum_likelihood = self.__model_forwarding_and_backwarding__(feature, sq_feature, feature.shape[0],
                                                                           sum_a_numo, sum_deno, sum_avg_numo,
                                                                           sum_var_numo, log_a, sum_likelihood)
        else:
            # training in multithreading mode
            for i in range(len(data_loader)):
                feature = Misc.to_tensor(data_loader[i])
                sq_feature = Misc.to_tensor(data_loader.get_squared_feature(i))
                sum_likelihood = self.__model_forwarding_and_backwarding__(feature, sq_feature, feature.shape[0],
                                                                           sum_a_numo, sum_deno, sum_avg_numo,
                                                                           sum_var_numo, log_a, sum_likelihood)
                broadcast_progbar.update(1)

        # update parameters
        for i in range(1, self.states - 1):
            self.avg[:, i] = sum_avg_numo[:, i] / sum_deno[i]
            self.var[:, i] = sum_var_numo[:, i] / sum_deno[i] - Math.pow(self.avg[:, i], 2)
        for i in range(1, self.states - 1):
            for j in range(1, self.states - 1):
                self.a[i, j] = sum_a_numo[i, j] / sum_deno[i]

        # recover fixed parameters
        self.a[self.states - 1, self.states - 1] = 1.
        self.a[self.states - 2, self.states - 1] = 1 - self.a[self.states - 2, self.states - 2]
        return sum_likelihood
        # Misc.print("Likelihood = "+str(sum_likelihood))

    def model_fit(self, data_loader, model_checkpoint=None, model_checkpoint_suffix=None, pbar=None,
                  iteration=TRAINING_ITERATIONS):
        """
        Train the model
        :param data_loader: source of data
        :param model_checkpoint: checkpoint path
        :param model_checkpoint_suffix: name suffix of checkpoint file
        :param pbar: tqdm tqdm progress bar. (should be none when the train runs in the single-worker mode)
        :param iteration: current iteration
        :return: sum of likelihood
        """
        sum_likelihood = 0
        for i in range(iteration):
            sum_likelihood += self.model_training_single_iteration(data_loader, pbar)
            if model_checkpoint is not None:
                save_ckpt = HiddenMarkovModelCheckpoint(self.avg, self.var, self.a)
                save_ckpt.save(model_checkpoint + "/" + self.name + model_checkpoint_suffix + ".hmmckpt")
        return sum_likelihood

    def model_fit_multiproc(self, data_loader, model_checkpoint=None, model_checkpoint_suffix=None, pbar=None,
                  iteration=TRAINING_ITERATIONS):
        """
        Train the model in multiprocess env
        :param data_loader: source of data
        :param model_checkpoint: checkpoint path
        :param model_checkpoint_suffix: name suffix of checkpoint file
        :param pbar: tqdm tqdm progress bar. (should be none when the train runs in the single-worker mode)
        :param iteration: current iteration
        :return: sum of likelihood
        """

        sum_likelihood = 0
        for i in range(iteration):
            sum_likelihood += self.model_training_single_iteration(data_loader, pbar)
            if model_checkpoint is not None:
                save_ckpt = HiddenMarkovModelCheckpoint(self.avg, self.var, self.a)
                save_ckpt.save(model_checkpoint + "/" + self.name + model_checkpoint_suffix + ".hmmckpt")
        return sum_likelihood,self

    def __model_predict_single__(self, feature, feature_length, log_a):
        """
        Subprocess used for prediction. (Using Viterbi algorithm)
        :param feature: mfcc feature
        :param feature_length: length of feature
        :param log_a: matrix of log(a)
        :return:
        """
        fjt = (-Math.inf()) * np.ones((self.states, feature_length))
        for j in range(1, self.states - 1):
            fjt[j, 0] = log_a[0, j] + self.log_observation_prob(feature[0], j)
        for t in range(1, feature_length):
            for j in range(1, self.states - 1):
                fmax = -Math.inf()
                imax = -1
                f = -Math.inf()
                tmp = self.log_observation_prob(feature[t], j)
                for i in range(1, j + 1):
                    if fjt[i, t - 1] > -Math.inf():
                        f = fjt[i, t - 1] + log_a[i, j] + tmp
                    if f > fmax:
                        fmax = f
                        imax = i
                if imax != -1:
                    fjt[j, t] = fmax
        fopt = -Math.inf()
        for i in range(1, self.states - 1):
            f = fjt[i, feature_length - 1] + log_a[i, self.states - 1]
            if f > fopt:
                fopt = f
        return fopt

    def model_predict(self, data_loader, pbar=None):
        """
        Return likelihoods of given samples
        :param data_loader: source of test data
        :return: likelihoods of samples
        """
        ret = []
        self.a[self.states - 1, self.states - 1] = 1
        log_a = Math.log(self.a)
        if pbar is None or (isinstance(pbar,int) and pbar == 0):
            desc = "Validating  "
            if pbar is None or (isinstance(pbar,int) and pbar == 0):
                desc += self.name
            for i in tqdm.tqdm(range(len(data_loader)), desc=desc, file=sys.stdout,position=self.order + 1, leave=(not isinstance(pbar,int))):
                feature = Misc.to_tensor(data_loader[i])
                prob = self.__model_predict_single__(feature, feature.shape[0], log_a)
                ret.append(prob)
        else:
            for i in range(len(data_loader)):
                feature = Misc.to_tensor(data_loader[i])
                prob = self.__model_predict_single__(feature, feature.shape[0], log_a)
                ret.append(prob)
                pbar.update(1)
        return self.order,ret

    def model_load(self,checkpoint):
        ckpt = sio.loadmat(checkpoint,appendmat=False)
        self.var = ckpt['var']
        self.avg = ckpt['avg']
        self.a = ckpt['mat']


class HiddenMarkovModelValidator:
    """ Utility class for HMM testing
    """

    def __init__(self, models):
        self.models = models

    def predict(self, data_loader):
        """
        Predict labels
        :param data_loader: source of testing data
        :return: list of possible label indexes
        """
        global cur_prob
        prob = np.ones((len(self.models), len(data_loader)))
        cur_prob = prob
        pred = np.ones((len(data_loader)))
        for i in range(len(self.models)):
            print("Validating Model " + str(i + 1) + " of " + str(len(self.models)))
            prob[i, :] = self.models[i].model_predict(data_loader)[1]
        for i in range(len(data_loader)):
            pred[i] = np.argmax(prob[:, i])
        return pred.astype("int")

    def predict_ex(self, data_loader, dict_list):
        """
        Predict labels, converting labels according to model names
        :param data_loader: source of data
        :param dict_list: list of model names
        :return: list of possible labels
        """
        correct_samples = 0
        if USING_MULTITHREAD:
            pred = self.predict_multi_worker(data_loader)
        else:
            pred = self.predict(data_loader)
        predval = []
        correct_val = []
        for j in range(len(data_loader)):
            predval.append(dict_list[pred[j]])
            correct_val.append(data_loader.get_label(j))
            if dict_list[pred[j]] == data_loader.get_label(j):
                correct_samples += 1
        return predval, correct_samples

    def __predict_multi_worker_task__(self, idx, data_loader, pbar):
        ret = self.models[idx].model_predict(data_loader, pbar)[1]
        return idx, ret

    def predict_multi_worker(self, data_loader):
        """
        Predict labels, multi workers
        :param data_loader: source of testing data
        :return: list of possible label indexes
        """
        global cur_prob
        prob = np.ones((len(self.models), len(data_loader)))

        pred = np.ones((len(data_loader)))
        if USING_MULTIPROCESS:
            cur_prob = prob
            print("Validating Models (Multi-processing Mode)")
            mpr.freeze_support()
            lock = mpr.Lock()
            pool = mpr.Pool(len(self.models), initializer=tqdm.tqdm.set_lock, initargs=(lock,))
            with tqdm.tqdm(total=len(self.models), desc="Validating ", file=sys.stdout, leave=True, position=0) as pbar:
                global cur_pbar
                cur_pbar = pbar
                for i in range(len(self.models)):
                    pool.apply_async(multi_worker_val_w, args=(i, self.models, data_loader),
                                     callback=multi_worker_val_callback)
                pool.close()
                pool.join()
            prob = cur_prob
        else:
            print("Validating Models (Multi-threading Mode)")
            with tqdm.tqdm(total=len(data_loader) * len(self.models), desc="Validating ", file=sys.stdout, leave=True, position=0) as pbar:
                with ThreadPoolExecutor(max_workers=len(self.models)) as ex:
                    futures = [ex.submit(self.__predict_multi_worker_task__, idx, data_loader, pbar)
                               for idx in
                               range(len(self.models))]
                    for future in as_completed(futures):
                        ret = future.result()
                        prob[ret[0], :] = ret[1]

        for i in range(len(data_loader)):
            pred[i] = np.argmax(prob[:, i])
        return pred.astype("int")


class MultiWorkerUtils:
    """ Utility class used for multi-threading training
    """
    likelihood = 0
    hooked_train_result = None

    @staticmethod
    def multi_worker_train(idx, models, train_data_loaders, iteration, pbar):
        return models[idx].model_fit(train_data_loaders[idx], model_checkpoint=MODEL_CHECKPOINT,
                                     model_checkpoint_suffix="_iteration-" + str(iteration), pbar=pbar)

    @staticmethod
    def multi_worker_mp_callback(ret):
        MultiWorkerUtils.hooked_train_result.append(ret)
        return ret


def multi_worker_train(idx, models, train_data_loaders, iteration, pbar):
    return models[idx].model_fit(train_data_loaders[idx], model_checkpoint=MODEL_CHECKPOINT,
                                 model_checkpoint_suffix="_iteration-" + str(iteration), pbar=pbar)


def multi_worker_train_w(x, models, train_data_loaders, iter,ckpt):
    return models[x].model_fit_multiproc(train_data_loaders[x], model_checkpoint=ckpt,
                                 model_checkpoint_suffix="_iteration-" + str(iter), pbar=0)


def multi_worker_val_w(x, models, val_data_Loader):
    return models[x].model_predict(val_data_Loader,0)


def multi_worker_callback(ret):
    result_a.append(ret[0])
    models[ret[1].order] = ret[1]
    cur_pbar.update(1)


def multi_worker_val_callback(ret):
    cur_prob[ret[0], :] = ret[1]
    cur_pbar.update(1)


def training_main():
    print("Starting Timestamp:" + str(time.time()))
    print("Model Checkpoint:"+MODEL_CHECKPOINT)
    # Tackling Dataset
    dict_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    #dict_list = ['1']

    total_train_samples = 0
    val_data_loaders = []
    val_loader = MFCCDataLoader([])
    for i in range(len(dict_list)):
        dataset = MFCCDataSetDivider('wav', dict_list[i], VAL_RATIO)
        train_data_loaders.append(dataset.get_train_data_loader())
        val_data_loaders.append(dataset.get_val_data_loader())
        val_loader.join(val_data_loaders[i])
        total_train_samples += len(train_data_loaders[i])

    # Model init
    for i in range(len(dict_list)):
        Misc.print("\nPreparing Model " + str(i + 1) + " of " + str(len(dict_list)))
        train_data_loaders[i].preload()
        models.append(HiddenMarkovModel(12, "model_" + str(i), i))
        models[i].model_initialization(train_data_loaders[i])

    # Train & Test
    for t in range(TRAINING_EXT_ITERATIONS):
        global result_a
        iter = t
        # Train
        Misc.print("\nTraining Iteration " + str(t + 1) + " of " + str(TRAINING_EXT_ITERATIONS))
        if USING_MULTITHREAD:
            if USING_MULTIPROCESS and USING_MULTITHREAD:
                Misc.print("\nTraining Models (Multi-processing Mode)")
            else:
                Misc.print("\nTraining Models (Multi-threading Mode)")
            result_a = []

            if not USING_MULTIPROCESS:
                with tqdm.tqdm(total=total_train_samples, desc="Training ", file=sys.stdout) as pbar:
                    with ThreadPoolExecutor(max_workers=len(dict_list)) as ex:
                        futures = [
                            ex.submit(MultiWorkerUtils.multi_worker_train, idx, models, train_data_loaders, t, pbar)
                            for idx in
                            range(len(dict_list))]
                        for future in as_completed(futures):
                            result_a.append(future.result())
            else:
                mpr.freeze_support()
                lock = mpr.Lock()
                pool = mpr.Pool(len(dict_list), initializer=tqdm.tqdm.set_lock, initargs=(lock,))
                with tqdm.tqdm(total=len(models), desc="Training ", file=sys.stdout, leave=True, position=0) as pbar:
                    global cur_pbar
                    cur_pbar = pbar
                    for i in range(len(dict_list)):

                        pool.apply_async(multi_worker_train_w, args=(i, models, train_data_loaders, t,MODEL_CHECKPOINT),
                                         callback=multi_worker_callback)
                    pool.close()
                    pool.join()

            likelihoods = np.sum(np.array(result_a))
            Misc.print("Log Likelihood = " + str(likelihoods))
        else:
            for i in range(len(dict_list)):
                Misc.print("\nTraining Model " + str(i + 1) + " of " + str(len(dict_list)))
                models[i].model_fit(train_data_loaders[i], model_checkpoint=MODEL_CHECKPOINT,
                                    model_checkpoint_suffix="_iteration-" + str(t))
        Misc.print("\nValidating")
        # Validate
        #hmm_val = HiddenMarkovModelValidator(models)
        #pred, correct_samples = hmm_val.predict_ex(val_loader, dict_list)
        #Misc.print("Accuracy:" + str(correct_samples) + "/" + str(len(val_loader)))
    print("Finishing Timestamp:" + str(time.time()))


def testing_main():
    print("Starting Timestamp:" + str(time.time()))
    # Tackling Dataset
    dict_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    #dict_list = ['1']

    total_train_samples = 0
    val_data_loaders = []
    val_loader = MFCCDataLoader([])
    for i in range(len(dict_list)):
        dataset = MFCCDataSetDivider('wav', dict_list[i], 1)
        train_data_loaders.append(dataset.get_train_data_loader())
        val_data_loaders.append(dataset.get_val_data_loader())
        val_loader.join(val_data_loaders[i])
        total_train_samples += len(train_data_loaders[i])

    # Model init
    for i in range(len(dict_list)):
        #Misc.print("\nPreparing Model " + str(i + 1) + " of " + str(len(dict_list)))
        models.append(HiddenMarkovModel(12, "model_" + str(i), i))
        models[i].model_load(USING_TEST_PATH+models[i].name+"_iteration-"+str(USING_TEST_ITERATION-1)+".hmmckpt")

    val_loader.preload()

    # Validate
    hmm_val = HiddenMarkovModelValidator(models)
    pred, correct_samples = hmm_val.predict_ex(val_loader, dict_list)
    Misc.print("Accuracy:" + str(correct_samples) + "/" + str(len(val_loader)))
    print("Finishing Timestamp:" + str(time.time()))


class AssignmentEntry:
    @staticmethod
    def train_entry(parallel_type,data_source,val_split,iterations,save_path):
        global TRAINING_EXT_ITERATIONS,VAL_RATIO,USING_TEST_MODE,USING_MULTITHREAD,USING_MULTIPROCESS
        global USING_MFCC_LIBRARY,USING_CACHED_MFCC,MODEL_CHECKPOINT
        TRAINING_EXT_ITERATIONS = iterations
        VAL_RATIO = val_split
        USING_TEST_MODE = False
        if parallel_type == "mp":
            USING_MULTITHREAD = True
            USING_MULTIPROCESS = True
            Misc.print("WARNING: You are running in multiprocessing mode. Ensure that you have enough pagefile & memory to "
                       "avoid crash!")
        elif parallel_type == "mt":
            USING_MULTITHREAD = True
            USING_MULTIPROCESS = False
        elif parallel_type == "st":
            USING_MULTITHREAD = False
            USING_MULTIPROCESS = False

        if data_source == "self":
            USING_MFCC_LIBRARY = False
            USING_CACHED_MFCC = False
        elif data_source == "lib":
            USING_MFCC_LIBRARY = True
            USING_CACHED_MFCC = False
        elif data_source == "matlab":
            USING_MFCC_LIBRARY = False
            USING_CACHED_MFCC = True
        MODEL_CHECKPOINT = save_path
        training_main()

    @staticmethod
    def test_entry(parallel_type,data_source,use_iteration,ckpt_path):
        global TRAINING_EXT_ITERATIONS, VAL_RATIO, USING_TEST_MODE, USING_MULTITHREAD, USING_MULTIPROCESS
        global USING_MFCC_LIBRARY, USING_CACHED_MFCC
        global USING_TEST_PATH,USING_TEST_ITERATION
        VAL_RATIO = 1
        USING_TEST_MODE = False
        USING_TEST_PATH = ckpt_path
        USING_TEST_ITERATION = use_iteration
        if parallel_type == "mp":
            USING_MULTITHREAD = True
            USING_MULTIPROCESS = True
            Misc.print("WARNING: You are running in multiprocessing mode. Ensure that you have enough pagefile & memory to "
                       "avoid crash!")
        elif parallel_type == "mt":
            USING_MULTITHREAD = True
            USING_MULTIPROCESS = False
        elif parallel_type == "st":
            USING_MULTITHREAD = False
            USING_MULTIPROCESS = False

        if data_source == "self":
            USING_MFCC_LIBRARY = False
            USING_CACHED_MFCC = False
        elif data_source == "lib":
            USING_MFCC_LIBRARY = True
            USING_CACHED_MFCC = False
        elif data_source == "matlab":
            USING_MFCC_LIBRARY = False
            USING_CACHED_MFCC = True
        testing_main()



if __name__ == "__main__":
    if USING_TEST_MODE:
        testing_main()
    else:
        training_main()
