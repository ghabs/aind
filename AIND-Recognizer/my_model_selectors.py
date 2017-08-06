import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
	'''
	base class for model selection (strategy design pattern)
	'''

	def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
				 n_constant=3,
				 min_n_components=2, max_n_components=10,
				 random_state=14, verbose=False):
		self.words = all_word_sequences
		self.hwords = all_word_Xlengths
		self.sequences = all_word_sequences[this_word]
		self.X, self.lengths = all_word_Xlengths[this_word]
		self.this_word = this_word
		self.n_constant = n_constant
		self.min_n_components = min_n_components
		self.max_n_components = max_n_components
		self.random_state = random_state
		self.verbose = verbose

	def select(self):
		raise NotImplementedError

	def base_model(self, num_states):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# warnings.filterwarnings("ignore", category=RuntimeWarning)
		try:
			hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
									random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
			if self.verbose:
				print("model created for {} with {} states".format(self.this_word, num_states))
			return hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
			return None


class SelectorConstant(ModelSelector):
	""" select the model with value self.n_constant

	"""

	def select(self):
		""" select based on n_constant value

		:return: GaussianHMM object
		"""
		best_num_components = self.n_constant
		return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
	""" select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		try:
			best_score = float("-Inf")
			best_model = None
			for n_components in range(self.min_n_components, self.max_n_components):
				al = 0.0
				count = 0.0
				hmm_model = self.base_model(n_components)
				try:
					logL = hmm_model.score(self.X, self.lengths)
					for w in self.hwords:
						X, lengths = self.hwords[w]
						al += hmm_model.score(X, lengths)
						count += 1.0
					al = al / count
					score = logL - al
				except:
					score = float("-Inf")
				if score > best_score:
					best_score, best_model = score, hmm_model
			return best_model
		except Exception as e:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))



class SelectorBIC(ModelSelector):
	'''
	select the model with the lowest Bayesian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		try:
			best_score = float("Inf")
			best_model = None
			for n_components in range(self.min_n_components, self.max_n_components):
				hmm_model = self.base_model(n_components)
				try:
					p = (hmm_model.startprob_.size - 1) + (hmm_model.transmat_.size - 1) + hmm_model.means_.size + hmm_model.covars_.diagonal().size
					score = -2* hmm_model.score(self.X, self.lengths) + p*math.log(len(self.sequences))
				except:
					score = float("Inf")
				if score < best_score:
					best_score, best_model = score, hmm_model
			return best_model
		except Exception as e:
			raise


class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def select(self):
		# with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		warnings.filterwarnings("ignore", category=RuntimeWarning)
		best_score = float("-Inf")
		best_model = None
		try:
			n_splits = 3
			split_method = KFold(n_splits=n_splits)
			for n_components in range(self.min_n_components, self.max_n_components):
				scores = []
				hmm_model = None
				for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
					try:
						if len(self.sequences) < n_splits:
							break
						self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
						hmm_model = self.base_model(n_components)
						testX, testLengths = combine_sequences(cv_test_idx, self.sequences)
						logL = hmm_model.score(testX, testLengths)
						scores.append(logL)
					except:
						hmm_model = self.base_model(n_components)
						scores.append(float("-Inf"))
				logL = np.mean(scores)
				if logL > best_score:
					best_score = logL
					best_model = hmm_model
		except:
			if self.verbose:
				print("failure on {} with {} states".format(self.this_word, num_states))
		if best_model is None:
			return self.base_model(self.n_constant)
		return best_model
