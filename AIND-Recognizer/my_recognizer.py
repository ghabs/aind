import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
	""" Recognize test word sequences from word models set

   :param models: dict of trained models
	   {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
	   both lists are ordered by the test set word_id
	   probabilities is a list of dictionaries where each key a word and value is Log Liklihood
		   [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
			]
	   guesses is a list of the best guess words ordered by the test set word_id
		   ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	probabilities = []
	guesses = []
	words = test_set.get_all_Xlengths()
	for word, (X, lengths) in words.items():
		probability = {}
		best_score = float('-Inf')
		best_guess = None
		for word, model in models.items():
			try:
				score = model.score(X, lengths)
				probability[word] = score
			except:
				score = float('-Inf')
				probability[word] = float('-Inf')
			if score > best_score:
				best_score, best_guess = score, word
		probabilities.append(probability)
		guesses.append(best_guess)
	return probabilities, guesses
