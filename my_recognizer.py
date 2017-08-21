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

    for i in range(len(test_set.get_all_Xlengths())):
        x, lengths = test_set.get_item_Xlengths(i)
        score_dict = {}
        for word, model in models.items():
            try:
                logL = model.score(x, lengths)
                score_dict[word] = logL
            except:
                score_dict[word] = float('-inf')
        probabilities.append(score_dict)
        guesses.append(max(probabilities[i], key=probabilities[i].get))

    return (probabilities, guesses)

def recognize_sentences(models: dict,  Xlengths: dict):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param Xlengths: dict of x, length tuples
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

    for i in range(len(Xlengths)):
        x, lengths = Xlengths[i]
        score_dict = {}
        for word, model in models.items():
            try:
                logL = model.score(x, lengths)
                score_dict[word] = logL
            except:
                score_dict[word] = float('-inf')
        probabilities.append(score_dict)
        guesses.append(max(probabilities[i], key=probabilities[i].get))

    return (probabilities, guesses)


