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


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = []
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)    
            
                logL = model.score(self.X,self.lengths)
                bic_score= -2*logL + np.log(len(self.X))*(i**2 -1 + 2*i*len(self.X[0]))
                best_score.append((bic_score, model))
            except:
                pass
        if len(best_score) == 0:
            return  None
        else:        
            return min(best_score, key = lambda x: x[0])[1]

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. A model selection criterion for classification: Application to hmm topology optimization.
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        all_scores = []
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
            except:
                continue
            try:
                words = 0
                count = 0
                for k in self.hwords.keys():
                    if k != self.this_word:
                        x,l = self.hwords[k]
                        try:
                            words += model.score(x,l)
                            count += 1
                        except:
                            pass
                               
                dic_score = logL - words/(count-1)
                #print(count, words, dic_score)
                all_scores.append((dic_score, model))
            except:
                pass
        return max(all_scores, key = lambda z: z[0])[1]

    



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        kf = KFold()
        try:
            trainlist = []
            testlist = []
            kf_sequences = kf.split(self.sequences)
            for kfs in kf_sequences:
                        train, test = kfs
                        trainlist.append(combine_sequences(train, self.sequences))
                        testlist.append(combine_sequences(test,self.sequences))
        except:
            return self.base_model(self.n_constant)
        
        average_scores = []
        for i in range(self.min_n_components,self.max_n_components+1):
            scores = []
            for j in range(len(trainlist)):
                xtrain, ltrain = trainlist[j]
                xtest, ltest = testlist[j]
                try:
                    model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(xtrain,ltrain)
                
                    logL = model.score(xtest,ltest)
                    scores.append((logL, model))
                except:
                    print('fail1')
                
            if len(scores)>0:
                mean_scores = sum([z[0] for z in scores])/len(scores)
                average_scores.append((mean_scores, scores[0][1]))
        return max(average_scores, key = lambda y: y[0])[1]


# if __name__ == "__main__":
#     from  asl_test_model_selectors import TestSelectors
#     test_model = TestSelectors()
#     test_model.setUp()
#     test_model.test_select_bic_interface()