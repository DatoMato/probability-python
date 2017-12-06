import numpy as np
import pprint
from fractions import Fraction

###########################################################################################
#                                      CLASS PEXP                                         #
###########################################################################################

class PExp():
    '''
PExp (class):
Allows to create a probabilistic experiment along with the number of
outcomes, alphabet of symbols, elements population and associated
probabilities.
An experiment can be created by:

PExp( [int or list], [alphabet = list], [prior_prob = list],
    [population = {dict or list}], [n_trials = int], [rec = boolean] )
    int: is the number of the experiment outcomes. It is sufficient to
         create a simple experiment. Must be at least 2. Future version
         may accept 1 and create a binary complementary experiment (A,
         notA). If no more arguments are passed, associated alphabet,
         population and probabilities are automatically set. The
         instance variable name is 'n_outcomes'.
    list: contains the alphabet of the experiment. Elements can be
         simple numbers or strings. If no more arguments are passed,
         associated probabilities, population and number of outcomes
         are automatically set. The instance variable name is
         'alphabet'.
    alphabet = list: this keyword input is equal to the previous one.
    prior_prob: contains the prior probabilities associated to the
         experiment alphabet. Elements may not sum to one: they are
         normalized during the experiment initialization. Even in this
         case, if no other arguments are passed the experiment will be
         automatically set. The instance variable name is 'prior_prob'.
    population: describes the population individuals of the experiment.
         It can be a dictionary or a list. When dictionary, keys are
         the elements name (as in alphabet) and values are the amount
         of individuals. For instance, {'W': 10, 'B': 5} describes a
         box with 10 white balls and 5 black balls, for a total of 15
         balls. Otherwise, if list, the same information can be given
         by ['W','W','W','W','W','W','W','W','W','W','B','B','B','B',
         'B']. Alphabet and prior_prob are set accordingly. If
         population is given, it has the greatest priority: all other
         arguments will be neglected.
         WARNING: if alphabet or prior_prob are given, population is
         created by the name of alphabet and values are set to 1
         (therefore, it is not representative of the experiment
         probabilities).
    n_trials: set the number of trials for the evaluation of
         post-probabilities based on frequency.
         Default is None.
    rec: defines if the experiment is recursive or not. If True, after
         any experiment extraction, 
         the related population individual value will be decreased by
         1, setting the related probability accordingly.

  Methods:
    make_exp: allows to evaluate the post-probabilities based on
         frequency. If an argument is passed, it is the number of
         trials (n_trials).
    set_recursive: set 'rec' to True or False depending on the passed
         argument.
    reset_exp: reset the experiment by setting the population to the
         original values.
    info: prints the main informations about the experiment.

  Operators:
    *: (__MUL__) makes the cartesian product between two experiments.
    '''
    
    def __init__(self,*args,**kwargs):
        ####################################################
        #              Check input arguments               #
        ####################################################
        self._has_population = False
        self._has_alphabet = False
        self._has_probability = False
        self._has_n_outcomes = False

        if 'population' in kwargs:
            self._has_population = True
        if 'alphabet' in kwargs:
            self._has_alphabet = True
        if 'prior_prob' in kwargs:
            self._has_probability = True

        self.alphabet = []
        self.population = {}
        self.n_outcomes = 0

        ################### POPULATION #####################
        # Population has the highest priority
        if self._has_population:
            # If population is a dictionary, key is the element name and the value is the amount of elements
            if type(kwargs['population']) == dict:
                self.dict2alphabet_population(kwargs['population'])
            # If population is a list it contains the sequence of all elements
            elif type(kwargs['population']) == list:
                pop = {}
                aset = set(kwargs['population'])
                for sym in aset:
                    pop[sym] = 0
                for sym in kwargs['population']:
                    pop[sym] += 1
                self.dict2alphabet_population(pop)
            self.set_n_outcomes_from_alphabet()
            self.population2probability()

        ################### ALPHABET #####################
        elif self._has_alphabet:
            self.list2alphabet(kwargs['alphabet'])
            self.set_n_outcomes_from_alphabet()
            if self._has_probability:
                self.set_probability(kwargs['prior_prob'])
            else:
                self.set_probability(None)
        
        ################### PROBABILITY #####################
        elif self._has_probability:
            self.set_probability(kwargs['prior_prob'])
            self.set_n_outcomes_from_probability()
            if self._has_alphabet:
                self.list2alphabet(kwargs['alphabet'])
            else:
                if (len(args) > 0) and (type(args[0])==list):
                    self.list2alphabet(args[0])
                else:
                    self.list2alphabet(list(range(1,self.n_outcomes+1)))


        elif (len(args) > 0) and (type(args[0])==list):
            self.list2alphabet(args[0])
            self.set_n_outcomes_from_alphabet()
            if self._has_probability:
                self.set_probability(kwargs['prior_prob'])
            else:
                self.set_probability(None)

        elif (len(args) > 0) and (type(args[0])==int):
            self.n_outcomes = args[0]
            self._has_n_outcomes = True
            if self._has_probability:
                self.set_probability(kwargs['prior_prob'])
            else:
                self.set_probability(None)
            if self._has_alphabet:
                self.list2alphabet(kwargs['alphabet'])
            else:
                self.list2alphabet(list(range(1,self.n_outcomes+1)))

        else:
            raise AttributeError('Experiment must be initialized')

        if self._has_population == False:
            self.set_population()

        ################### Build Experiment Dictionary ###################
        self.set_exp_dict()

        ################### Set n_trials ###################
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']
        else:
            self.n_trials = None

        ################### SET RECURSIVE EXP #####################
        if 'rec' in kwargs:
            self.rec = kwargs['rec']
        else:
            self.rec = False

        ################### SET ORIGINAL VALUES ##################### 
        self._orig_population = None
        if self._has_population:
            self._orig_population = self.population.copy()
        self._orig_prob = self.prior_prob.copy()

        self.has_changed = False



    def set_population(self):
        for s in self.alphabet:
            self.population[s] = 1


    def set_exp_dict(self):
        ################### INIT POST PROBABILITY #####################  
        # Initialize post_prob
        self.post_prob = np.zeros(self.n_outcomes)
        
        ################### INIT EXP DICTIONARY #####################  
        # Make experiment dictionary
        self.exp_dict = {}
        for k in range(self.n_outcomes):
            self.exp_dict[self.alphabet[k]] = {'prior_prob':self.prior_prob[k],
                                               'prior_prob_frac':Fraction(self.prior_prob[k]).limit_denominator(1000),
                                               'post_prob':self.post_prob[k]}


    def list2alphabet(self,l):
        self.alphabet = []
        for k in range(len(l)):
            if type(l[k])==tuple:
                self.alphabet.append(l[k])
            else:
                self.alphabet.append((str(l[k]),))
        self._has_alphabet = True

    def dict2alphabet_population(self,d):
        l = list(d.keys())
        for s in l:
            if type(s)==tuple:
                self.alphabet.append(s)
                self.population[s] = d[s]
            else:
                self.alphabet.append((str(s),))
                self.population[(str(s),)] = d[s]
        self._has_population = True
        self._has_alphabet = True

    def set_n_outcomes_from_alphabet(self):
        if self._has_alphabet:
            self.n_outcomes = len(self.alphabet)
            self._has_n_outcomes = True
        else:
            raise ValueError('Alphabet must be initialized')

    def set_n_outcomes_from_probability(self):
        if self._has_probability:
            self.n_outcomes = len(self.prior_prob)
            self._has_n_outcomes = True
        else:
            raise ValueError('Prior_prob must be initialized')


    def population2probability(self):
        self.prior_prob = np.zeros(self.n_outcomes)
        if self._has_alphabet and self._has_n_outcomes:
            for k in range(self.n_outcomes):
                self.prior_prob[k] = self.population[self.alphabet[k]]
            self.prior_prob /= np.sum(self.prior_prob)
            self._has_probability = True
        else:
            raise ValueError('Alphabet must be initialized')

    def set_probability(self,p):
        if self._has_probability:
            if type(p) == list:
                self.n_outcomes = len(p)
                self._has_n_outcomes = True
            self.prior_prob = np.zeros(self.n_outcomes)
            for k in range(self.n_outcomes):
                self.prior_prob[k] = p[k]
        else:
            if self._has_n_outcomes == False:
                if self._has_alphabet == False:
                    raise ValueError('Alphabet or Prior_prob or N_outcomes must be initialized')
                else:
                    set_n_outcomes_from_alphabet()
            self.prior_prob = np.zeros(self.n_outcomes)
            for k in range(self.n_outcomes):
                self.prior_prob[k] = 1

        self.prior_prob /= np.sum(self.prior_prob)
        self._has_probability = True
    
    def info(self):
        print('-'*100)
        print('Population: {}'.format(self.population))
        print('Alphabet: {}'.format(self.alphabet))
        print('Prior_prob: {}'.format(self.prior_prob))
        for k in range(self.n_outcomes):
        	print('Prior_prob_frac[{}]: {}'.format(k+1,self.exp_dict[self.alphabet[k]]['prior_prob_frac']))
        print('N_outcomes: {}'.format(self.n_outcomes))
        print('N_trials: {}'.format(self.n_trials))
        print('Post_prob: {}'.format(self.post_prob))
        print('Recursive exp: {}'.format(self.rec))
        print('Exp_dict:')
        pprint.pprint(self.exp_dict)
        print()
        
    def make_exp(self,*args):
        if len(args)==1:
            self.n_trials = args[0]
        unif = np.random.uniform(size=self.n_trials)
        csp = np.cumsum(self.prior_prob)
        p0=0.0
        for k in range(self.n_outcomes):
            self.post_prob[k] = np.sum(csp[k] > unif)/self.n_trials - p0
            p0 += self.post_prob[k]
            self.exp_dict[self.alphabet[k]]['post_prob'] = self.post_prob[k]

    def update_exp_dict(self):
        for k in range(self.n_outcomes):
            self.exp_dict[self.alphabet[k]]['prior_prob'] = self.prior_prob[k]
            self.exp_dict[self.alphabet[k]]['prior_prob_frac'] = Fraction(self.prior_prob[k]).limit_denominator(100)


    def update_exp(self,el):
        self.population[el] -= 1
        k=0
        for sym in self.alphabet:
            self.prior_prob[k] = self.population[sym]
            k+=1
        # Normalize prior_prob
        self.prior_prob /= np.sum(self.prior_prob)
        self.update_exp_dict()
        self.has_changed = True


    def reset_exp(self):
        self.population = self._orig_population.copy()
        self.prior_prob = self._orig_prob.copy()
        self.update_exp_dict()
        self.has_changed = False

    def set_recursive(self,flag):
        self.rec = flag

    
    def __mul__(self,right):
        if type(right) != PExp:
            raise TypeError('All the operands must be PExp objects')
        alphabet = []
        n_outcomes = self.n_outcomes * right.n_outcomes
        prior_prob = np.zeros(n_outcomes,dtype='float')
        t = []
        for x in self.alphabet:
            for y in right.alphabet:
                t.append(list(x)+list(y))
        for x in t:
            alphabet.append(tuple(x))
        
        ind = 0
        for x in self.prior_prob:
            for y in right.prior_prob:
                prior_prob[ind] = x*y
                ind +=1
        
        return PExp(alphabet=alphabet,prior_prob=prior_prob)



###########################################################################################
#                                      CLASS EVENT                                         #
###########################################################################################

class Event():
    '''
Event(list, PExp)
    list: contains the experiment elements (outcomes) describing the
         event.
    PExp: the associated experiment.

  Methods:
    get_prob(): returns the event associated probability
    get_prob_frac(): returns the event associated probability as a
         fraction
    print_prob(): prints the event associated probability
    is_independent(Event): returns the boolean value True or False if
         the passed Event argument is independent (True) or not (False)
         of the calling Event.

  Operators:
    +: (__SUM__) makes the union of two events
    *: (__MUL__) makes the intersection of two events
    |: (__OR__) makes the conditional operation between the events
    -: (__NEG__) makes the negation of the calling event
    '''

    def __init__(self,abc,exp):
        if type(exp) != PExp:
            raise TypeError('Exp must be a member of PExp class')
            
        if type(abc) != list:
            raise TypeError('Elements must be a list')

        for k in range(len(abc)):
            if type(abc[k]) != tuple:
                abc[k] = (str(abc[k]),)

        
        fle = [True]*len(exp.alphabet)
        
        self.exp = exp
        
        tabc = []
        # Cicla su tutti gli elementi dell'evento
        for el in abc:
            if '*' not in el:
                tabc.append(el)
            else:
                for k in range(len(el)):
                    if el[k] != '*':
                        for j in range(len(exp.alphabet)):
                            if el[k] != exp.alphabet[j][k]:
                                fle[j] = False
                            
                for k in range(len(fle)):
                    if fle[k] == True:
                        tabc.append(exp.alphabet[k])
        
        self.abc = []
        for el in tabc:
            if el in exp.alphabet:
                self.abc.append(el)
                        
        self.prob = 0.0
        for el in self.abc:
            if el in exp.exp_dict:
                self.prob += exp.exp_dict[el]['prior_prob']
                if exp.rec:
                	exp.update_exp(el)
        
        
    def get_prob(self):
        return self.prob
        
    def get_prob_frac(self):
    	return Fraction(self.prob).limit_denominator(100)

    def print_prob(self):
    	print('Probability:',self.prob,'(',Fraction(self.prob).limit_denominator(100),')')

    def is_independent(self,ev2):
        evcond = self|ev2
        return np.isclose(evcond.prob,self.prob)
        
        
    def __neg__(self):
        abctemp = []
        for k in self.exp.alphabet:
            if k not in self.abc:
                abctemp.append(k)
        return Event(abctemp,self.exp)
    
    def __add__(self,right):
        if type(right) != Event:
            raise TypeError('All the operands must be Event objects')
        abctemp = list(set(self.abc + right.abc)) # Lists concatenation
        abctemp.sort()
        return Event(abctemp,self.exp)
    
    def __mul__(self,right):
        if type(right) != Event:
            raise TypeError('All the operands must be Event objects')
        abctemp = []
        for k in self.abc:
            if k in right.abc:
                abctemp.append(k)
        return Event(abctemp,self.exp)
    
    def __or__(self,right):
        if type(right) != Event:
            raise TypeError('All the operands must be Event objects')
        exptemp = PExp(alphabet=right.abc)
        evtemp = Event(self.abc,exptemp)
        return Event(evtemp.abc,exptemp)
    
    def __repr__(self):
        return(str(self.abc)+': '+str(self.prob))
    def __str__(self):
        return(str(self.abc)+': '+str(self.prob))

