import numpy as np
import pprint
from fractions import Fraction

class PExp():
    
    def __init__(self,*args,**kwargs):
        # Population has the highest priority
        if 'population' in kwargs:
            # Population is a list
            if type(kwargs['population']) == list:
                # Create alphabet from population
                self.alphabet = sorted(list(set(kwargs['population'])))
                # Set alphabet items as tuples
                for k in range(len(self.alphabet)):
                    if type(self.alphabet[k]) != tuple:
                        self.alphabet[k] = tuple(self.alphabet[k])
                # Set n_outcomes
                self.n_outcomes = len(self.alphabet)
                # Set prior_prob
                self.prior_prob = np.zeros(self.n_outcomes)
                # Set population
                self.population = dict.fromkeys(self.alphabet,0)
                k = 0
                for sym in self.alphabet:
                    for s in kwargs['population']:
                        if tuple(s) == sym:
                            self.population[sym]+=1
                            self.prior_prob[k]+=1
                    k+=1
                # Normalize prior_prob
                self.prior_prob /= np.sum(self.prior_prob)
            
            # Population is a dictionary
            elif type(kwargs['population']) == dict:
                # Create alphabet from population
                self.alphabet = sorted(kwargs['population'].keys())
                # Set population
                self.population = kwargs['population']
                # Set n_outcomes
                self.n_outcomes = len(self.alphabet)
                # Set prior_prob
                self.prior_prob = np.zeros(self.n_outcomes)
                k=0
                for sym in self.alphabet:
                    self.prior_prob[k] = kwargs['population'][sym]
                    k+=1
                # Normalize prior_prob
                self.prior_prob /= np.sum(self.prior_prob)
                
            # Population argument can be only a list or a dictionary: raising excpetion
            else:
                raise TypeError('Population must be a list or a dictionary')
                
        # If population is not present at input, build experiment with prior_prob
        elif 'prior_prob' in kwargs:
            # Set population to None
            self.population = None
            # If prior_prob argument is a list, transform to a numpy array
            if type(kwargs['prior_prob']) == list:
                self.prior_prob = np.array(kwargs['prior_prob'],dtype='float')
            # prior_prob argument is a numpy array
            elif isinstance(kwargs['prior_prob'],np.ndarray):
                self.prior_prob = kwargs['prior_prob']
            # Normalize prior_prob
            self.prior_prob /= np.sum(self.prior_prob)
            # Set n_outcomes
            self.n_outcomes = len(self.prior_prob)
            # If alphabet argument is present and is a list, set alphabet
            if 'alphabet' in kwargs:
                self.alphabet = []
                if(len(kwargs['alphabet']) != self.n_outcomes):
                    raise ValueError('Alphabet length must be equal to prior_prob length')
                for a in kwargs['alphabet']:
                    self.alphabet.append(tuple(a))
            # If alphabet argument is not present, build an artificial alphabet (1,2,3,...)
            else:
                self.alphabet = []
                for k in range(self.n_outcomes):
                    self.alphabet.append(tuple()+(str(k+1),))
                    #self.alphabet.append(tuple(k+1))
                    #self.alphabet.append((k+1,))
                    
        # If neither population nor prior_prob are arguments, experiment is built by alphabet
        elif 'alphabet' in kwargs:
            # Set population to None
            self.population = None
            # Set alphabet
            self.alphabet = []
            for a in kwargs['alphabet']:
                if type(a) == tuple:
                    self.alphabet.append(a)
                else:
                    self.alphabet.append(tuple()+(a,))
            # Set n_outcomes
            self.n_outcomes = len(self.alphabet)
            # Set prior_prob as uniform
            self.prior_prob = np.ones(self.n_outcomes,dtype='float')/self.n_outcomes
            
        # If no arguments are present (population, prior_prob, alphabet), check if a single numeric
        # value is passed: this is n_outcomes
        else:
            if len(args)>1:
                raise AttributeError('PExp admits only one free argument as n_outcomes')
            elif len(args)==1:
                self.n_outcomes = args[0]
            else:
                self.n_outcomes = 2
            self.population = None
            self.prior_prob = np.ones(self.n_outcomes,dtype='float')/self.n_outcomes
            self.alphabet = []
            for k in range(self.n_outcomes):
                self.alphabet.append(tuple()+(str(k+1),))
                #self.alphabet.append(tuple(k+1))
                #self.alphabet.append((k+1,))
                
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']
        else:
            self.n_trials = 1000*self.n_outcomes
        
        # Initialize post_prob
        self.post_prob = np.zeros(len(self.prior_prob))
        
        # Make experiment dictionary
        self.exp_dict = {}
        for k in range(self.n_outcomes):
            self.exp_dict[self.alphabet[k]] = {'prior_prob':self.prior_prob[k],'post_prob':self.post_prob[k]}
            
    
    def info(self):
        print('-'*100)
        print('Population: {}'.format(self.population))
        print('Alphabet: {}'.format(self.alphabet))
        print('Prior_prob: {}'.format(self.prior_prob))
        print('N_outcomes: {}'.format(self.n_outcomes))
        print('N_trials: {}'.format(self.n_trials))
        print('Post_prob: {}'.format(self.post_prob))
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
    

class Event():
    def __init__(self,abc,exp):
        if type(exp) != PExp:
            raise TypeError('Exp must be a member of PExp class')
            
        if type(abc) != list:
            raise TypeError('Elements must be a list')
        
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
        
        
    def get_prob(self):
        return self.prob
        
        
        
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

