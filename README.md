# probability-python

## Probability classes in Python

The present code has been written for educational purpose only. Any other application is not recommended since the debugging stage and the stress test will not be implemented.  
  
  
PExp module consists of two classes: PExp and Event.  
  
  
PExp (class):


Allows to create a probabilistic experiment along with the number of outcomes, alphabet of symbols, elements population and associated probabilities.  
An experiment can be created by:  


#### PExp( [int or list], [alphabet = list], [prior_prob = list], [population = {dict or list}], [n_trials = int], [rec = boolean] )  
- _int_: is the number of the experiment outcomes.  It is sufficient to create a simple experiment. Must be at least 2. Future version may accept 1 and create a binary complementary experiment (A, notA). If no more arguments are passed, associated alphabet, population and probabilities are automatically set. The instance variable name is 'n_outcomes'.  
- _list_: contains the alphabet of the experiment.  Elements can be simple numbers or strings. If no more arguments are passed, associated probabilities, population and number of outcomes are automatically set. The instance variable name is 'alphabet'.  
- _alphabet_ = list: this keyword input is equal to the previous one.
- _prior_prob_: contains the prior probabilities associated to the experiment alphabet.  Elements may not sum to one: they are normalized during the experiment initialization. Even in this case, if no other arguments are passed the experiment will be automatically set. The instance variable name is 'prior_prob'.  
- _population_: describes the population individuals of the experiment.  It can be a dictionary or a list. When dictionary, keys are the elements name (as in alphabet) and values are the amount of individuals. For instance, {'W': 10, 'B': 5} describes a box with 10 white balls and 5 black balls, for a total of 15 balls. Otherwise, if list, the same information can be given by ['W','W','W','W','W','W','W','W','W','W','B','B','B','B','B']. Alphabet and prior_prob are set accordingly. If population is given, it has the greatest priority: all other arguments will be neglected.  
WARNING: if alphabet or prior_prob are given, population is created by the name of alphabet and values are set to 1 (therefore, it is not representative of the experiment probabilities).  
- _n_trials_: set the number of trials for the evaluation of post-probabilities based on frequency. Default is None.  
- _rec_: defines if the experiment is recursive or not. If True, after any experiment extraction, the related population individual value will be decreased by 1, setting the related probability accordingly.  


Methods:  
- _make_exp([int])_: allows to evaluate the post-probabilities based on frequency. If an argument is passed, it is the number of trials (n_trials).  
- _set_recursive(Bool)_: set 'rec' to True or False depending on the passed argument.  
- _reset_exp()_: reset the experiment by setting the population to the original values.   
- _info()_: prints the main informations about the experiment.  
  
Operators:  
- \*: (__MUL__) makes the cartesian product between two experiments.  

  
  
  
An event can be created by the Event class:  
  
#### Event(list, PExp)  
- _list_: contains the experiment elements (outcomes) describing the event.  
- _PExp_: the associated experiment.  
  
  
Methods:  
- _get_prob()_: returns the event associated probability  
- _get_prob_frac()_: returns the event associated probability as a fraction  
- _print_prob()_: prints the event associated probability  
- _is_independent(Event)_: returns the boolean value True or False if the passed Event argument is independent (True) or not (False) of the calling Event.  

Operators:  
- +: (__SUM__) makes the union of two events  
- \*: (__MUL__) makes the intersection of two events  
- |: (__OR__) makes the conditional operation between the events  
- \-: (__NEG__) makes the negation of the calling event  
  
  
  
