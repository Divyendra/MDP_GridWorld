import os.path as path
from collections import OrderedDict


class MDP(object):
    
    def __init__(self, actions, fname):
        if not path.exists(fname):
            raise ValueError("Improper path or File doesnt exist")
        
        a_map = OrderedDict()
        for index, a in enumerate(actions):
            a_map[a] = index; 

        #Get no. of states
        s = 1; e = 2
        with open(fname, 'r') as f:
            n = int(next(get_lines(f, s, e)))
    
        #Get all State rewards
        s = e; e = s+n
        R = []
        with open(fname, 'r') as f:
            for r in get_lines(f, s, e): 
                R.append(float(r))
    
        #Get Transition distribution
        T = []; #3d array with dimensions as actions, states & successors respectively
        for a in a_map: #For every action
            S = [] #All State transitionprobs for a particular action 'a'
            s = e; e = s+n
            with open(fname, 'r') as f:
                for state in get_lines(f, s, e): 
                    state_probs = state.rstrip('\n').split(',')
                    state_probs = [float(_) for _ in state_probs]
                    S.append(state_probs)
            T.append(S)
        
        self.n = n
        self._R = R
        self._T = T
        self._actions = actions
        self._a_map =a_map
        self.loopy = []  #Loopy states
        
    def T(self, s, a, s1):
        action_index = self._a_map[a]
        return self._T[action_index][s][s1]
    
    def R(self, s):
        return self._R[s]
    
    def actions(self):
        return self._actions
    
    def filter(self):
        'Filters the loopy states with negative rewards'
        states = [i for i in range(self.n)]
        for s in states:
            loopy = all([int(self.T(s,a,s)) == 1 for a in self.actions()])
            loopy = loopy and (self.R(s) < 0)
            if loopy:
                self.loopy.append(s) 
    

def value_iteration(mdp, gamma=1.0, epsilon=0.001):
    '''
    @param gamma: float :: The discount factor
    @param epsilon: float :: The allowed error from optimal utilities
    
    @return : A tuple of (Utilityvalues, OptimalPolicy) 
    '''
    states = [i for i in range(mdp.n)]
    successors = [i for i in range(mdp.n)]
    
    R, T, actns = mdp.R, mdp.T, mdp.actions()
    
    def expected_utility(s, a, succs, U):
        return sum([T(s,a,s1)*U[s1] for s1 in succs])

#     U_t = [0.0 for i in range(mdp.n)] #State utilities. Start from 0
    U_t = []
    for i in range(mdp.n):
        if i in mdp.loopy:
            U_t.append(-1000.0)
        else:
            U_t.append(0.0)
        
    U_t1 = ['dummy' for i in range(mdp.n)]
    P_t = ['dummy' for i in range(mdp.n)]   #Policy
    while True:    
        delta = 0
        for s in states: 
            (q_val, a) = max([(expected_utility(s, a, successors, U_t),a) for a in actns], key= lambda x: x[0])
            U_t1[s] = R(s) + gamma * q_val
            if not int(q_val) == 0:
                P_t[s] = a
            else:  #Terminal state
                P_t[s] = 'T' 
            delta = max(delta, U_t1[s] - U_t[s])
        
#         print(delta)
        #Termination Condition
        if delta <= epsilon * (1 - gamma)/gamma:
            return (U_t, P_t)
        
        U_t = U_t1.copy() 

        

#Helpers
def get_lines(fileobj, s, e):
    'Returns backs the lines between [s, e). s-start;e-end'
    cnt = 0
    #l_no: linenumber; l-line
    for l_no, l in enumerate(fileobj):
        if not (s <= l_no+1 < e): continue
        cnt += 1
        yield l.rstrip('\n')  #returning EOL striped line
        if e - s == cnt: break   #Done reading the required lines, so why loop further

#Prints state utility values
def print_1_U(U):
    print("-"*50)
    for no, s in enumerate(U):
        if no != 24:
            print("%7.2f" % s , end='|', flush=True)
        if (no+1) % 6 == 0:
            print('\n' + "-"*50)

#Prints Policy using unicode art
def print_1_P(P):
    print("-"*30)
    for no, a in enumerate(P):
        if no != 24:
            #Unicode ordinal value setting
            if a == 'L':
                uni = chr(8656)
            elif a == 'U':
                uni = chr(8657)
            elif a == 'R':
                uni = chr(8658)
            elif a == 'D':
                uni = chr(8659)
            elif a == 'T':
                uni = chr(8622)
                
            print("%3s" % uni , end='|', flush=True)
        if (no+1) % 6 == 0:
            print('\n' + "-"*30)
            
def print_2_U(U):
    print("-"*30)
    for no, s in enumerate(U):
        if no != 12:
            print("%7.2f" % s , end='|', flush=True)
        if (no+1) % 4 == 0:
            print('\n' + "-"*30)
            
#Prints Policy using unicode art
def print_2_P(P):
    print("-"*30)
    for no, a in enumerate(P):
        if no != 12:
            #Unicode ordinal value setting
            if a == 'L':
                uni = chr(8656)
            elif a == 'U':
                uni = chr(8657)
            elif a == 'R':
                uni = chr(8658)
            elif a == 'D':
                uni = chr(8659)
            elif a == 'T':
                uni = chr(8622)
                
            print("%3s" % uni , end='|', flush=True)
        if (no+1) % 4 == 0:
            print('\n' + "-"*30)
                            
def main():
#-----------1.1-------------
#     fname = "gw1.txt"
#     actions = ['L', 'U', 'R', 'D']
#     mdp = MDP(actions, fname)
#  
#     U, P = value_iteration(mdp, gamma=1)
#      
#     print_1_U(U)
#     print_1_P(P)

#-----------1.2-------------
    fname = "gw2.txt"
    actions = ['L', 'U', 'R', 'D']
    mdp = MDP(actions, fname)
    mdp.filter()
 
    U, P = value_iteration(mdp, gamma=1)
     
    print_2_U(U)
    print_2_P(P)

    
if __name__ == "__main__":
    main()