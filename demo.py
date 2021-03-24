import numpy as np
import matplotlib.pyplot as plt

# modules from this repository
import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from mdp_display.render import MarkovDisplay

def main():
    
    if 0:
        #--------------------------------------------------------------------------
        # 2-state Markov chain
        #--------------------------------------------------------------------------
        P = np.array([[0.8, 0.2], [0.1, 0.9]]) # Transition matrix
        mc = MarkovDisplay(P, ['1', '2'])
        mc.draw("img/markov-chain-two-states.png")
    
    if 0:
        #--------------------------------------------------------------------------
        # 3-state Markov chain
        #--------------------------------------------------------------------------
        P = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
        ])
        mc = MarkovDisplay(P, ['A', 'B', 'C'])
        mc.draw("img/markov-chain-three-states.png")
 
    if 1:
        #--------------------------------------------------------------------------
        # 4-state Markov chain
        #--------------------------------------------------------------------------
        P = np.array([
            [0.8, 0.1, 0.1, 0.0], 
            [0.1, 0.7, 0.0, 0.2],
            [0.1, 0.0, 0.7, 0.2],
            [0.1, 0.0, 0.7, 0.2]
        ])
        mc = MarkovDisplay(P, ['1', '2', '3', '4'])
        mc.draw("img/markov-chain-four-states.png")
    
    
    if 0:
        #--------------------------------------------------------------------------
        # N-state Markov chain
        #--------------------------------------------------------------------------
        from random import randrange
        from mdp_display.MDP import normalize_discrete_dist, sample_bernoulli
        N = 9
        P = np.zeros( (N,N) )
        for i in range(N-1):
            row = np.zeros( N )
            for j in range(i, N):
                if (abs(i-j) <= 2) and sample_bernoulli( 0.65 ):
                    row[j] = randrange(20) 
            P[i,:] = np.array(  normalize_discrete_dist( row )  )
        mc = MarkovDisplay( P, [str(i) for i in range(1,N+1)] )
        mc.draw("img/markov-chain-N-states.png")
            
    

if __name__ == "__main__":
    main()

