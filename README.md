# Markov Chain Transition Diagrams in Python

Simple Markov Chain visualization module in Python. Only requires **matplotlib** and **numpy** to work.


## Getting Started

### Dependencies

* matplotlib
* numpy

### Installation

Clone this repo to your script directory. Then

```
from mdp_display.render import MarkovDisplay
```

#### 2-state Markov chain demo

```
P = np.array([[0.8, 0.2], [0.1, 0.9]]) # Transition matrix
mc = MarkovChain(P, ['1', '2'])
mc.draw("../img/markov-chain-two-states.png")
```

![two state markov chain transition diagram python](https://github.com/jwatson-CO-edu/mdp_display/blob/master/img/markov-chain-two-states.png)


#### 3-state Markov chain demo

```
P = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.7, 0.2],
    [0.1, 0.7, 0.2],
])
mc = MarkovChain(P, ['A', 'B', 'C'])
mc.draw("../img/markov-chain-three-states.png")
```

![three state markov chain transition diagram python](https://github.com/jwatson-CO-edu/mdp_display/blob/master/img/markov-chain-three-states.png)


#### 4-state Markov chain demo

```
P = np.array([
    [0.8, 0.1, 0.1, 0.0],
    [0.1, 0.7, 0.0, 0.2],
    [0.1, 0.0, 0.7, 0.2],
    [0.1, 0.0, 0.7, 0.2]
])
mc = MarkovChain(P, ['1', '2', '3', '4'])
mc.draw("../img/markov-chain-four-states.png")
```

![four state markov chain transition diagram python](https://github.com/jwatson-CO-edu/mdp_display/blob/master/img/markov-chain-four-states.png)

#### N-state Markov chain demo

```
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
```  
Note that parallel edges that span multiple rows/columns are not handled well.

![N state markov chain transition diagram python](https://github.com/jwatson-CO-edu/mdp_display/blob/master/img/markov-chain-N-states.png)



## License

This project is licensed under the GPL V3 licence.

