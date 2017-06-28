# GeneticAlgorithm
Working my way to creating an version of Ragnarok Online that runs a genetic algorithm. lol.

1. shakespeare_monkey_thecodingtrain.py
    - inspired by: The Coding Train's https://www.youtube.com/watch?v=-jv3CgDN9sc&list=PLRqwX-V7Uu6bJM3VgzjNV5YxVxUwzALHV&index=4

2. agent_survival.py
    - inspired by:
        - The Coding Train's https://www.youtube.com/watch?v=-jv3CgDN9sc&list=PLRqwX-V7Uu6bJM3VgzjNV5YxVxUwzALHV&index=4
        - emgoz's https://github.com/emgoz/Neural-network-snake
        - Bart van Dooren's https://github.com/Nerfling/Evolving-Creatures-Using-Neural-Networks-And-Genetic-Algorithms
        - stela zhang's https://www.youtube.com/watch?v=eZ14la6zttM
        - phyces' https://www.youtube.com/watch?v=GvEywP8t12I
        - Geoff. S. Nitschke and Leo. H. Langenhoven's Neuro-Evolution for Competitive Co-evolution of Biologically Canonical Predator and Prey Behaviors (https://people.cs.uct.ac.za/~gnitschke/projects/papers/2010-Neuro-Evolution%20for%20Competitive%20Co-evolution%20of%20Biologically%20Canonical%20Predator%20and%20Prey%20Behaviors.pdf)
    - implemented using pybrain and pygame
        - pybrain: http://pybrain.org/docs/
        - pygame: http://thepythongamebook.com
            - Most of the implementations I found were on Java or C. That being said, it was quite a challenge to make pygame work. lol.
    
    - Notes (or questions I was able to answer through study):
        a. Important to understand the difference of a normal feedforward neural net (with backprop) and a feedforward neural net that is optimized using genetic algorithms/neuroevolution. 
        b. What does it mean to evolve a neural net based on optimal behavior?
        c. Is genome always a list when being subjected to crossovers? or can different parts of the genome have its own individual crossover instance?
        d. Are there different ways of doing natural selection, generation, crossover, and mutation?
        e. How should the fitness function be created to steer agents towards a certain direction of optimal behavior?
