import string
import random
from math import floor

# population = Populaton(target, mutationRate, popmax)

class Population():
    def __init__(self, target, mutationRate, popmax):
        self.target = target
        self.mutationRate = mutationRate
        self.popmax = popmax
        self.DNA = []
        self.done = 0
        
        for i in range(popmax):
            genes = newDNA(len(target))
            self.DNA.append(genes)
    
    #calculate fitness
    def calcPopFitness(self):
        for i in range(self.popmax):
            self.DNA[i].calcFitness(self.target)
            #print self.DNA[i].fitness
            
    #generate mating pool        
    def naturalSelection(self):
        self.matingpool = []
        self.maxFitness = 0
        for i in range(self.popmax):
            if self.DNA[i].fitness > self.maxFitness:
                self.maxFitness = self.DNA[i].fitness
        #print self.maxFitness
        for i in range(self.popmax):
            self.DNA[i].normfitness = int((self.DNA[i].fitness / self.maxFitness) * 100)
        
            for j in range(self.DNA[i].normfitness):
                self.matingpool.append(self.DNA[i])
    
    def generate(self):
        for i in range(self.popmax):
            male = random.randint(0,len(self.matingpool)-1)
            female = random.randint(0,len(self.matingpool)-1)
            male = self.matingpool[male]
            female = self.matingpool[female]
            child = self.crossover(male, female)
            child = self.mutate(child, self.mutationRate)
            self.DNA[i] = child
    
    def crossover(self, male, female):
        child = newDNA(len(self.target))
        
        a = male.genes
        b = female.genes
        
        split = int(len(self.target) / 2)
        
        a = male.genes[:split]
        b = female.genes[split:]
        a.extend(b)
        
        child.genes = a
        return child
    
    def mutate(self, child, mutationRate):
        gene_pool = string.lowercase
        gene_pool += ' '
        for i in range(len(child.genes)):
            if mutationRate > random.random():
                x = random.randint(0, len(gene_pool)-1)
                child.genes[i] = gene_pool[x]
        return child
    
    def evaluate(self, generation):
        for i in range(self.popmax):
            j = ''.join(self.DNA[i].genes)
            print 'Generation:', generation, '===', j, '=== Fitness:', float(self.maxFitness)
            if j == target:
                print 'HOORAY!'
                self.done = 1
    
class newDNA():
    def __init__(self, length):
        self.genes = self.newGenes(length)
        self.fitness = 0
        self.normfitness = 0
    
    def newGenes(self, length):
        DNA = []
        gene_pool = string.lowercase
        gene_pool += ' '
        for i in range(length):
            x = random.randint(0, len(gene_pool)-1)
            #print i,':',x
            DNA.append(gene_pool[x])
        return DNA
        
    def calcFitness(self, target):
        #print self.genes
        #print target
        for i in range(len(target)):
            #print self.genes[i], '-', target[i]
            if  str(self.genes[i]) == str(target[i]):
                self.fitness += 1
        self.fitness = float(self.fitness) / float(len(target))
        #print self.fitness
        
if __name__ == '__main__':
    target = 'ragnarok'
    target = target.lower()
    #create initial population
    population = Population(target, mutationRate = .1, popmax = 200)
    generation = 1
    while True:
        #calculate fitness
        population.calcPopFitness()
        #print 'Fitness calculated...'

        #generate mating pool
        population.naturalSelection()
        #print 'Mating pool created...'

        #generate children
        population.generate()
        #print 'Children created...'

        population.evaluate(generation)

        if population.done == 1:
            print 'Done!!!'
            break
        
        generation += 1
        
