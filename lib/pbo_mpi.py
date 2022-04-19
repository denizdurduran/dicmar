from mpi4py import MPI
import numpy as np
from numpy import matlib as mb
import time

# Initialize communication
comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()


class GenericMPI_PBO:
    '''
        Generic MPI Population Based Optimization
    '''
    def __init__(self,p0):
        self.p = None
        self.p_best = None
        self.y = None

        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_size = comm.Get_size()
        self.mpi_rank = comm.Get_rank()

        self.f = lambda x: x
        '''
            p0: initial population, can be a vector or a matrix if a matrix rows are number of states columns are number of particle
        '''
        if(len(p0.shape) == 2):
            if(p0.shape[0] is not self.mpi_size):
                print("Error population size is to big compared to number of mpi instances expected {} got {}".format(self.mpi_size,p0.shape[1]))
                raise ValueError
                
        self.dim = p0.shape[1]
        self.population = None
        self.fitnesses = None
        self.generation = None

        if self.mpi_rank == 0:
            self.generation = -1
            if(len(p0.shape) == 2):
                self.population = p0.reshape(p0.size)
            else:
                self.population = np.array([p0 for i in range(self.mpi_size)])
            self.fitnesses = np.empty(size, dtype='d')
            #print(self.population)
            

    def update(self):
        raise NotImplementedError()

    def getIndividualFitness(self,i):
        return self.fitnesses[i]

    def getPopulation(self,population=None):
        if population is None:
            return self.population.reshape(self.mpi_size, self.dim)
        else :
            return population.reshape(self.mpi_size, self.dim)

    def getFlattenPopulation(self,population=None):
        if population is None:
            return self.population.reshape(self.population.size)
        else :
            return population.reshape(population.size)


    def getIndividual(self,i):
        return self.population.reshape(self.mpi_size, self.dim)[i,:]

    def getBestIndividualFitness(self):
        i = np.argmin(self.fitnesses)
        return self.fitnesses[i]

    def getBestIndividual(self):
        return self.population.reshape(self.mpi_size, self.dim)[np.argmin(self.fitnesses),:]

    def step(self):
        
        slave_toBeReceived = np.empty(self.dim, dtype='d') # allocate space for slave_toBeReceived
        comm.Scatter(self.population, slave_toBeReceived, root=0)
        self.p = slave_toBeReceived
        if(self.mpi_rank == 0):
            self.generation = self.generation + 1
        
        self.y = self.f(self.p)

        comm.Gather(self.y, self.fitnesses, root=0)

        if(self.mpi_rank == 0):
            #print('Generation {}: Best fitness is : {}'.format(self.generation,min(self.fitnesses)))
            self.update()


class RandomSearch(GenericMPI_PBO):
    def __init__(self,p0):
        super().__init__(p0)
    def update(self):
        best = self.getBestIndividual()
        self.population = best.tolist()
        for _ in range(self.mpi_size-1):
            self.population += (best+0.01*np.random.randn(best.shape[0])).tolist()
        self.population = np.array(self.population)


class PSO(GenericMPI_PBO):
    def __init__(self,p0, RANGES, F):
        super().__init__(p0)
        self.f = F
        if self.mpi_rank == 0:
            self.ranges = RANGES
            self.speed = self.getPopulation()
            self.global_best = self.getIndividual(0)
            self.global_best_fitness = 1e6
            self.local_bests = self.getPopulation()
            self.local_bests_fitness = 1e6+np.zeros(self.fitnesses.shape)
            

    def clip(self,x):
        return np.clip(x,self.ranges[0],self.ranges[1])


    def updateBests(self):
        if (self.getBestIndividualFitness() < self.global_best_fitness):
            #print("Found a better individual {}".format(self.getBestIndividualFitness()))
            self.global_best = self.getBestIndividual()
            self.global_best_fitness = self.getBestIndividualFitness()
        for i in range(self.mpi_size):
            if(self.getIndividualFitness(i) < self.local_bests_fitness[i]):
                #print("Found local better individual")
                self.local_bests_fitness[i] = self.getIndividualFitness(i)
                self.local_bests[i] = self.getIndividual(i)
        
            
    def update(self):
        decay = 1.0*(1/(1+1.0*np.log10(self.generation)))
        self.updateBests()
        X = self.getPopulation()
        V = self.speed
        w = 0.705
        b1 = 2.05
        b2 = 2.05
        _b1 = np.random.uniform(0,b1)
        _b2 = np.random.uniform(0,b2)
        self.speed = self.clip(
            w*V 
            + _b1 * ( self.local_bests - X)
            + _b2 * ( mb.repmat(self.global_best+0.01*decay*np.random.randn(self.global_best.shape[0]),self.mpi_size,1) - X))
        
        self.population = self.clip(self.getFlattenPopulation(X + decay*self.speed))
        print('Generation {}: Best fitness is : {}'.format(self.generation,self.global_best_fitness))
        if(self.generation > 20000):
            import ipdb; ipdb.set_trace()

