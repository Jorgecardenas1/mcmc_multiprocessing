
from mpi4py import MPI

class multiproc:

    
    
    def __init__(self):
        self.comm = self.createComm(MPI)
        
    def getSize(self,comm):
        return comm.Get_size()


    def getRank(self,comm):

        rank = comm.Get_rank()
        print('\nMy rank is ',rank)
        return rank

    def createComm(self,MPI):
        
        comm = MPI.COMM_WORLD
        return comm




