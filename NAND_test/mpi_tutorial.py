import numpy as np
import sys
from time import sleep
from time import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI procs
rank = comm.Get_rank()  # i.d. for local proc

def rel_time(start_time):
    current_time = time()
    return (current_time - start_time)


stime = None
final_list = None
trial_n = 0
new_vals = [0]

# make sure that the clock starts the same for all procs
if rank==0:
    stime=time()
    final_list = [0]
stime = comm.bcast(stime, root=0)

def tprint(string):
    print(f'{rel_time(stime): 3.0f}'+' '+string)
    return

i = 0
j = 0


while len(comm.bcast(final_list,root=0)) < 10:
    i += 1
    sleep(rank+.1)

    # only make a new guess for procs that aren't proc 0
    if rank != 0:
        trial_n = np.random.randint(10)
        tprint(f'rank {rank} guess {i}: {trial_n}; has {j} points \r ')
        sys.stdout.flush()

    # here we check against the list held in proc 0, rather than trying to sync the list between all procs
    if trial_n not in comm.bcast(final_list, root=0):
        j += 1

        #send the guesses to the zero proc only when a new value is found that hasn't been guessed, and only to the zero proc
        comm.send(trial_n, dest=0)

    # now, proc 0 (and only proc 0) will receive any of the new values
    if rank==0:
        new_vals = []
        # we loop over all the procs that are not proc 0, receive any incoming integers and then add them to the new_vals
        for i in range(size-1):

            trial_n = comm.recv(source=i+1)
            tprint(f'rank {rank} recieved {trial_n}')
            sys.stdout.flush()
            new_vals.append(trial_n)

        IFS = final_list
        final_list =list(set((final_list + new_vals)))

        if IFS != final_list:
            tprint(f'new list {final_list}')
            sys.stdout.flush()

comm.barrier()
if rank !=0:
    tprint(f'rank {rank} got {j} points with {i} guesses')