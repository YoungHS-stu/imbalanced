from multiprocessing import Process, Queue
import multiprocessing
import time
import math
def do_sum(q,l):
    q.put(sum(l))

def main():
    start_time = time.time()
    my_list = range(100000000)

    q = Queue()

    p1 = Process(target=do_sum, args=(q, my_list[:25000000]))
    p2 = Process(target=do_sum, args=(q, my_list[25000001:50000000]))
    p3 = Process(target=do_sum, args=(q, my_list[50000001:75000000]))
    p4 = Process(target=do_sum, args=(q, my_list[75000001:]))
    
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    
    r1 = q.get()
    r2 = q.get()
    r3 = q.get()
    r4 = q.get()
    
    print(r1+r2+r3+r4)
    end_time = time.time()
    print("time after improvement: {}".format(end_time-start_time))

def old_main():
    start_time = time.time()
    my_list = range(100000000)
    print(sum(my_list))
    end_time = time.time()
    print("time after improvement: {}".format(end_time-start_time))

def func2(args):  # multiple parameters (arguments)
    # x, y = args
    x = args[0]  # write in this way, easier to locate errors
    y = args[1]  # write in this way, easier to locate errors

    time.sleep(1)  # pretend it is a time-consuming operation
    return x - y


def run__pool():  # main process
    from multiprocessing import Pool

    cpu_worker_num = 1
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]

    print(f'| inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func2, process_args)
    print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

def worker(input):
    time.sleep(input+1)
    print(f'input: {input} finished')
    return input

if __name__=='__main__':
    # main()
    pool = multiprocessing.Pool(20)
    queue = multiprocessing.Queue()
    for i in range(20):
        res = pool.apply_async(worker, (i,))
    
    pool.close()
    
    pool.join()
    
