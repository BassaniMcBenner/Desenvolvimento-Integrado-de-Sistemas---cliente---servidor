from multiprocessing import Process, Queue
from time import sleep
import queue
import os
from threading import Thread, Lock
import csv
import time
from pathlib import Path
import sys


# ACTUAL_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

MAX_WORKERS = 8
class CSV:
    def __init__(self, filename):
        self.__file = open(filename, 'w', encoding='UTF-8')
        self.__writer = csv.writer(self.__file)
        self.__lock = Lock()

    def write(self, row):
        if self.__file.closed:
            return

        with self.__lock:
            self.__writer.writerow(row)

    def flush(self):      #força a gravação para o disco
        if self.__file.closed:
            return
        
        with self.__lock:
            self.__file.flush()

    def close(self):
        if self.__file.closed:
            return
                    
        self.__file.close()

class Worker:
    def __init__(self, queue, id):
        self.queue = queue
        self.id = id

class ServerData:
    def __init__(self, reports, models):
        self.reports = reports
        self.models = models

class RequestData:
    def __init__(self, username, algorithm, model, signal):
        self.username = username
        self.algorithm = algorithm
        self.model = model
        self.signal = signal

class Relatorio:
    def __init__(self):
        curr_time = time.time()
        # self.images = CSV(ACTUAL_DIR / "relatorio" / f"imagens-relatorio_{curr_time}.csv")
        # self.performance = CSV(ACTUAL_DIR / "relatorio" / f"performance-relatorio_{curr_time}.csv")

        # self.images.write(["Username", "Image name", "Algorithm", "Model type", "Iterations", "Reconstruction time"])
        # self.performance.write(["Measured at", "CPU usage", "Memory usage"])

    # def close(self):
    #     self.images.close()
    #     self.performance.close()


def calc_square(numbers, q):
    for n in numbers:
        try:
            q.put_nowait(n*n)
        except queue.Full:
            print("Fila cheia! Não foi possível inserir:", n*n)

def run_queue_worker(request_queue, server_data):
    worker_queue = Queue(MAX_WORKERS)
    for i in range(MAX_WORKERS):
        worker_queue.put(i)

    # while not worker_queue.empty():
    #     print('workers: ', worker_queue.get())


    while True:
        worker_id = worker_queue.get()
        worker = Worker(worker_queue, worker_id)  # item, fila

        print(f'worker{worker_id}:: ')
        print("worker.queue: ", worker.queue, "     worker.id: ", worker.id)

        # request_data = request_queue.get()

        # worker_thread = Thread(target=run_worker, args=[ worker, server_data, request_data])
        # worker_thread.start()

        


if __name__ == '__main__':
    numbers = [2,3,5]
    # q = Queue(2)
    # p = Process(target=calc_square, args=(numbers, q))

    # p.start()
    # p.join()


    # while not q.empty():
    #     print(q.get())

    reports = Relatorio()

    models = {
        "30x30": "30x30",
        "60x60": "60x60"
    }

    request_queue = Queue()
    server_data = ServerData(reports, models)

    run_queue_worker(request_queue, server_data)










    #  joel = Process(target=calc_square, args=(numbers,q))

    # marcos = Process(target=calc_square, args=(numbers,q))

    # davi = Process(target=calc_square, args=(numbers,q))


    # joel.start()
    # marcos.start()
    # davi.start()