from threading import Thread, Lock
import socket
import os
import csv
import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from pathlib import Path
import sys
import base64
from multiprocessing import Queue, Value
from ctypes import c_bool
from datetime import datetime, timezone
from time import time, sleep
import psutil
import base64
import matplotlib.pyplot as plt
import json

# Configurações
MIN_ERROR = .0001
MAX_WORKERS = 8

MODEL_SHAPES = {
    '30x30': (27904, 900),
    '60x60': (50816, 3600)
}

ACTUAL_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

IMAGE_SHAPES = {
    '30x30': (900, 1),
    '60x60': (3600, 1)
}

FINAL_IMAGE_SHAPE = {
    '30x30': (30, 30),
    '60x60': (60, 60)
}

def sgemm(alpha, a, b, trans_a=False):
    if trans_a:
        a = a.T
    return alpha * np.dot(a, b)

def calc_error(b, a):
    return norm(b, 2) - norm(a, 2)

def cgne(h, g, image_shape, final_image_shape):
    f0 = np.zeros(image_shape, np.float32)
    r0 = g - sgemm(1.0, h, f0)
    p0 = sgemm(1.0, h, r0, trans_a=True)

    total_iterations = 0

    while total_iterations < 30:
        total_iterations += 1

        a0 = sgemm(1.0, r0, r0, trans_a=True) / sgemm(1.0, p0, p0, trans_a=True)
        f0 = f0 + a0 * p0
        r1 = r0 - a0 * sgemm(1.0, h, p0)

        error = calc_error(r1, r0)
        if error < MIN_ERROR:
            break

        beta = sgemm(1.0, r1, r1, trans_a=True) / sgemm(1.0, r0, r0, trans_a=True)
        p0 = sgemm(1.0, h, r0, trans_a=True) + beta * p0
        r0 = r1

    f0 = f0.reshape(final_image_shape)
    return f0, total_iterations

def cgnr(h, g, image_shape, reshaped_image_shape):
    f0 = np.zeros(image_shape, np.float32)
    r0 = g - sgemm(1.0, h, f0)
    z0 = sgemm(1.0, h, r0, trans_a=True)
    p0 = np.copy(z0)

    total_iterations = 0
    while total_iterations < 30:
        # Count iterations
        total_iterations += 1

        w = sgemm(1.0, h, p0)
        norm_z = norm(z0, 2) ** 2
        a = norm_z / norm(w) ** 2
        f0 = f0 + a * p0
        r1 = r0 - a * w

        error = abs(calc_error(r1, r0))
        if error < MIN_ERROR:
            break

        z0 = sgemm(1.0, h, r1, trans_a=True)
        b = norm(z0, 2) ** 2 / norm_z
        p0 = z0 + b * p0
        r0 = r1

    f0 = f0.reshape(reshaped_image_shape)

    return f0, total_iterations

def create_image(username):
    path = ACTUAL_DIR / "images" / username    
    if not path.exists():
        os.mkdir(path)

def read_model(model):
    with open(ACTUAL_DIR / "models" / f"model-{model}.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        res = np.empty(MODEL_SHAPES[model], dtype=np.float32)
        for i, line in enumerate(reader):
            res[i] = np.array(line, np.float32)
        return res
    
clients = []
images_64 = {}

ALGORITHM = {
    'cgne': cgne,
    'cgnr': cgnr
}

def read_and_code(file_path, filename):
    with open(file_path, 'rb') as f:
        converted = base64.b64encode(f.read()).decode('utf-8')
        images_64[filename] = converted

def bytes_to_mb(value_bytes):
    return value_bytes / (1024 ** 2)

def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu = process.cpu_percent(interval=None)
    mem = bytes_to_mb(process.memory_info().rss)

    # cpu = bytes_to_gigas(psutil.virtual_memory().used)
    # mem = psutil.virtual_memory().percent
    return cpu, mem

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

class Relatorio:
    def __init__(self):
        curr_time = time()
        self.images = CSV(ACTUAL_DIR / "relatorio" / f"imagens-relatorio_{curr_time}.csv")
        self.performance = CSV(ACTUAL_DIR / "relatorio" / f"performance-relatorio_{curr_time}.csv")

        self.images.write(["Username", "Image name", "Algorithm", "Model type", "Iterations", "Reconstruction time"])
        self.performance.write(["Measured at", "CPU usage", "Memory usage"])

    def close(self):
        self.images.close()
        self.performance.close()

class Worker:
    def __init__(self, id, queue):
        self.id = id
        self.queue = queue

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

def run_profiler(close_profiler_worker, server_data):
    # "aquecimento" inicial do cpu_percent
    process = psutil.Process(os.getpid())
    process.cpu_percent(None)
    sleep(0.1)

    while not close_profiler_worker.value:
        start = time()

        measured_at = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # mede uso de CPU desde a última chamada
        cpu, mem = get_resource_usage()
       
        server_data.reports.performance.write([measured_at, f"{cpu}%", f"{mem: .2f} MB"])
        server_data.reports.performance.flush()

        # espera até completar 1 segundo total de loop
        elapsed = time() - start
        if elapsed < 1:
            sleep(1 - elapsed)

def run_queue_worker(request_queue, server_data):
    worker_queue = Queue(MAX_WORKERS)
    for i in range(MAX_WORKERS):
        worker_queue.put(i)

    while True:
        worker_id = worker_queue.get()
        worker = Worker(worker_id, worker_queue)  # item, fila

        request_data = request_queue.get()

        worker_thread = Thread(target=run_worker, args=[ worker, server_data, request_data])
        worker_thread.start()


def run_worker(worker, server_data, request_data):
    print(f"[worker-{worker.id}] Started")

    username = request_data.username
    model_type = request_data.model
    signal = request_data.signal

    model = server_data.models[model_type]
    
    image_shape = IMAGE_SHAPES[model_type]
    reshaped_image_shape = FINAL_IMAGE_SHAPE[model_type]

    initial_time = time()

    img, iterations = ALGORITHM[request_data.algorithm](
        model,
        signal,
        image_shape,
        reshaped_image_shape
    )

    final_time = time()
    elapsed_time = final_time - initial_time

    filename = f"{username}-final-{final_time}.png"
    filepath = ACTUAL_DIR / "images" / username / filename
    started_at = datetime.fromtimestamp(initial_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ended_at = datetime.fromtimestamp(final_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

     # Save image
    metadata = {
        'Title': filename.replace(".png", ""),
        'Author': f"CGNR Processor",
        'Description': f"Username: {username} | Algorithm: {request_data.algorithm.upper()} | Started at: {started_at} | Ended at: {ended_at} | "
                       f"Size: {FINAL_IMAGE_SHAPE[model_type]} | Iterations: {iterations}"
    }
    plt.imsave(filepath, img, cmap='gray', metadata=metadata)

    # Save to images report
    server_data.reports.images.write([
        username,
        filename,
        request_data.algorithm.upper(),
        model_type,
        iterations,
        elapsed_time
    ])
    server_data.reports.images.flush()

    read_and_code(filepath, filename)

    print(f"[worker-{worker.id}] Executed in {elapsed_time}s")

    worker.queue.put(worker.id)

def handle_client(client, addr, request_queue):
    print(f"[NOVA CONEXÃO] {addr} conectado")

    connected = True
    recebendo = False
    buffer = b''
    username = None

    while connected:
        try:
            chunk = client.recv(100000)
            if not chunk:
                break

            if chunk.startswith(b'EXIT'):
                connected = False

            if recebendo:
                buffer += chunk

                if b'fim' in buffer:
                    json_bytes, _, buffer = buffer.partition(b'fim')
                    data = json.loads(json_bytes.decode())
                    signal = np.array(data['signal'], np.float32).reshape((-1, 1))

                    request_data = RequestData(username, data['algorithm'], data['model'], signal)
                    request_queue.put(request_data)

                    recebendo = False  # pronto para próxima mensagem
                    buffer = b''

                    create_image(username)
                    continue

            if chunk.startswith(b'1_|'):
                parts = chunk.decode().split('|', 2)
                username = parts[1]
                # buffer = parts[2].encode()  # início do JSON
                recebendo = True

            elif chunk.startswith(b'2_'):
                parts = chunk.decode().split(':')
                username = parts[1]

                # converte os bytes para strings
                dict_user = {key: value for key, value in images_64.items() if username in key}

                # envia um JSON completo
                message = {
                    "type": "2_",
                    "username": username,
                    "payload": dict_user
                }

                client.send(json.dumps(message).encode())

        except ConnectionResetError:
            print(f"[DESCONECTADO] {addr} encerrou a conexão.")
            break

    client.close()

def main():
    global clients

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(('localhost', 7776))
        server.listen(5)
        print('Servidor iniciado e aguardando conexões...')
    except Exception as e:
        print(f'\nErro ao iniciar o servidor: {e}\n')
        return


    models = {
        "30x30": read_model("30x30"),
        "60x60": read_model("60x60")
    }

    request_queue = Queue()
    close_profiler_worker = Value(c_bool)

    reports = Relatorio()
    server_data = ServerData(reports, models)

    profiler_worker = Thread(target=run_profiler, args=[close_profiler_worker, server_data])
    profiler_worker.start()

    
    queue_worker = Thread(target=run_queue_worker, args=[ request_queue, server_data])
    queue_worker.start()


    while True:
        client, addr = server.accept()
        clients.append(client)
        thread = Thread(target=handle_client, args=[client, addr, request_queue])
        thread.start()

if __name__ == "__main__":
    main()
