from threading import Thread, Lock
import threading
import socket
import time
import hashlib
import os
import csv
from scipy.linalg.blas import sgemm
import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from pathlib import Path
import sys
import base64
import asyncio

# import index from index.html

MIN_ERROR = .0001
MAX_WORKERS = 8


Model = NDArray[np.float32]

MODEL_SHAPES = {
    '30x30': (27904, 900),
    '60x60': (50816, 3600)
}


ACTUAL_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

clients = []
images_64 = {}

def calc_error(b: NDArray[np.float32], a: NDArray[np.float32]):
    return norm(b, 2) - norm(a, 2)

# class IMG:
#     def __init__(self, filename) -> None:
#         self.__file = open(filename, 'w', encoding='UTF-8')
#         self.__writer = csv.writer(self.__file)
#         self.__file.close()
#         self.__lock = Lock()  #Lock evita conflitos

#     def write(self, row: Iterable[any]):
#         if self.__file.closed:
#             return

#         with self.__lock:
#             self.__writer.writerow(row)

#     def flush(self):
#         if self.__file.closed:
#             return

#         with self.__lock:
#             self.__file.flush()

#     def close(self):
#         if self.__file.closed:
#             return

#         self.__file.close()

def handle_client(client, addr):
    print(f"[NEW CONNECTION] {addr} connected ")

    connected = True
    while connected:
        try:
            data = client.recv(100000)
            if not data:
                break

            if data.startswith(b'EXIT'):
                connected = False

            elif data.startswith(b'1_'):
                parts = data.decode().split(':')
                username = parts[1]
                res = asyncio.run(send_images(username)) 
                client.send(f'1_:{username}:{res}'.encode())

        except ConnectionResetError:
            print(f"[DISCONNECTED] {addr} encerrou a conexão.")
            break

    client.close()

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

# def cgnr(h: Model, g: Signal, image_shape, reshaped_image_shape) -> tuple[Image, int]:
#     f0 = np.zeros(image_shape, np.float32)
#     r0 = g - sgemm(1.0, h, f0)
#     z0 = sgemm(1.0, h, r0, trans_a=True)
#     p0 = np.copy(z0)

#     total_iterations = 0
#     while total_iterations < 30:
#         # Count iterations
#         total_iterations += 1

#         w = sgemm(1.0, h, p0)
#         norm_z = norm(z0, 2) ** 2
#         a = norm_z / norm(w) ** 2
#         f0 = f0 + a * p0
#         r1 = r0 - a * w

#         error = abs(calc_error(r1, r0))
#         if error < MIN_ERROR:
#             break

#         z0 = sgemm(1.0, h, r1, trans_a=True)
#         b = norm(z0, 2) ** 2 / norm_z
#         p0 = z0 + b * p0
#         r0 = r1

#     f0 = f0.reshape(reshaped_image_shape)

#     return f0, total_iterations

def read_model(model: str) -> Model:
    with open(ACTUAL_DIR / "models" / f"model-{model}.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        res = np.empty(MODEL_SHAPES[model], dtype=np.float32)
        i = 0
        for line in reader:
            res[i] = np.array(line, np.float32)
            i += 1
        return res

def read_and_code(file_path, filename):
    with open(file_path, 'rb') as f:
        converted = base64.b64encode(f.read())
        images_64[filename] = converted


async def send_images(username: str):
        dict_user = {}
        for key, value in images_64.items():
            if username in key:
                dict_user[key] = value

        return dict_user


def main():
    global clients

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        server.bind(('localhost', 7776))
        server.listen(5)
        print('Servidor iniciado')
    except:
        return print('\nNão foi possível iniciar o servidor!\n')
    
    models = {
        "30x30": read_model("30x30"),
        "60x60": read_model("60x60")
    }


    while True:
        client, addr = server.accept()
        clients.append(client)

        thread = Thread(target=handle_client, args=[client, addr])
        thread.start()


main()
