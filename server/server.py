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
import random
from PIL import Image
import gc
import io

file_lock = Lock()       # para logs / csv
send_lock = Lock()       # para enviar mensagens no socket
print_lock = Lock()      # opcional: evita prints embaralhados


ACTUAL_DIR = Path(os.path.dirname(os.path.abspath(sys.argv[0])))

MIN_ERROR = .0001
MAX_WORKERS = 8

MODEL_SHAPES = {
    '30x30': (27904, 900),
    '60x60': (50816, 3600)
}

IMAGE_SHAPES = {
    '30x30': (900, 1),
    '60x60': (3600, 1)
}

FINAL_IMAGE_SHAPE = {
    '30x30': (30, 30),
    '60x60': (60, 60)
}
def reconstruct_cgnr(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=5e-3, min_iterations=10, lambda_reg: float = 0.0, logger=None) -> tuple:
    m, n = H.shape
    f = np.zeros((n, 1))
    g = g.reshape(-1, 1)
    r = g - H @ f
    z = H.T @ r
    p = z.copy()
    initial_residual_norm = np.linalg.norm(r)
    min_div = 1e-12
    number_iterations = 0

    if logger is not None:
        logger.info(f"CGNR: tol={tol:.3e}")

    for i in range(max_iterations):
        w = H @ p

        # >>> Correção dos warnings
        z_dot = (z.T @ z).item()
        w_dot = (w.T @ w).item() + min_div
        # <<<

        alpha = z_dot / w_dot

        f_new = f + alpha * p
        r_new = r - alpha * w
        z_new = H.T @ r_new

        # >>> Correção dos warnings
        z_new_dot = (z_new.T @ z_new).item()
        # <<<

        beta = z_new_dot / (z_dot + min_div)
        p_new = z_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(
                f"Iteracao {i + 1}: erro relativo = {relative_error:.6e}, residuo = {current_residual_norm:.3e}"
            )

        f, r, z, p = f_new, r_new, z_new, p_new
        number_iterations = i + 1

        if number_iterations >= min_iterations and relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            break

    final_residual = g - H @ f
    final_error = np.linalg.norm(final_residual) / (np.linalg.norm(g) + min_div)
    return f.flatten(), number_iterations, final_error

def reconstruct_cgne(H: np.ndarray, g: np.ndarray, max_iterations: int, tol=1e-6, min_iterations=10, reg_factor: float = 0.0, logger=None) -> tuple[np.ndarray, int, float]:
    N = H.shape[1]
    f = np.zeros((N, 1))
    g = g.reshape(-1, 1)
    r = g - H @ f
    p = H.T @ r
    initial_residual_norm = np.linalg.norm(r)
    min_div = 1e-12
    final_iterations = 0

    if logger is not None:
        logger.info(f"CGNE: tol={tol:.3e}")

    for i in range(max_iterations):
        Hp = H @ p

        # >>> Correções dos warnings
        alpha_num = (r.T @ r).item()
        alpha_den = (Hp.T @ Hp).item() + min_div
        # <<<

        if alpha_den < min_div:
            break

        alpha = alpha_num / alpha_den
        f_new = f + alpha * p
        r_new = r - alpha * (H @ p)

        # >>> Correções dos warnings
        beta_num = (r_new.T @ r_new).item()
        beta_den = (r.T @ r).item() + min_div
        # <<<

        beta = beta_num / beta_den
        p_new = H.T @ r_new + beta * p

        current_residual_norm = np.linalg.norm(r_new)
        relative_error = current_residual_norm / (initial_residual_norm + min_div)

        if logger is not None:
            logger.info(f"Iteracao {i + 1}: erro relativo = {relative_error:.6e}")

        f, r, p = f_new, r_new, p_new
        final_iterations = i + 1

        if final_iterations >= min_iterations and relative_error < tol:
            if logger is not None:
                logger.info(f"Convergiu com erro relativo {relative_error:.2e} < {tol:.2e}")
            break

    final_error = np.linalg.norm(g - H @ f) / (np.linalg.norm(g) + min_div)
    return f.flatten(), final_iterations, final_error

def create_pasta(username):
    path = ACTUAL_DIR / "images" / username    
    if not path.exists():
        os.mkdir(path)

ALGORITHM = {
    'cgne': reconstruct_cgne,
    'cgnr': reconstruct_cgnr
}

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

class Relatorio:
    def __init__(self):
        curr_time = time()
        self.images = CSV(ACTUAL_DIR / "relatorio" / f"imagens-relatorio_{curr_time}.csv")
        self.performance = CSV(ACTUAL_DIR / "relatorio" / f"performance-relatorio_{curr_time}.csv")

        self.images.write(["Username   ", "Image name   ", "Algorithm   ", "Model type   ", "Iterations   ", "Reconstruction time   "])
        self.performance.write(["Measured at   ", "CPU usage   ", "Memory usage"])

class ServerData:
    def __init__(self, reports, models):
        self.reports = reports
        self.models = models

def apply_signal_gain(g_vector: np.ndarray):
    S = len(g_vector); g_out = g_vector.copy().astype(np.float32)
    for l in range(S): g_out[l] *= (100.0 + (1.0/20.0)*(l+1)*np.sqrt(l+1))
    return g_out

def get_dynamic_mem_limit():
    # Limite: 80% da RAM total, mas sempre deixa pelo menos 1GB livre
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    if total_gb <= 2:
        return 35.0   # Em PCs com pouca RAM, seja mais conservador
    # 80% do total, mas nunca usar mais que total-1GB
    # max_percent = 100.0 - (1.0 / total_gb) * 100.0
    # max_percent = 100 - (1/16) * 100 = 93.75%

    return min(42.0, 44.0)
    
def get_dynamic_cpu_limit():
    # Limite: 80% dos núcleos lógicos
    n_cores = psutil.cpu_count(logical=True)
    # return max(50.0, min(90.0, n_cores * 80.0 / n_cores))  # 80% (ajustável)
    return 50
def get_percent_virtual_memory(close_profiler_worker, server_data):
    while not close_profiler_worker.value:
        start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cpu_percent = psutil.cpu_percent(interval=0.5)

        mem = psutil.virtual_memory()
        mem_percent = mem.percent
       
        server_data.reports.performance.write([start_dt, f"    {cpu_percent}%", f"    {mem_percent} %"])
        server_data.reports.performance.flush()

        sleep(0.5)

def run_queue_worker(request_queue):
    while True:
        # checar limites
        cpu_limit = get_dynamic_cpu_limit()
        mem_limit = get_dynamic_mem_limit()

        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem_percent = psutil.virtual_memory().percent

        data = request_queue.get()   # pega o item da fila
        if data is None:
            break  # comando de shutdown

        username = data["username"]
        idx = data["idx"]

        # respeitar tempo mínimo para retry
        now = time()
        next_try = data.get("next_try", 0)
        if now < next_try:
            request_queue.put(data)
            sleep(0.5)
            continue

        print(f"[WORKER] CPU={cpu_percent:.1f}% (lim={cpu_limit}%) | "
              f"RAM={mem_percent:.1f}% (lim={mem_limit}%)")

        if cpu_percent > cpu_limit or mem_percent > mem_limit:
            print(f"[WORKER] Recursos altos — requeue >> {username} idx={idx}")

            retry = data.get("retry", 0)
            retry = min(retry + 1, 5)
            data["retry"] = retry
            data["next_try"] = time() + retry  # backoff

            print(f"[WORKER]: tempo de espera ")

            request_queue.put(data)
            continue

        print(f"[WORKER] Processando >> {username} idx={idx}")
        process_job(data)

        # reset do retry para próximos jobs sucessivos
        data["retry"] = 0

def process_job(data):
    username = data["username"]
    algorithm = data["algorithm"]
    model = data["model"]
    signal = data["signal"]
    idx = data["idx"]
    client = data["client"]

    start_time = time()
    start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #carrega os dados
    H_matrix = np.loadtxt(model, delimiter=',', dtype=np.float32)
    signal_path = os.path.join("..", signal + ".csv")
    g_vector = np.loadtxt(signal_path, delimiter=",", dtype=np.float32)

    g_processed = apply_signal_gain(g_vector)

    tol_requisito = 1e-4

    if algorithm.upper() == 'CGNR':
        f, iters, final_error = reconstruct_cgnr(H_matrix, g_processed, 5, tol=tol_requisito)
    elif algorithm.upper() == 'CGNE':
        f, iters, final_error = reconstruct_cgne(H_matrix, g_processed, 5, tol=tol_requisito)

    f = f.flatten()
    f_min, f_max = f.min(), f.max()

    #se forem iguais converte tudo para cinza
    if f_max != f_min:
        f_norm = (f - f_min) / (f_max - f_min) * 255
    else:
        f_norm = np.full_like(f, 128)

    #converte o vetor para imagem quadrada
    lado = int(np.sqrt(len(f_norm)))
    imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')
    imagem_array = np.clip(imagem_array, 0, 255)
    imagem = Image.fromarray(imagem_array.astype('uint8'))

    #converte para bytes (PNG)
    img_bytes = io.BytesIO()
    imagem.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    bytes_img = img_bytes.getvalue()

    end_time = time()
    end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    del H_matrix, g_vector, g_processed, f, f_norm, imagem_array, imagem
    # gc.collect()

    # CORREÇÃO: usar bytes_img
    img_b64 = base64.b64encode(bytes_img).decode()


    data_info = {
        "username": username,
        "index": idx,
        "algorithm": algorithm,
        "model": model,
        "signal": signal,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "size": f"{len(img_b64)}",
        "iters": iters,
        "time": end_time - start_time,
    }

    
    mensagem = {
        "type": "2_",
        "payload": {
            "header": data_info,   # CORREÇÃO: enviar o header certo
            "image": img_b64
        }
    }

    # enviar mensagem *com quebra de linha*
    client.send((json.dumps(mensagem) + "\n").encode())
    print(f"[FINALIZADO] Process → {username}  idx -> {idx}")

def handle_client(client, addr, request_queue):
    print(f"[NOVA CONEXÃO] {addr} conectado")

    connected = True
    username = None

    while connected:
        try:
            data = client.recv(1000000)
            if not data:
                break

            if data.startswith(b'EXIT'):
                connected = False

            elif data.startswith(b'2_'):
                parts = data.decode().split('|', 2)
                username = parts[1]
                json_bytes = parts[2].encode()
                
                data_list = json.loads(json_bytes)

                for idx, item in enumerate(data_list):
                    cpu_limit = get_dynamic_cpu_limit()
                    mem_limit = get_dynamic_mem_limit()
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    mem_percent = psutil.virtual_memory().percent

                    print(f"username:: {username}    CPU agora = {cpu_percent:.1f}% (limite={cpu_limit}%)  |  "
                    f"RAM agora = {mem_percent:.1f}% (limite={mem_limit}%)")


                    if cpu_percent > cpu_limit or mem_percent > mem_limit:
                        print('PAUSA::: ', username, 'idx:: ', idx)
                        algorithm = item['algorithm']
                        model = item['model']
                        signal = item['signal']

                        data = {
                            "algorithm": algorithm,
                            "model": model,
                            "signal": signal,
                            "username": username,
                            "idx": idx,
                            "client": client
                        }

                        request_queue.put(data)
                        continue  
                    else:
                        algorithm = item['algorithm']
            
                        model = item['model']
                    
                        signal = item['signal']
            
                        start_time = time()
                        start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        H_matrix = np.loadtxt(model, delimiter=',', dtype=np.float32)
                        signal_path = os.path.join("..", signal + ".csv")
                        g_vector = np.loadtxt(signal_path, delimiter=",", dtype=np.float32)

                        g_processed = apply_signal_gain(g_vector)

                        tol_requisito = 1e-4

                        if algorithm.upper() == 'CGNR':
                            f, iters, final_error = reconstruct_cgnr(H_matrix, g_processed, 5, tol=tol_requisito)
                        elif algorithm.upper() == 'CGNE':
                            f, iters, final_error = reconstruct_cgne(H_matrix, g_processed, 5, tol=tol_requisito)

                        f = f.flatten()
                        f_min, f_max = f.min(), f.max()

                        if f_max != f_min:
                            f_norm = (f - f_min) / (f_max - f_min) * 255
                        else:
                            f_norm = np.full_like(f, 128)

                        lado = int(np.sqrt(len(f_norm)))
                        imagem_array = f_norm[:lado*lado].reshape((lado, lado), order='F')
                        imagem_array = np.clip(imagem_array, 0, 255)
                        imagem = Image.fromarray(imagem_array.astype('uint8'))

                        img_bytes = io.BytesIO()
                        imagem.save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        bytes_img = img_bytes.getvalue()

                        end_time = time()
                        end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        del H_matrix, g_vector, g_processed, f, f_norm, imagem_array, imagem
                        gc.collect()

                        
                        img_b64 = base64.b64encode(bytes_img).decode()
                
                        data_info = {
                            "username": username,
                            "index": idx,
                            "algorithm": algorithm,
                            "model": model,
                            "signal": signal,
                            "start_dt": start_dt,
                            "end_dt": end_dt,
                            "size": f"{len(img_b64)}",
                            "iters": iters,
                            "time": end_time - start_time,
                        }

                    
                        mensagem = {
                            "type": "2_",
                            "payload": {
                                "header": data_info,   # CORREÇÃO: enviar o header certo
                                "image": img_b64
                            }
                        }

                        with send_lock:
                            client.send((json.dumps(mensagem) + "\n").encode())

        except ConnectionResetError:
            print(f"[DESCONECTADO] {addr} encerrou a conexão.")
            break

    client.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(('localhost', 7776))
        server.listen(5)
        print('Servidor iniciado e aguardando conexões...')
    except Exception as e:
        print(f'\nErro ao iniciar o servidor: {e}\n')
        return


    request_queue = Queue()
    close_profiler_worker = Value(c_bool)

    reports = Relatorio()
    server_data = ServerData(reports, None)

    profiler_worker = Thread(target=get_percent_virtual_memory, args=[close_profiler_worker, server_data])
    profiler_worker.start()

    
    queue_worker = Thread(target=run_queue_worker, args=[ request_queue])
    queue_worker.start()

    while True:
        client, addr = server.accept()
        thread = Thread(target=handle_client, args=[client, addr, request_queue])
        thread.start()

if __name__ == "__main__":
    main()
