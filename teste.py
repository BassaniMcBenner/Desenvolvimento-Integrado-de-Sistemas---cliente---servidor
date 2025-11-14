import psutil
import os


process = psutil.Process(os.getpid())
cpu = process.cpu_percent(interval=None)

while True:
    x = 10
    y = 10000
    z = 0
    count = 1
    for z in range(y):
        count = count * 10
        # print('count: ', count)
        print(cpu)
