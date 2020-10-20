# -*- coding:utf-8 -*-

from multiprocessing import Process

from client import handle
from server import serve

if __name__ == '__main__':
    processes = []
    host = "127.0.0.1"
    port = 7902
    p = Process(target=serve, args=(host, port))
    p.start()
    processes.append(p)
    for i in range(5):
        p = Process(target=handle, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
