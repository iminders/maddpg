# -*- coding:utf-8 -*-
import pickle
import zlib
from multiprocessing import Process

import numpy as np
import zmq


def handle(id, host="127.0.0.1", port=7902):
    c = zmq.Context()
    s = c.socket(zmq.REQ)
    s.connect('tcp://%s:%d' % (host, port))
    i = 0
    while i < 100:
        msg = np.asarray([id, i] * i)
        p = pickle.dumps(msg)
        z = zlib.compress(p)
        s.send_pyobj(z)
        msg = s.recv_pyobj()
        print("client[%d] recv %dth msg:" % (id, i), msg)
        i += 1


if __name__ == '__main__':
    processes = []
    for i in range(5):
        p = Process(target=handle, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
