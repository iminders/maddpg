# -*- coding:utf-8 -*-
import pickle
import zlib

import zmq


def serve(host="127.0.0.1", port=7902):
    c = zmq.Context()
    s = c.socket(zmq.REP)
    s.bind('tcp://%s:%d' % (host, port))
    while True:
        z = s.recv_pyobj()
        p = zlib.decompress(z)
        data = pickle.loads(p)
        print("server get data", data.shape)
        s.send_pyobj({'success': True})
    s.close()


if __name__ == '__main__':
    serve()
