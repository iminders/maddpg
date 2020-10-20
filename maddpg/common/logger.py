# -*- coding:utf-8 -*-

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')

logger = logging.getLogger()

logger_dir = os.getenv("MADDPG_LOG_DIR")
if logger_dir is None or logger_dir == "":
    logger_dir = os.path.join("/tmp", "maddpg")
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)

handler = logging.FileHandler(os.path.join(logger_dir, "run.log"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
