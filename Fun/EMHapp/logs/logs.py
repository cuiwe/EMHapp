# -*- coding:utf-8 -*-
# @Time    : 2022/2/14
# @Author  : cuiwei
# @File    : logs.py
# @Software: PyCharm
# @Script to:
#   -生成log文件

import logging
from logging.handlers import RotatingFileHandler


def set_log_file(output_format, fname, overwrite=False):
    for h in list(logging.getLogger("py.warnings").handlers):
        # only remove our handlers (get along nicely with nose)
        if isinstance(h, RotatingFileHandler):
            h.close()
            logging.getLogger("py.warnings").removeHandler(h)
    mode = 'w' if overwrite else 'a'
    logger_file_handler = RotatingFileHandler(fname, mode=mode)
    logger_file_handler.setLevel(logging.DEBUG)
    logger_file_handler.setFormatter(logging.Formatter(output_format))
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(logger_file_handler)
    logging.captureWarnings(True)
