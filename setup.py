# -*- coding:utf-8 -*-

import codecs
import os

from setuptools import find_packages, setup


def read(fname):
    return codecs.open(os.path.join(
        os.path.dirname(__file__), fname)).read().strip()


def read_install_requires():
    reqs = [
            'gym==0.10.5',
            'minio',
            'pyzmq',
            'tensorflow>=2.3.1',
            'tensorboard==2.3.0']
    return reqs


setup(name='maddpg',
      version=read('maddpg/VERSION.txt'),
      description='',
      url='https://github.com/iminders/maddpg',
      author='liuwen',
      author_email='liuwen.w@qq.com',
      long_description_content_type="text/markdown",
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
      python_requires='>=3',
      install_requires=read_install_requires(),
      package_data={'': ['*.csv', '*.txt']},
      )
