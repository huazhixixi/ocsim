from setuptools import setup
import setuptools

setup(
    name="ocsim",
    version='0.0.8',
    author='huazhi lun',
    author_email='huazhi.lun@sjtu.edu.cn',
    description="simulator for optical communications",
    url='https://github.com/huazhixixi/ocsim.git',
    packages=setuptools.find_packages(),
    install_requires=['prettytable',' DensityPlot', 'resampy', 'matplotlib',
                      'scipy', 'numpy', 'numba', 'bitarray', 'SciencePlots']
)
