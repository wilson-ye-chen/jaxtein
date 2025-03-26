from setuptools import setup

setup(
    name='jaxtein',
    version='0.3.1',
    description='Stein Thinning for JAX',
    url='https://github.com/wilson-ye-chen/jaxtein',
    author='Wilson Ye Chen',
    license='MIT',
    packages=['jaxtein'],
    install_requires=['qpsolvers', 'numpy', 'jax', 'tqdm']
    )
