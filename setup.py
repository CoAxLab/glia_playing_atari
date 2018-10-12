from setuptools import setup

setup(
    name='gliafun',
    version='0.0.1',
    description="Glia computers!",
    url='',
    author='Erik J. Peterson',
    author_email='erik.exists@gmail.com',
    license='MIT',
    packages=['gliafun'],
    scripts=['gliafun/exp/glia_vision.py', 'gliafun/exp/glia_xor.py'],
    zip_safe=False)
