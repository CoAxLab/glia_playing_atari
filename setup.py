from setuptools import setup

setup(
    name='glia',
    version='0.0.1',
    description="Glia computers!",
    url='',
    author='Erik J. Peterson',
    author_email='erik.exists@gmail.com',
    license='MIT',
    packages=['glia'],
    scripts=[
        'glia/exp/glia_digits.py', 'glia/exp/glia_xor.py',
        'glia/exp/tune_digits.py'
    ],
    zip_safe=False)
