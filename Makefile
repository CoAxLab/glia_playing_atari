SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/glia_playing_atari/data/
DATA_PATH=/home/stitch/Code/glia_playing_atari/data/

# ----------------------------------------------------------------------------
xor_exp1:
	glia_xor.py --glia=False

xor_exp2:
	glia_xor.py --glia=True

# ----------------------------------------------------------------------------
# low N epoch. SOA doesn't matter
digits_exp1:
	glia_digits.py --glia=False --epochs=3 --progress=True

digits_exp2:
	glia_digits.py --glia=True --epochs=10 --progress=True

# Change to ELU non lin
# c512575b734cca2c23cd8854ea6dfb6fbb825196
# Acc: 0.53, but started declining by epoch 15 or so
digits_exp3:
	glia_digits.py --glia=True --epochs=100 --progress=True --lr=0.1

# Acc: 0.49 (linear improvements)
digits_exp4:
	glia_digits.py --glia=True --epochs=100 --progress=True --lr=0.01

# Back to Tanh
# 7d649590227b657d2ab2ca3902ad2621ed158c5a
# Acc: 0.59 (linear improvements)
digits_exp5:
	glia_digits.py --glia=True --epochs=100 --progress=True --lr=0.01

# Increase epochs
# Acc 86
digits_exp6:
	glia_digits.py --glia=True --epochs=200 --progress=True --lr=0.01

# Increase epochs
# Acc: 88.4 (linear improvement, until that last 50 or so. Stuck at ~0.88 after)
digits_exp7:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.01

# Increase lr, slightly
# Learning happens much earlier on now. Seems to be progressing more 
# rapidly.
# About 100 epocs in learning may have peaked near 0.77. Acc starting to 
# decline and have higher variance after that. Volatility suggests lr to high?
# How long should I let this go before calling it. Acc dropped below 50. 
# Stopped early.
digits_exp8:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.015

# Bump lr up a little (in between 7 and 8)
# Learning stalled at 0.49. I don't get this at all!
digits_exp9:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.0125

# Re-run 7, using ADAM instead of SGD.
# 4ea82d4730139bd7c8d033ffc6325868009d46af
# First test iter w/ ADAM already at 0.53!
# But variance became high-er-ish. ~0.7 250 epochs in 
digits_exp10:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.01

# Decrease lr 
# Acc: 0.93, stable linear though slowing by ~250.
digits_exp11:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005

# exp11 w/ cuda
digits_exp12:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_cuda=True


# ----------------------------------------------------------------------------
# Perceptron exps/testing
# (using exp11 w/ cuda params)
#
# d36e5372e6252269ecd45a38abc80bd57c2f0e7f
# Neurons
digits_exp13:
	glia_digits.py --glia=False --epochs=300 --progress=True --lr=0.005 --use_cuda=True --device_num=0 --conv=False

# Glia (seg faults on my laptop)
# Using: a2f1e10ee6793a28d03679e3ec39eaec3c107520
# Learning to 70% Acc is fast (Epoch 2); 78% by Epoch 3; 
# Peaked at Epoch 50 89%.
digits_exp14:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_cuda=True --device_num=0 --conv=False --debug=False

# ----------------------------------------------------------------------------
# VAE testing
# CPU
digits_exp15:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_cuda=False --device_num=0 --debug=True | tee $(DATA_PATH)/digits_exp15.log

# w/ glia
digits_exp16:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.01 --use_cuda=True --device_num=0 --debug=False | tee $(DATA_PATH)/digits_exp16.log

# w/ neurons
digits_exp17:
	glia_digits.py --glia=False --epochs=300 --progress=True --lr=0.005 --use_cuda=True --device_num=0 --debug=False | tee $(DATA_PATH)/digits_exp17.log

# ----------------------------------------------------------------------------
# Hyper params sweep 1 for AGN
# Setup to run on Churchlands
# exp1 ran fine, but not great results
tune_digits_exp1:
	tune_digits.py tune_1 data/exp1/ --num_samples=50 --max_iterations=20 --use_cuda=True

# tweaked config 
# up samples 
# 7dae017059c9b976da462d55cd85aa579b33bfaa
tune_digits_exp2:
	tune_digits.py tune_1 data/exp2/ --num_samples=100 --max_iterations=20 --use_cuda=True

# Froze VAE lr
# fa11ee7a53df05b52167b859370a0824b7de6ff2
tune_digits_exp3:
	tune_digits.py tune_1 data/exp3/ --num_samples=200 --max_iterations=20 --use_cuda=True

# Moved to py36 to get tensoprboard working. Can't easily vis the above.
# Had to parse logs. Booos. Rerun exp3, now w/ TB?
# fa11ee7a53df05b52167b859370a0824b7de6ff2
tune_digits_exp4:
	tune_digits.py tune_1 data/exp4/ --num_samples=200 --max_iterations=20 --use_cuda=True

# 12-10-2018
# Prev tunes were totally usefless. Somehow I forked the wrong code so none
# of the hyperparams that were suppose to be run, were. Only lr was sampled
# in any of the above. That error is now bieng fixed....
# e10df71dfdf3ae020e72384e566fbd26669c034b
#
# For laptop:
tune_digits_exp5:
	tune_digits.py tune_1 data/exp5/ --num_samples=2 --max_iterations=2 --use_cuda=False

# Test VAEGather
# e10df71dfdf3ae020e72384e566fbd26669c034b
tune_digits_exp6:
	tune_digits.py tune_1 data/exp6/ --num_samples=100 --max_iterations=20 --use_cuda=True

# Test LinearGather
# 8d54eb5c4385c0c7d5bd33db84b83c431b19c36a
# Errored: debug later.
tune_digits_exp7:
	tune_digits.py tune_1 data/exp7/ --num_samples=100 --max_iterations=20 --use_cuda=True

# Test VAESlide
# 5e3286b12b87d26615c04a966c48f5476c8531f0
# Notinhg above 45%. Best models had < 5 layers, used ELU or Tanh
tune_digits_exp8:
	tune_digits.py tune_1 data/exp8/ --num_samples=100 --max_iterations=20 --use_cuda=True

# Test VAESlide
# Repeat exp8 but sample inline w/ those results,
# and search the epsilon param in ADAM.
# num_hidden 1-5
# lr 0.005-0.1
# Fix ELU
# epsilon 1e-8 - .1 (Huge per tensorflow rec)
# dcdf500fe7609056b80efab8f16bcfec83615d37
tune_digits_exp9:
	tune_digits.py tune_1 data/exp9/ --num_samples=200 --max_iterations=20 --use_cuda=True


# VAESpread: search num_hidden and lr
# f09b166ba5640613d4b63b0a705e1b61ae98e555
# This gave a bump in final ACC to ~0.51, the largest increase for HP so far.
# Best model: lr=0.0097, num_hidden=2
tune_digits_exp10:
	tune_digits.py tune_1 data/exp10/ --num_samples=200 --max_iterations=20 --use_cuda=True


# ---------------------------------------------------------------------------
# 5-6-2019
# 7dd363c757700feb81647b4aa213b574401d9e66
# Glia comp is not analogoues to point source intiation followed by a circular
# Ca traveling wave, than eventually gets summarized/decoded to digits.

digits_exp18:
	glia_digits.py --glia=False --epochs=3 --progress=True

digits_exp19:
	glia_digits.py --glia=True --epochs=100 --progress=True
