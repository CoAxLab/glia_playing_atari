SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/glia_playing_atari/data/
# DATA_PATH=/Users/qualia/Code/glia_playing_atari/data
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
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_gpu=True


# ----------------------------------------------------------------------------
# Perceptron exps/testing
# (using exp11 w/ cuda params)
#
# d36e5372e6252269ecd45a38abc80bd57c2f0e7f
# Neurons
digits_exp13:
	glia_digits.py --glia=False --epochs=300 --progress=True --lr=0.005 --use_gpu=True --device_num=0 --conv=False

# Glia (seg faults on my laptop)
# Using: a2f1e10ee6793a28d03679e3ec39eaec3c107520
# Learning to 70% Acc is fast (Epoch 2); 78% by Epoch 3; 
# Peaked at Epoch 50 89%.
digits_exp14:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_gpu=True --device_num=0 --conv=False --debug=False

# ----------------------------------------------------------------------------
# VAE testing
# CPU
digits_exp15:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.005 --use_gpu=False --device_num=0 --debug=True | tee $(DATA_PATH)/digits_exp15.log

# w/ glia
digits_exp16:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.01 --use_gpu=True --device_num=0 --debug=False | tee $(DATA_PATH)/digits_exp16.log

# w/ neurons
digits_exp17:
	glia_digits.py --glia=False --epochs=300 --progress=True --lr=0.005 --use_gpu=True --device_num=0 --debug=False | tee $(DATA_PATH)/digits_exp17.log

# ----------------------------------------------------------------------------
# Hyper params sweep 1 for AGN
# Setup to run on Churchlands
# exp1 ran fine, but not great results
tune_digits_exp1:
	tune_digits.py tune_1 data/exp1/ --num_samples=50 --max_iterations=20 --use_gpu=True

# tweaked config 
# up samples 
# 7dae017059c9b976da462d55cd85aa579b33bfaa
tune_digits_exp2:
	tune_digits.py tune_1 data/exp2/ --num_samples=100 --max_iterations=20 --use_gpu=True

# Froze VAE lr
# fa11ee7a53df05b52167b859370a0824b7de6ff2
tune_digits_exp3:
	tune_digits.py tune_1 data/exp3/ --num_samples=200 --max_iterations=20 --use_gpu=True

# Moved to py36 to get tensoprboard working. Can't easily vis the above.
# Had to parse logs. Booos. Rerun exp3, now w/ TB?
# fa11ee7a53df05b52167b859370a0824b7de6ff2
tune_digits_exp4:
	tune_digits.py tune_1 data/exp4/ --num_samples=200 --max_iterations=20 --use_gpu=True

# 12-10-2018
# Prev tunes were totally usefless. Somehow I forked the wrong code so none
# of the hyperparams that were suppose to be run, were. Only lr was sampled
# in any of the above. That error is now bieng fixed....
# e10df71dfdf3ae020e72384e566fbd26669c034b
#
# For laptop:
tune_digits_exp5:
	tune_digits.py tune_1 data/exp5/ --num_samples=2 --max_iterations=2 --use_gpu=False

# Test VAEGather
# e10df71dfdf3ae020e72384e566fbd26669c034b
tune_digits_exp6:
	tune_digits.py tune_1 data/exp6/ --num_samples=100 --max_iterations=20 --use_gpu=True

# Test LinearGather
# 8d54eb5c4385c0c7d5bd33db84b83c431b19c36a
# Errored: debug later.
tune_digits_exp7:
	tune_digits.py tune_1 data/exp7/ --num_samples=100 --max_iterations=20 --use_gpu=True

# Test VAESlide
# 5e3286b12b87d26615c04a966c48f5476c8531f0
# Notinhg above 45%. Best models had < 5 layers, used ELU or Tanh
tune_digits_exp8:
	tune_digits.py tune_1 data/exp8/ --num_samples=100 --max_iterations=20 --use_gpu=True

# Test VAESlide
# Repeat exp8 but sample inline w/ those results,
# and search the epsilon param in ADAM.
# num_hidden 1-5
# lr 0.005-0.1
# Fix ELU
# epsilon 1e-8 - .1 (Huge per tensorflow rec)
# dcdf500fe7609056b80efab8f16bcfec83615d37
tune_digits_exp9:
	tune_digits.py tune_1 data/exp9/ --num_samples=200 --max_iterations=20 --use_gpu=True


# VAESpread: search num_hidden and lr
# f09b166ba5640613d4b63b0a705e1b61ae98e555
# This gave a bump in final ACC to ~0.51, the largest increase for HP so far.
# Best model: lr=0.0097, num_hidden=2
tune_digits_exp10:
	tune_digits.py tune_1 data/exp10/ --num_samples=200 --max_iterations=20 --use_gpu=True


# ---------------------------------------------------------------------------
# 5-6-2019
# 7dd363c757700feb81647b4aa213b574401d9e66
digits_test:
	glia_digits.py VAE --glia=True --epochs=10 --progress=True --use_gpu=False 

# Glia comp is not analogoues to point source intiation followed by a circular
# Ca traveling wave, than eventually gets summarized/decoded to digits.

digits_exp18:
	glia_digits.py VAE --glia=False --epochs=10 --progress=True --use_gpu=True | tee $(DATA_PATH)/digits_exp18.log

# SUM: Training peaked at 87.8% - a (huge) new record. Traveling waves are the way to go.
# NEXT: Explore epochs, lr, Explore a Spread (w/ gather)?
digits_exp19:
	glia_digits.py VAE --glia=True --epochs=500 --progress=True --use_gpu=True | tee $(DATA_PATH)/digits_exp19.log

# ---------------------------------------------------------------------------
# 5-7-2019
# NOTE: EXPs 20-24 use a GPU, AND the device is set manually so each can be 
# run in parallel.

# Repeat of 19, saving model this time
digits_exp20:
	glia_digits.py VAE --glia=True --epochs=1000 --progress=True --use_gpu=True --device_num=0 --save=$(DATA_PATH)/digits_exp20 | tee $(DATA_PATH)/digits_exp20.log

# Longer version of 19
digits_exp21:
	glia_digits.py VAE --glia=True --epochs=1000 --progress=True --use_gpu=True --device_num=0 --save=$(DATA_PATH)/digits_exp21 | tee $(DATA_PATH)/digits_exp21.log

# Longer version of 19, lr=0.02
digits_exp22:
	glia_digits.py VAE --glia=True --epochs=1000 --lr=0.02 --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp22 | tee $(DATA_PATH)/digits_exp22.log

# Test exp! Try a growing the shrinking traveling wave
digits_exp23:
	glia_digits.py VAE --glia=True --epochs=10 --lr=0.01 --wave_size=24 --debug=True --device_num=1 --progress=True --use_gpu=False

# Try a growing the shrinking traveling wave
digits_exp24:
	glia_digits.py VAE --glia=True --epochs=1000 --lr=0.01 --wave_size=40 --debug=True --device_num=2 --progress=True --use_gpu=True --save=$(DATA_PATH)/digits_exp24 | tee $(DATA_PATH)/digits_exp24.log

# Try a growing the shrinking traveling wave: lr=0.02
digits_exp25:
	glia_digits.py VAE --glia=True --epochs=1000 --lr=0.02 --wave_size=40 --debug=True --device_num=2 --progress=True --use_gpu=True --save=$(DATA_PATH)/digits_exp25 | tee $(DATA_PATH)/digits_exp25.log

# ---------------------------------------------------------------------------
# 8-20-2019
#
# Fixed in-place errors 
# 302f2eb3004371752e1d411e69c834fa9bec1841

digits_rp_test:
	glia_digits.py RP --random_projection=GP --glia=True --epochs=10 --progress=True --use_gpu=False 


# Try two RP exps, one for each projection type
# I'm getting an error for numpy use w/ GPU on. 
# To get result now, I'm swtiching to CPU. Revisit later!
#

# GP
# SUM: Test accuracy was chance. ~12 %
digits_exp26:
	glia_digits.py RP --glia=True --random_projection=GP --epochs=500 --progress=True --use_gpu=False | tee $(DATA_PATH)/digits_exp126.log

# SP
# SUM: Test accuracy was change. ~52 %
digits_exp27:
	glia_digits.py RP --glia=True --random_projection=SP --epochs=500 --progress=True --use_gpu=False  | tee $(DATA_PATH)/digits_exp127.log

# VAE (Re-run 19; consistency check)
# SUM: Accuarcy was 87% (the high so far w/ glia)
digits_exp28:
	glia_digits.py VAE --glia=True --random_projection=GP --epochs=500 --progress=True --use_gpu=False | tee $(DATA_PATH)/digits_exp128.log

# ---------------------------------------------------------------------------
# 8-21-2019

# SP w/ neuronal learning
# SUM: accuracy was ~75% (Glia have ~20% to go)
digits_exp129:
	glia_digits.py RP --glia=False --random_projection=SP --epochs=500 --progress=True --use_gpu=False  | tee $(DATA_PATH)/digits_exp129.log

# GP w/ neuronal learning
# SUM: accuracy was ~75%; (Meanwhile glia were at chance)
digits_exp130:
	glia_digits.py RP --glia=False --random_projection=GP --epochs=500 --progress=True --use_gpu=False  | tee $(DATA_PATH)/digits_exp130.log

# VAE w/ neuronal learning
# SUM: accuracy was 95.47 % (glia have ~10% to go; try some metaparam opt?).
digits_exp131:
	glia_digits.py VAE --glia=False --epochs=500 --progress=True --use_gpu=False | tee $(DATA_PATH)/digits_exp131.log

# ---------------------------------------------------------------------------
# 8-22-2019
# e27907b58c2d56568d5758da1b3abdfd1a78ed20

# Tune some glia nets w/ random search 
# Note: the search has changed from the above in two key ways. Because it is
# buggy in bad way, Ray was dropped. Also, some of the API changed to work
# w/ a broader range of HP.

test_tune_digits:
	tune_digits.py random $(DATA_PATH)/test_tune_digits run_VAE \
		--num_samples=2 --seed_value=1 \
		--num_epochs=5 --glia=False --use_gpu=False --lr='(0.001, 0.1)' --lr_vae='(0.01, 0.1)'

# Tune lrs.
# ANN
tune_digits11:
	tune_digits.py random $(DATA_PATH)/tune_digits11 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=100 --glia=False --use_gpu=True --lr='(0.000001, 0.1)' --lr_vae='(0.000001, 0.1)'

# Glia - VAE
tune_digits12:
	tune_digits.py random $(DATA_PATH)/tune_digits12 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=100 --glia=True --use_gpu=True --lr='(0.000001, 0.1)' --lr_vae='(0.000001, 0.1)'

# Glia - GP
tune_digits13:
	tune_digits.py random $(DATA_PATH)/tune_digits13 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=100 --glia=True --use_gpu=True --lr='(0.000001, 0.1)' --random_projection=GP

# Glia - SP
tune_digits14:
	tune_digits.py random $(DATA_PATH)/tune_digits14 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=100 --glia=True --use_gpu=True --lr='(0.000001, 0.1)' --random_projection=SP
