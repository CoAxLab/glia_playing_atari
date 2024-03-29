SHELL=/bin/bash -O expand_aliases
# DATA_PATH=/Users/type/Code/glia_playing_atari/data/
# DATA_PATH=/Users/qualia/Code/glia_playing_atari/data
DATA_PATH=/home/stitch/Code/glia_playing_atari/data/

# ----------------------------------------------------------------------------
# Tests - should always run fine
digits_test:
	glia_digits.py VAE --glia=True --num_epochs=150 --use_gpu=False --lr=0.004 --lr_vae=0.01 --debug=True --seed_value=None --save=$(DATA_PATH)/digits_test

fashion_test:
	glia_fashion.py VAE --glia=True --num_epochs=150 --use_gpu=False --lr=0.008 --lr_vae=0.01 --debug=True --seed_value=None --save=$(DATA_PATH)/fashion_test

# ----------------------------------------------------------------------------
xor_exp1:
	glia_xor.py --glia=False --debug=True 

xor_exp2:
	glia_xor.py --glia=True --debug=True 

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

# NOTE: The command line API changed from the above. 
#       --epoch -> --num_epoch
#       --use_cuda -> --use_gpu

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


# ---------------------------------------------------------------------------
# 8-29-2019
#
# Search z and lr (fix lr_VAE)
# 898b8ba9e971cc840dd03bac6f2447d2eb4841fa

# SUM: When z=24 or 40 all simulations gave numerically identical results. 
# This was true with and between 15 and 16. Something is quite wrong.
# Try in ANN mode next. Is the problem the perceptron or elsewhere?

# Glia - VAE
tune_digits15:
	tune_digits.py random $(DATA_PATH)/tune_digits15_z12 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=12
	tune_digits.py random $(DATA_PATH)/tune_digits15_z16 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=16
	tune_digits.py random $(DATA_PATH)/tune_digits15_z20 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=20
	tune_digits.py random $(DATA_PATH)/tune_digits15_z24 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=24
	tune_digits.py random $(DATA_PATH)/tune_digits15_z40 run_VAE \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=40

# Glia - SP
tune_digits16:
	tune_digits.py random $(DATA_PATH)/tune_digits16_z12 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=12
	tune_digits.py random $(DATA_PATH)/tune_digits16_z16 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=16
	tune_digits.py random $(DATA_PATH)/tune_digits16_z20 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=20
	tune_digits.py random $(DATA_PATH)/tune_digits16_z24 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=24
	tune_digits.py random $(DATA_PATH)/tune_digits16_z40 run_RP \
		--num_samples=100 --seed_value=1 \
		--num_epochs=25 --glia=True --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=40

# ---------------------------------------------------------------------------
# 9-4-2019
# 9ba8fd48cdca494659eed215c26fa4c73f160f1a
# 
# Control exps for 15/16
# Run z=40 but using ANN mode. Are the sims still identical?
# 
# SUM: Results look fine. The bug--if there is one--is in the GliaNet()
tune_digits17:
	tune_digits.py random $(DATA_PATH)/tune_digits17_a run_VAE \
		--num_samples=16 --seed_value=1 \
		--num_epochs=25 --glia=False --use_gpu=True --lr='(0.0001, 0.1)' --lr_vae=0.01 --z_features=40 | tee $(DATA_PATH)/tune_digits17_b.log
	tune_digits.py random $(DATA_PATH)/tune_digits17_b run_RP \
		--num_samples=16 --seed_value=1 \
		--num_epochs=25 --glia=False --use_gpu=True --lr='(0.0001, 0.1)' --random_projection=SP --z_features=40 | tee $(DATA_PATH)/tune_digits17_b.log

# ---------------------------------------------------------------------------
# 9-4-2019
# 4e6f7ed4d4431bb2874e9bca3039dbeddd14d3a2
#
# Train a VAE only. Use it to train glia/neurons. This should improve final
# performance of both models?
digits_exp134:
	glia_digits.py VAE_only --num_epochs=100 --z_features=20 --progress=True --use_gpu=True --save=$(DATA_PATH)/digits_exp134_VAE_only | tee $(DATA_PATH)/digits_exp134.log

# Note: device_num set for the next three exps
# SUM: Correct: 78.67 
digits_exp135:
	glia_digits.py VAE --glia=True --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp135 | tee $(DATA_PATH)/digits_exp135.log

# SUM: Correct: 93.73
digits_exp136:
	glia_digits.py VAE --glia=False --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=2 --save=$(DATA_PATH)/digits_exp136 | tee $(DATA_PATH)/digits_exp136.log

# Glia nets have 5x the layers (for z=20). Up the training time to compensate.
# SUM: Correct: 80.96
digits_exp137:
	glia_digits.py VAE --glia=True --num_epochs=500 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=3 --save=$(DATA_PATH)/digits_exp137 | tee $(DATA_PATH)/digits_exp137.log

# -
# cc48ca7fa3ab575ada3deb6b8d9e0a24ee6548ac
# Try a faster lr? (default is 0.005)
# SUM: poor progress at the start.
# SUM: stopped after 10 iterations; no learning progress.
digits_exp138:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.01 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp138 | tee $(DATA_PATH)/digits_exp138.log

# Slower, lr=0.001 (default is 0.005)
# SUM: Correct: 75.45 
digits_exp139:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.001 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp139 | tee $(DATA_PATH)/digits_exp139.log

# ---------------------------------------------------------------------------
# 9-5-2019
# 
# Commented out the slide layer in GliaNet(); Otherwise it is a rep of 135.
# 5288667057166ec79faa73d3c29c173313937545
#
# SUM: Correct: 73.73
digits_exp140:
	glia_digits.py VAE --glia=True --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp140 | tee $(DATA_PATH)/digits_exp140.log

# -
# 15490ee1f3357d3d730947454c81a23aa69e78b9
# (Turned Slide back on)

# Increase VAE only training. N = 250 (up from 100).
digits_exp141:
	glia_digits.py VAE_only --num_epochs=250 --z_features=20 --progress=True --use_gpu=True --save=$(DATA_PATH)/digits_exp141_VAE_only | tee $(DATA_PATH)/digits_exp141.log

# Run w/ VAE from 141; otherwise this is a rep of 135.
# SUM: Correct: 77% (peak of 81%); Helps? 
#      About training epoch 75 saw a peak of 81% correct. Should I be stopping
#      sooner?
digits_exp142:
	glia_digits.py VAE --glia=True --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp141_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp142 | tee $(DATA_PATH)/digits_exp142.log

# Run w/ larger batch_size = 256 (up from 128); VAE is 134.
# SUM: Correct: 77.3; Larger batch made no difference
digits_exp143:
	glia_digits.py VAE --glia=True --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp134_VAE_only.pytorch --batch_size=256 --test_batch_size=256 --progress=True --use_gpu=True --device_num=2 --save=$(DATA_PATH)/digits_exp143 | tee $(DATA_PATH)/digits_exp143.log

# -
# Increase VAE only training. N = 500 (up from 250 and 100).
digits_exp144:
	glia_digits.py VAE_only --num_epochs=500 --z_features=20 --progress=True --use_gpu=True --save=$(DATA_PATH)/digits_exp144_VAE_only | tee $(DATA_PATH)/digits_exp144.log

# Run w/ VAE from 141; otherwise this is a rep of 135.
# SUM: Correct: 79.76; Small increase. Tweak lr using ths VAE?
digits_exp145:
	glia_digits.py VAE --glia=True --num_epochs=100 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp145 | tee $(DATA_PATH)/digits_exp145.log

# -
# 9-5-2019
# Small lr tweaks
# Baseline: lr=0.005
# 790554c028ab153587a510d0bb3b9b41b571e080

# SUM 146-150: lr=0.001-0.005 looks like the best range (by a small margin).

# lr=0.0025
# SUM: Correct: 78.41
digits_exp146:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.0025 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=0 --save=$(DATA_PATH)/digits_exp146 | tee $(DATA_PATH)/digits_exp146.log

# lr=0.0075
# SUM: Correct: ~10%
digits_exp147:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.0075 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=1 --save=$(DATA_PATH)/digits_exp147 | tee $(DATA_PATH)/digits_exp147.log

# lr=0.0015
# SUM: Correct: 79.64
digits_exp148:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.0015 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=2 --save=$(DATA_PATH)/digits_exp148 | tee $(DATA_PATH)/digits_exp148.log

# lr=0.0005
# SUM: Correct: 75.46
digits_exp149:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.0005 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=3 --save=$(DATA_PATH)/digits_exp149 | tee $(DATA_PATH)/digits_exp149.log

# lr=0.001
# SUM: Correct: 76.56
digits_exp150:
	glia_digits.py VAE --glia=True --num_epochs=100 --lr=0.001 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --progress=True --use_gpu=True --device_num=0 --save=$(DATA_PATH)/digits_exp150 | tee $(DATA_PATH)/digits_exp150.log


# ---------------------------------------------------------------------------
# 9-7-2019
# CCN poster runs.  N=20 runs. No fixed seed.
#
# VAE - train both
# Glia
digits_exp151:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/digits_exp151_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
digits_exp152:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=False --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/digits_exp152_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	
# VAE - train ANN/AAN. Use pretrained VAE
digits_exp153:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp153_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
digits_exp154:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=False --num_epochs=150 --use_gpu=True --lr=0.004 --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp154_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# -
# Random projection
# Glia
digits_exp155:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py RP --glia=True --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/digits_exp155_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
digits_exp156:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py RP --glia=False --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/digits_exp156_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 9-9-2019
# 85d2f53efc7fedb62dc83abc4aaabfdc69746f51
# CCN XOR runs
xor_exp3:
	parallel -j 20 -v \
		--nice 19 --delay 2 --colsep ',' \
		'glia_xor.py --glia=False --seed_value=None --save=$(DATA_PATH)/xor_exp3_{1}' ::: {1..20}

xor_exp4:
	parallel -j 20 -v \
		--nice 19 --delay 2 --colsep ',' \
		'glia_xor.py --glia=False --seed_value=None --save=$(DATA_PATH)/xor_exp4_{1}' ::: {1..20}


# ---------------------------------------------------------------------------
# 10-8-2019
# d5bbf16f235d261adee7326dcb2176685784007d
# Leak and digits
#
# SUM: there to-chance drop in between 0.2 and 0.5. Need more resolution. 

digits_exp157:
	# sigma: 0.1
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.1 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp157_s01_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# sigma: 0.2
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.2 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp157_s02_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.5 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp157_s05_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.6 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp157_s06_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 10-10-2019
# d5bbf16f235d261adee7326dcb2176685784007d
# Expanding run from 157
#
digits_exp158:
	# sigma: 0.3
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.3 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp158_s03_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# sigma: 0.35
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.35 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp158_s035_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# sigma: 0.4
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --leak=True --sigma=0.4 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp158_s04_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 10-20-2019
# Added a new Fashion-able experiment.
#
# As a first test try both neurons and glia (w/ VAE) using params from MINST
# digits.
#
# SUM: Performance good on both. Approaching digits

# Glia
fashion_exp1:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py VAE --glia=True --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/fashion_exp1{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
fashion_exp2:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py VAE --glia=False --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/fashion_exp2_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 10/24/2019
# Random projection
#
# SUM: Performance good on both. Approaching digits

# Glia
fashion_exp3:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py RP --glia=True --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/fashion_exp3_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
fashion_exp4:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py RP --glia=False --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/fashion_exp4_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 10/28/2019
# Try some noise connections (on digit learning)
# e09448ddcc631f87f290c9858fb6c7bf50f330ff
#
# SUM: Noise at 0.1 and 0.2 started to decrease avg accuracy, but only slightly.
#      Do another run with more noise.
digits_exp159:
	# sigma: 0.1
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.01 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp159_s01_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# sigma: 0.2
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.05 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp159_s05_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.1 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp159_s1_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.2 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp159_s2_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Try some dropped connections (on digit learning)
# e09448ddcc631f87f290c9858fb6c7bf50f330ff
# SUM:
digits_exp160:
	# p: 0.1
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --drop=True --p=0.01 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp160_p01_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# p: 0.2
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --drop=True --p=0.05 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp160_p05_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --drop=True --p=0.1 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp160_p1_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --drop=True --p=0.2 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp160_p2_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# ---------------------------------------------------------------------------
# 10/30/2019
# fe8d4c6618ab7e68e3edb597b69d114fa97b9dfb
# expansion of exp159 -- more noise!
# 
# SUM:
digits_exp161:
	# sigma: 0.1
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.3 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_s3_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	# sigma: 0.2
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.4 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_4_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.5 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_s5_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.6 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_s6_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.7 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_s7_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --noise=True --sigma=0.8 --num_epochs=150 --use_gpu=True --lr=0.004  --vae_path=$(DATA_PATH)/digits_exp144_VAE_only.pytorch --seed_value=None --save=$(DATA_PATH)/digits_exp161_s8_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5


# ---------------------------------------------------------------------------
# 5/18/21
# f5a1221
#
# In a recent commit (f5a1221) I added loging of loss and acc, by epoch. Rerrun
# the main results to get these curves (for neuralIPS)

# ---
# Org exp codes:
# digits: 151-152 VAE
# digits: 155-156 RP

# Glia
digits_exp162:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=True --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/digits_exp162_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
digits_exp163:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py VAE --glia=False --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/digits_exp163_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Random projection
# Glia
digits_exp164:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py RP --glia=True --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/digits_exp164_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
digits_exp165:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_digits.py RP --glia=False --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/digits_exp165_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5


# ---
# Org exp codes:
# fashion: 1-2 VAE
# fashion: 3-4 RP
fashion_exp5:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py VAE --glia=True --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/fashion_exp5{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
fashion_exp6:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py VAE --glia=False --num_epochs=150 --use_gpu=True --lr=0.004 --lr_vae=0.01 --seed_value=None --save=$(DATA_PATH)/fashion_exp6_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Glia
fashion_exp7:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py RP --glia=True --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/fashion_exp7_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5

# Neurons
fashion_exp8:
	parallel -j 16 -v \
		--nice 19 --delay 2 --colsep ',' \
	    'glia_fashion.py RP --glia=False --num_epochs=150 --random_projection=SP --use_gpu=True --lr=0.004 --seed_value=None --save=$(DATA_PATH)/fashion_exp8_{1}{2} --device_num={1}' ::: 0 1 2 3 ::: 1 2 3 4 5
