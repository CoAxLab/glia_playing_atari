SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/glia_playing_atari/data/


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
# How long should I let this go before calling it. Acc dropped below 50. Stopped early.
digits_exp8:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.015

# Bump lr up a little (in between 7 and 8)
# Learning stalled at 0.49. I don't get this at all!
digits_exp9:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.0125

# Re-run 7, using ADAM instead of SGD.
digits_exp10:
	glia_digits.py --glia=True --epochs=300 --progress=True --lr=0.01
