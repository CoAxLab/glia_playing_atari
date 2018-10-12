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
	glia_digits.py --glia=False --epochs=3 

digits_exp2:
	glia_digits.py --glia=True --epochs=3
