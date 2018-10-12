SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/glia_playing_atari/data/


# ----------------------------------------------------------------------------
xor_exp1:
	glia_xor.py --glia=False

xor_exp2:
	glia_xor.py --glia=True

# ----------------------------------------------------------------------------
digits_exp1:
	glia_digits.py --glia=False

digits_exp2:
	glia_digits.py --glia=True
