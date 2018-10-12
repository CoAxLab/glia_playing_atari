SHELL=/bin/bash -O expand_aliases
DATA_PATH=/Users/type/Code/actionflow/data/
# DATA_PATH=/home/ejp/src/azad/data/


# ----------------------------------------------------------------------------
xor_exp1:
	glia_xor.py --glia=False

xor_exp2:
	glia_xor.py --glia=True