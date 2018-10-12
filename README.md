# Glia functions!

Here we define and study Artifical Glia Networks. Glia aren't typicaly thought of as doing function approximiation. However the orginal proof by [Barron](http://www.stat.yale.edu/~arb4/publications_files/UniversalApproximationBoundsForSuperpositionsOfASigmoidalFunction.pdf) which showed how artificial _neural_ networks are univeral function approximators, is easy actually to extend to glia networks, as:

1. Glia release neurotransmitters
2. The Ca++ response to neurotransmitters in glia is sigmoidal
3. But, unlike neurons, Glia don't generate extended processes, so we assume thay can only form local diffuse connections.

While we'll show this proof extension in a companion paper, here we explore the computaional properties of AGNs in practice.

As a first pass in building AGNs we interprest 'local and diffuse' to mean that only gial who are nearest neighbors can communicate. For example:

TODO diagram: comparing ANNs and AGNs

TODO diagram: AGN Grow and Shrink layers

# Install

`pip install git+https://github.com/CoAxLab/glia_playing_atari.git`

# Dependencies

- Standard anaconda
- PyTorch (4.1)