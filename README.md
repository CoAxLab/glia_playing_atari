# Glia functions!

Here we define and study Artificial Glia Networks. 

A proof by [Barron](http://www.stat.yale.edu/~arb4/publications_files/UniversalApproximationBoundsForSuperpositionsOfASigmoidalFunction.pdf) established that artificial neural networks are universal function approximators. This has, in part anyway, motivated much study of ANNs. Turns out though that Barron's result is easy to extend to glia networks, if that is we consider Calcium (Ca) concentrations in place of firing rates. This is because:

1. Glia release neurotransmitters.
2. The Ca++ response to neurotransmitters in glia is sigmoidal


But!

3. Unlike neurons, Glia don't form specific synapses. This means their communication is (probably) limited to local but diffuse connections.

While we'll show the proof extension in a companion paper (it rests on linear superpostions of summations), here we explore the computational properties of AGNs in practice. 

As a first pass in building AGNs we interpret 'local and diffuse' to mean that only gila who are nearest neighbors can communicate with fidelity. For example:

TODO diagram: comparing ANNs and AGNs

TODO diagram: AGN Grow and Shrink layers

# Install

`pip install git+https://github.com/CoAxLab/glia_playing_atari.git`

# Dependencies

- Standard anaconda
- PyTorch (4.1)