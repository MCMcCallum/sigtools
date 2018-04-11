
sigtools
===

Written by Matt C. McCallum.

A number of reusable Python audio signal processing functions I use in my code.

The current responsibilities of this module include:

    - Accessors for media files
    - Media file conversion functions
    - Signal processing analysis algorithms, math, filters

Dependencies
===

Everything is in the usual requirements.txt file. The only complication might be with [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/).

Install
===

This is currently intended to be used as a git submodule:

`git submodule add https://github.com/MCMcCallum/sigtools`

Alternatively, once I need the convenience I'll probably write a distutils `setup.py` for this module.