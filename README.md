
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

Everything is in the usual requirements.txt file.

There may be a complication with [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) if portaudio is not installed.

Note that this module also depends on [this](https://github.com/MCMcCallum/data_access) url library, not available from regular PyPi servers.

You'll need [ffmpeg](https://www.ffmpeg.org/) if you want to read mp3 files.


Install
===

This is currently intended to be used as a git submodule:

`git submodule add https://github.com/MCMcCallum/sigtools`

Alternatively, once I need the convenience I'll probably write a distutils `setup.py` for this module.