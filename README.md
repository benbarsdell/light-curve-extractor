Light Curve Extractor
=====================

A C library for extracting light curves from (pre-identified) sparse sources in image sequences.

About
-----
Light Curve Extractor (liblce) is a C library for extracting light curves from
(pre-identified) sparse sources in image sequences. Computations are accelerated
using NVIDIA GPUs via CUDA and optimised for real-time operation.

Light curve extraction has applications in neural imaging, transient
astronomy and hopefully other fields too.

If you find the library useful, or have any questions, suggestions or bug
reports, please get in touch with me!

Building
--------
To build the library:

Edit Makefile.inc to match your system configuration

$ make

To install the library system-wide:

$ make install

To build the API documentation:

$ make doc

Usage
-----
C and C++ applications can use the C API directly.

#include <light_curve_extractor.h>

cc ... -llce ...

Python applications can use the provided Python wrapper.

from light_curve_extractor import LightCurveExtractor

Contact
-------
Ben Barsdell

benbarsdell at gmail
