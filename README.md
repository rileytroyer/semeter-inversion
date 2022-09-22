# semeter-inversion

This project takes an electron density profile as measured by the Poker Flat Incoherent Scatter Radar (PFISR) and estimates an incoming differential energy flux.

Be aware, this is not intended to be production quality code.

Python packages iri2016 and msise00 are required. You can see how to install them below

msise00: https://github.com/space-physics/msise00
iri2016: https://github.com/space-physics/iri2016

Note: for iri2016 you should check that the indices include the dates that you are using.

Updated indices are here: http://irimodel.org/indices/

On my anaconda installation indices are stored for the code at:
envs/semeter-inversion/lib/python3.9/site-packages/iri2016/iri2016/data/index/{apf107,ig_rz}.dat

