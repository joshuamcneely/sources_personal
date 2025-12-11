#!/bin/bash
mkdir build
sphinx-apidoc -f -o source/ ../
sphinx-build -b html source/ build/
