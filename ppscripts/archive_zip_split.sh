#!/bin/bash
mkdir -p zip
zip -s 10G zip/$1.zip -0 -r $1
