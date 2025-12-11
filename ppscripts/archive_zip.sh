#!/bin/bash
mkdir -p zip
zip zip/$1.zip -0 -r $1
