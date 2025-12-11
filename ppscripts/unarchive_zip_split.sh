#!/bin/bash
zip -s 0 $1 --out $1.joined.zip
unzip $1.joined.zip
