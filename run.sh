#!/bin/sh

dir=COVID-19
repo=https://github.com/CSSEGISandData/COVID-19.git

[ -d "$dir" ] && cd COVID-19 && git pull && cd .. || git clone $repo

python analysis.py
