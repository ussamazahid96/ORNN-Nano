#!/bin/bash

cd HLS
make
cd ..
export QUARTUS_ROOTDIR=$(which quartus)/../../
make


