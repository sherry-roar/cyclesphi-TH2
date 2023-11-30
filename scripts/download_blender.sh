#!/bin/bash

ROOT=$PWD

mkdir -p $ROOT/src
mkdir -p $ROOT/build
mkdir -p $ROOT/install
mkdir -p $ROOT/scripts
mkdir -p $ROOT/lib

cd $ROOT/src

#ml git
git clone git@code.it4i.cz:raas/cyclesphi.git

cd $ROOT

cp -r $ROOT/src/cyclesphi/client/scripts/* scripts/.

