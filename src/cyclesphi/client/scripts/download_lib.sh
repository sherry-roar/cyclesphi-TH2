#!/bin/bash

ROOT=$PWD

mkdir -p $ROOT/src
mkdir -p $ROOT/src/lib

cd $ROOT/src/lib

svn co https://svn.blender.org/svnroot/bf-blender/tags/blender-3.3-release/lib/linux_centos7_x86_64