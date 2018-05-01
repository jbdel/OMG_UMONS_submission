#!/usr/bin/env bash
echo Building Native ops...
mkdir -p build
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


# TODO: GPU support
#nvcc -std=c++11 -c -o count_sketch.cu.o count_sketch.cu.cc -I -I$TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED -g
#g++ -std=c++11 -shared -o count_sketch.so count_sketch.cc count_sketch.cu.o -fPIC -lcudart -I $TF_INC -D_GLIBCXX_USE_CXX11_ABI=0 -g


g++ -std=c++11 -shared -o build/count_sketch.so ops/count_sketch.cc -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework


