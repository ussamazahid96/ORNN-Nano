#pragma once

#ifndef QORNN_H
#define QORNN_H 

#include <iostream>

#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
// #include "HLS/math.h"

#define ASSERT(x, y) {if (!(x)) {std::cout<< "assertion failed: " << y  << std::endl; exit(-1); }}

#include "router.h"
#include "stream_utils.h"
#include "linear.h"
#include "tanh_lookup.h"
#include "activation.h"
#include "rnn.h"

#endif