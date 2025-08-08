// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTION1D_ARM_H
#define LAYER_DECONVOLUTION1D_ARM_H

#include "deconvolution1d.h"

namespace ncnn {

class Deconvolution1D_arm : public Deconvolution1D
{
public:
    Deconvolution1D_arm();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION1D_ARM_H
