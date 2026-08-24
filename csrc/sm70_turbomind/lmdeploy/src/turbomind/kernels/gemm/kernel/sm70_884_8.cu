// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {

template<class Gemm>
class Qwen38PrescaledFullTileKernelImpl final: public KernelImpl<Gemm> {
public:
    Qwen38PrescaledFullTileKernelImpl()
    {
        this->info_.name += "_q38_pscale_full";
    }

    bool is_feasible(const GemmDesc& desc) const noexcept override
    {
        return desc.m == 8000 && (desc.n == 4096 || desc.n == 3584) && desc.k == 5120 && desc.num == 1
               && KernelImpl<Gemm>::is_feasible(desc);
    }
};

}  // namespace

void Registry::sm70_884_8()
{
    if constexpr (1) {
        // clang-format off
        using C = Config_E4M3<kColMajor, 0>;
        Add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 96, 128,  32, 2, 2, 1, D, S, 2, true, 1, 128,  48, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 64, 256,  16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();

        using CP = Config_E4M3_Prescaled<kColMajor, 0>;
        using Q38PrescaledFull = CP::Type<64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128, 64, 128, 1, true>;
        Add(std::make_unique<Qwen38PrescaledFullTileKernelImpl<typename Q38PrescaledFull::Kernel>>());
        // clang-format on
    }
}

}  // namespace turbomind::gemm
