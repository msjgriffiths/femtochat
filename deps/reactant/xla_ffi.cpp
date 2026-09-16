// ABI adapter only: all numerical kernels remain Julia CUDA code.
// Headers must match the XLA revision used by libReactantExtra (see build script).
#include "xla/ffi/api/ffi.h"
#include <vector>
namespace ffi = xla::ffi;

static ffi::Error launch(void* stream, ffi::RemainingArgs args,
                         ffi::RemainingRets rets, int64_t callback_ptr) {
    std::vector<void*> buffers;
    buffers.reserve(args.size() + rets.size());
    for (size_t i = 0; i < args.size(); ++i) {
        auto buffer = args.get<ffi::AnyBuffer>(i);
        if (!buffer.has_value()) return ffi::Error::Internal("Invalid input buffer");
        buffers.push_back(buffer->untyped_data());
    }
    for (size_t i = 0; i < rets.size(); ++i) {
        auto buffer = rets.get<ffi::AnyBuffer>(i);
        if (!buffer.has_value()) return ffi::Error::Internal("Invalid output buffer");
        buffers.push_back((*buffer)->untyped_data());
    }
    auto callback = reinterpret_cast<bool (*)(void*, void**)>(callback_ptr);
    return callback(stream, buffers.data()) ? ffi::Error::Success() :
        ffi::Error::Internal("Julia CUDA callback failed; inspect ReactantCUDACall.errors");
}

XLA_FFI_DEFINE_HANDLER(handler, launch,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<void*>>()
        .RemainingArgs().RemainingRets().Attr<int64_t>("callback_ptr"));

extern "C" XLA_FFI_Error* register_femtochat_ffi(const char* name) {
    return ffi::Ffi::RegisterStaticHandler(XLA_FFI_GetApi(), name, "CUDA", handler);
}
