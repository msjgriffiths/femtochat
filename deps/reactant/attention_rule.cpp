// Enzyme-MLIR reverse interface for registered CUDA calls.
// This is compiler glue: no numerical kernels and no GPU runtime work.
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "MLIR/Interfaces/AutoDiffOpInterface.h"
#include "MLIR/Interfaces/GradientUtilsReverse.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct CUDAReverse : ReverseAutoDiffOpInterface::FallbackModel<CUDAReverse> {
    LogicalResult createShadowValues(Operation*, OpBuilder&, MGradientUtilsReverse*) const {
        return success(); // Ranked tensors are immutable SSA values.
    }

    SmallVector<Value> cacheValues(Operation* op, MGradientUtilsReverse* g) const {
        SmallVector<Value> caches;
        if (!op->hasAttr("femtochat.backward_target")) return caches;
        OpBuilder builder(g->getNewFromOriginal(op));
        builder.setInsertionPointAfter(g->getNewFromOriginal(op));
        // Save inputs and forward results for the registered reverse callback.
        for (Value v : op->getOperands())
            caches.push_back(g->initAndPushCache(g->getNewFromOriginal(v), builder));
        for (Value v : op->getResults())
            caches.push_back(g->initAndPushCache(g->getNewFromOriginal(v), builder));
        return caches;
    }

    LogicalResult createReverseModeAdjoint(Operation* op, OpBuilder& builder,
                                           MGradientUtilsReverse* g,
                                           SmallVector<Value> caches) const {
        auto target = op->getAttrOfType<StringAttr>("femtochat.backward_target");
        if (!target) return op->emitError("No registered FemtoChat reverse rule for custom call");
        if (g->isConstantValue(op->getResult(0))) return success();
        Value dO = g->diffe(op->getResult(0), builder);
        g->zeroDiffe(op->getResult(0), builder);
        SmallVector<Value> args{dO};
        for (Value cache : caches) args.push_back(g->popCache(cache, builder));

        SmallVector<Type> types;
        for (Value input : op->getOperands()) {
            auto type = cast<RankedTensorType>(input.getType());
            if (isa<FloatType>(type.getElementType())) types.push_back(type);
        }
        OperationState state(op->getLoc(), "stablehlo.custom_call");
        state.addOperands(args);
        state.addTypes(types);
        state.addAttribute("call_target_name", target);
        state.addAttribute("api_version", builder.getI32IntegerAttr(4));
        state.addAttribute("has_side_effect", builder.getBoolAttr(false));
        state.addAttribute("backend_config", op->getAttr("femtochat.backward_config"));
        state.addAttribute("operand_layouts", op->getAttr("femtochat.backward_operand_layouts"));
        state.addAttribute("result_layouts", op->getAttr("femtochat.backward_result_layouts"));
        Operation* reverse = builder.create(state);
        unsigned index = 0;
        for (Value input : op->getOperands()) {
            auto type = cast<RankedTensorType>(input.getType());
            if (!isa<FloatType>(type.getElementType())) continue;
            Value derivative = reverse->getResult(index++);
            if (g->isConstantValue(input)) continue;
            g->addToDiffe(input, derivative, builder);
        }
        return success();
    }
};
}

extern "C" bool register_femtochat_typed_reverse_rule(MlirContext context) {
    auto op = RegisteredOperationName::lookup("stablehlo.custom_call", unwrap(context));
    if (!op) return false;
    if (!op->hasInterface<ReverseAutoDiffOpInterface>())
        op->attachInterface<CUDAReverse>();
    return true;
}
