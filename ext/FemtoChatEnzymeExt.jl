module FemtoChatEnzymeExt

using Enzyme
using FemtoChat

import FemtoChat: loss_and_gradient!
using FemtoChat.Parameters: Params, GPTConfig, 🤖

Enzyme.Duplicated(params::Params) =
    Enzyme.Duplicated(params.Θ, params.δ)

loss(Θ, config, layout, tokens, targets) =
    🤖(Θ, config, layout)(tokens, targets)

"""
Compute the loss and accumulate its gradient directly into `params.δ`.

"""
function loss_and_gradient!(
    params::Params{T,P},
    config::GPTConfig,
    layout,
    tokens,
    targets,
) where {T,P<:Vector}
    fill!(params.δ, 0f0)

    Enzyme.API.strictAliasing!(false)
    Enzyme.API.looseTypeAnalysis!(true)
    mode = Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal)

    _, primal = Enzyme.autodiff(
        mode,
        loss,
        Enzyme.Active,
        Enzyme.Duplicated(params),
        Enzyme.Const(config),
        Enzyme.Const(layout),
        Enzyme.Const(tokens),
        Enzyme.Const(targets),
    )

    return primal
end

end
