from torch.optim import Adam, AdamW


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=1e-4,
    wd=1e-5,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=True,
    group_wd_params=True,
):
    has_wd = wd > 0

    # if filter_by_requires_grad:
    params = list(filter(lambda t: t.requires_grad, params))

    if group_wd_params and has_wd:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0},
        ]
    if not has_wd:
        return Adam(params, lr=lr, betas=betas, eps=eps)

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
