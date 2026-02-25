import torch as th


def task_2nd_argmax(x):
    return x.argsort(-1)[..., -2]


def task_argmax(x):
    return x.argsort(-1)[..., -1]


def task_argmedian(x):
    return x.argsort(-1)[..., x.shape[-1] // 2]


def task_median(x):
    middle = x.sort(-1).values[..., (x.shape[-1] - 1) // 2 : (x.shape[-1] + 2) // 2]
    if x.shape[-1] % 2 == 0:
        if not middle.is_floating_point():
            middle = middle.float()
        return middle.mean(-1)
    else:
        return middle.squeeze(-1)


def task_argoutlier(x):
    return (x - x.mean(-1)[..., None]).abs().argsort(-1)[..., -1]


def task_longest_cycle(x):
    # The longest cycle is guaranteed to have length at least 1,
    # so we return the length minus 1 so that it lies in range(n)
    n = x.shape[1]
    start_pos = th.arange(n, device=x.device)[None].expand(*x.shape)
    pos = start_pos
    paths = []
    for _ in range(n):
        pos = x.gather(1, pos)
        paths.append(pos)
    paths = th.stack(paths, dim=-1)
    return (paths == start_pos[..., None]).to(int).argmax(-1).amax(-1)


task_registry = {
    "2nd_argmax": (task_2nd_argmax, "continuous", "dist"),
    "argmax": (task_argmax, "continuous", "dist"),
    "argmedian": (task_argmedian, "continuous", "dist"),
    "median": (task_median, "continuous", "scalar"),
    "argoutlier": (task_argoutlier, "continuous", "dist"),
    "longest_cycle": (task_longest_cycle, "discrete", "dist"),
}
