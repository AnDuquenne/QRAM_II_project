def compute_return_365(r, days):
    R_365 = (1 + r) ** (365 / days) - 1
    return R_365