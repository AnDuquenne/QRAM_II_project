def compute_return_365(r, days):
    R_365 = (1 + r) ** (365 / days) - 1
    return R_365

def compute_return_252(r, days):
    R_252 = (1 + r) ** (252 / days) - 1
    return R_252

def compute_return_daily(r, days):
    R_daily = (1 + r) ** (1 / days) - 1
    return R_daily