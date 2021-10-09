from gmtb.gmpd import set_median
from gmtb.gmpd.convert import convert

def init_alpha( set_p, sod_func ,weights):
#GMPD_INIT_ALPHA finds an initialization for alpha (at the moment set median)
    
    set_m, _ = set_median(set_p, sod_func, weights)
   
    alpha = convert(set_m)

    return alpha

