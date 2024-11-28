import jax
import jax.numpy as jnp

def thickness_from_x(x, B, T, P):
    thickness = 1 - (jnp.abs(x-0.5)/0.5) ** (2/(B-1))
    thickness = thickness ** ((B-1)/2)
    thickness *= (1-x**P)
    thickness *= T

    return thickness

#Approx X position of max camber (Assuming R=0)
def max_camber_x(E):
    return 0.5 ** (1/E)

#All measurements are in c
#web_w is web width
#flange_w is flange width
#flange_h is flange height
#x is position of spar in c
#alpha is angle of attack in degrees
@jax.jit
def spar_moments(B, T, P, C, E, R, c, web_w, flange_w, flange_h, x, alpha)
    alpha = jnp.deg2rad(alpha)
    
    #Height of the web
    web_h = thickness_from_x(x, B, T, P) - 2 * flange_h
    
    #X position relative to centroid
    relative_x = x - max_camber_x(E)
    
    I = flange_w * flange_h ** 3 / 12 #Inertia of the flanges
    I += flange_w * flange_h * (web_h/2 + flange_h/2) ** 2 #Parallel axis theorem
    I *= 2 #Two flanges
    I += web_w * web_h ** 3 / 12 #Web
    
    #Correction factor for rotated airfoil
    I += (relative_x ** 2) * (alpha ** 2) * (web_w * web_h + 2 * flange_w * flange_h)
    
    Q0 = (web_w/8) * (2 * relative_x * alpha + web_h) ** 2
    Q0 += flange_w * flange_h * (relative_x * alpha + flange_h/2 + web_h/2)
    
    A = 2 * flange_w * flange_h + web_w * web_h #Cross section area
    
    #Unit conversion
    I *= c ** 4
    Q0 *= c ** 3
    A *= c ** 2
    
    return (I, Q0, A)