# Meeting 2 guide

## Topics
- SDE book Ito Integrals (Chapter 3)
- Simulating Brownian Motion & Multidimensional Brownian Motion
- Asset Allocation w/ Market Regime Switching Paper

## Ito Integrals

### What functions can be used for ito integrals

$\nu = \nu(S,T)$ is the class of functions <br>
Needs to have these properties:
- $(t,w) \to f(t,w)$ is $B x F$-measurable
- $f(t,w)$ is $f_t$-apdapted
- $E[\int_{S}^{T}f(t,w)^{2} \,dt] < \infin$


### Connection to project:
- Filtration
- Isometry $\to$ variance calc
- Martingale $\to$ ito integral is a martingale
- Extension of ito integral (multidimensional ito integral)


### Filtration
Main Idea: $F_t$ everything you know about $B_{t}(w)$ up until $t$ <br>
$F_t$ is the object that encodes information "past"

At time $t$ you know where the price has been from $[0,t]$, but you don't know about after $t$

### Martingale
Main Idea: The ito integral is a martingale, which tells us that an Ito integral is directly related to brownian motion

Since $B_t(w)$ is a martingale we know that stock options are a martingale because stocks can go only 2 ways up or down

### Multidimensional Ito Integral








