# Geometric Brownian Motion

This will show an application of Ito's Lemma that is crucial for understanding the Black Scholes equation. <br>

## Background
We know that GBM is a crucial aspect towards this project because we know that stocks behavior the way that the SDE of GBM behaves.

## Solution

Solution to the SDE <br>
$dS = \mu S_{t}dt + \sigma S_{t} dW_{t}$

The trick that we will use is to find the SDE that is satisfied by: <br>
$X_{t} = log(S_{t})$
- This also is an important concept because it shows why we we are taking the log returns of a stock within all other applicable simulations

**Using Ito's Lemma and SDE**

$dX_{t} = dlog(S)$ <br>
$dX_{t} = \partial _{s}u(S_{t}) + \frac{1}{2} \partial_{s}^{2} u(S_t)V(S_t)dt$ <br>
$dX_t = \frac{1}{S_{t}}(\mu S_tdt + \sigma S_tdW_t) - \frac{1}{2} \frac{1}{S_t^2}\sigma^2S_t^2dt$ <br>
Let $w_0 = 0 \to$ brownian motion starts at 0
Then we integrate both sides of the equation using ito calculus rules 

We get the following
$X_T = X_0 + (\mu - \frac{1}{2} \sigma^2)T + \sigma W_T$ <br>

$e^{X_0} = S_0$ <br>
$S_T = e^{X_T}$ <br>
$S_T = e^{X_0}e^{(\mu - \frac{1}{2} \sigma^2)T + \sigma W_T}$ <br>
$S_T = S_0e^{(\mu - \frac{1}{2} \sigma^2)T + \sigma W_T}$

## A couple things to note & Special Cases
One very interesting thing that we can note is that $S_T$ is only a function of $W_T$ meaning that in general the solution to the SDE only depends on the whole path $W_{[0,T]}$

We also will look at this special case where $\mu = 0$ That makes the $S_t$ a martingale which implies <br>

$E[dS_t | F_t] = 0 \to E[S_t]$ 

Solution $S_t \to 0$ as $t \to \infty$ <br>
And this is dominated by the deterministic part $\frac{-1}{2} \sigma^2 t$