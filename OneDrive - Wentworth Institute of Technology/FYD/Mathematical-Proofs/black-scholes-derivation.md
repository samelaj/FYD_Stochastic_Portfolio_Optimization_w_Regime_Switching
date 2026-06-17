# Black Scholes PDE Derivation

We will show the derivation of the BS PDE within this file

## Assumptions
- Geometric Brownian Motion is how stocks move
- A risk free rate is possible

## Notation
$S_t$ = Stock Price at time t <br>
$B_t$ = GBM <br>
$\mu$ = Expected rate of return <br>
$\sigma$ = Volitility (std of returns per unit time)

## Geometric Brownian Motion
$B_t$ is a martingale which implies the following

$B_0 = 0 \text{ and } B_t$ is normally distributed

## SDE of Stock price

$\frac{dS_t}{S_t} = \mu dt + \sigma dB_t$ <br>
$dS_t = \mu S_t dt + \sigma S_t dB_t$

$E[dB_t] = 0, dB_t \approx N(0,dt)$

For $E[(dB_t)^2]$ we need to use this formula for variance of a random variable

$Var(X) = E[X^2] - (E[X])^2$ <br>
$Var(dB_t) = dt \wedge E[dB_t] = 0$ <br>
So, if $ X = dB_t:$<br>$dt = E[(dB_t)^2]$

We know that $(dB_t)^2 = 0 \wedge (dt)^2 = 0 \wedge dB_t - dt =0$

## Ito's Lemma

First we will start off with ordinary multivariante calculus using a taylor series then make a cruical subsitution

Let $f(x,t)$ be a function and then we will taylor expand out to the second order

$df = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial t}dt   + \frac{\partial^2 f}{\partial x \partial t}dxdt + \frac{1}{2} \frac{\partial^2 f}{\partial x^2}(dx)^2 + \frac{1}{2}\frac{\partial^2 f}{\partial t^2}(dt)^2$

Now we will let $f=V \wedge x = S_t$ <br>
Where: $V$ is the option price of the stock

$dV = \frac{\partial V}{\partial S_t}dS_t + \frac{\partial V}{\partial t}dt   + \frac{\partial^2 V}{\partial S_t \partial t}dS_tdt + \frac{1}{2} \frac{\partial^2 V}{\partial S_t^2}(dS_t)^2 + \frac{1}{2}\frac{\partial^2 V}{\partial t^2}(dt)^2$

Now we know that $(dt)^2 = 0 \wedge dS_t \cdot dt = 0$, so we are now left with: <br>

$dV = \frac{\partial V}{\partial S_t}dS_t + \frac{\partial V}{\partial t}dt    + \frac{1}{2} \frac{\partial^2 V}{\partial S_t^2}(dS_t)^2$

Now we will solve for $(dS_t)^2$ using the original $dS_t$ equation by squaring both sides

$(dS_t)^2 = (\mu S_t dt)^2 + (\sigma S_t dB_t)^2$

$(dS_t)^2 = \mu^2 S_t^2 (dt)^2 + \sigma^2 S_t^2 (dB_t)^2$

From before $(dt)^2 = 0 \wedge (dB_t)^2 = dt$  so we are left with <br>
$(dS_t)^2 = \sigma^2 S_t^2 (dB_t)^2$ 

Now we see that: <br>
$dV = \frac{\partial V}{\partial S_t}dS_t + \frac{\partial V}{\partial t}dt    + \frac{1}{2}\sigma^2 S_t^2  \frac{\partial^2 V}{\partial S_t^2}dt$

Now we will subsitute in the intital $dS_t$ equation in order to expand the equation <br>

$dV = \frac{\partial V}{\partial S_t}\mu S dt + \frac{\partial V}{\partial S} \sigma S dB_t + \frac{\partial V}{\partial t}dt    + \frac{1}{2}\sigma^2 S_t^2  \frac{\partial^2 V}{\partial S_t^2}dt$

Now splitting the $dV$ equation into a deterministic and random pieces

$dV = (\frac{\partial V}{\partial S_t}\mu S +  \frac{\partial V}{\partial t} +     \frac{1}{2}\sigma^2 S_t^2  \frac{\partial^2 V}{\partial S_t^2})dt + (\frac{\partial V}{\partial S} \sigma S) dB_t$

## Riskless Hedging Portfolio
We will also now say that $S_t = S$ for simplictiy in notation <br>
Now both $dS \wedge dV$ are driven by the same $dB_t$

Goal: Cancel out the randomness term by holding a portfolio of the option and some stock

Portfolio: $\Pi = V - \Delta S$ <br>
Where: $\Delta = $ the number of shares

$d\Pi = dV - \Delta dS$

$d\Pi = (\frac{\partial V}{\partial S_t}\mu S +  \frac{\partial V}{\partial t} +     \frac{1}{2}\sigma^2 S_t^2  \frac{\partial^2 V}{\partial S_t^2})dt + (\frac{\partial V}{\partial S} \sigma S) dB_t - \Delta[(\mu S) dt + (\sigma S)dB_t]$

Solve for $\Delta$ we will just focus on the $dB_t$ terms <br>

$(\frac{\partial V}{\partial S} \sigma S) - \Delta \sigma S)dB_t =0$

$\Delta = \frac{\partial V}{\partial S}$

$d\Pi = (\frac{\partial V}{\partial S_t}\mu S +  \frac{\partial V}{\partial t} +     \frac{1}{2}\sigma^2 S_t^2  \frac{\partial^2 V}{\partial S_t^2})dt + (\frac{\partial V}{\partial S} \sigma S) dB_t- \frac{\partial V}{\partial S}[(\mu S) dt + (\sigma S)dB_t]$

Now from cancelation we can see that $d \Pi$ is equal to the following: <br>

$d\Pi = (\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial V}{\partial S})dt$ <br>

Now we need to introduce a new term $r$ that will represent a risk free rate and we can see that 

$d \Pi = r \Pi dt$

$\Pi = V - \Delta S = V - \frac{\partial V}{\partial S}S$

Now with the final step we will set the 2 equations equal to eachother and get

$(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial V}{\partial S})dt = r(V - \frac{\partial V}{\partial S}S)  dt$

Then by canceling out the $dt$ term and then moving everything to one side we are left with the famous Black-Scholes PDE

$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial V}{\partial S} + rS \frac{\partial V}{\partial S} - rV = 0$





