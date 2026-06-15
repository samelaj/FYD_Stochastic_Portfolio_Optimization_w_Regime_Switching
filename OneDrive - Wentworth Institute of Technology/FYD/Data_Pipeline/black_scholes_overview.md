# Black-Scholes Summary

# Core Idea

Imagine you are buying the right to purchase a stock at a fixed price in the future, the right to buy apple at $100 in 3 months no mater what the price is then how much should you pay today.

# Outcome
If you continuousy adjust a portfolio of the stock and a risk free bond, you can perfectly replicate the payoff of that option. Since two things w/ identical payoffs must have the same price the replicating portfoloios cost is the options for fair price.
Note: You don't need to know how much the stock is supposed ot grow, fair price not dependent

# Assumptions
1. Stock prices follow Geometric Brownian Motion, they are continuous no umps log-normally distributed returns
2. Volatility $\sigma$ is constant over time
3. Markets are frictionless, no transaction costs, continuous trading is possible
4. A risk free rate $r$ exists and is contant
5. No dividends

# Geometric Brownian Motion
Stock price is written as the following:
$dS = \mu Sdt + \sigma S d W_{t}$ <br>
where $S$ = stock price, $\sigma$ = volatility, $\mu$ = drift, $W_{t}$ = wiener process "random noise"

# Ito's Lemma
Key math tool need to use because you can't use ordinary calculus to b/c stock prices are stochastic this adds in the 2nd order correction term to account for randomness: <br>
Term: $\frac{1}{2} \sigma^{2}S^2 \frac{\partial^{2}V}{\partial S^2} \to$ comes from randomness

# Black Scholes PDE:
When applying Ito's lemma to an option $V(s,t)$ and constucting the riskless hedging portfolio shows us: <br>
$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2S^2\frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$
<br>
Note: $\mu$ (expected return "drift") dropped out entirely $\to$ risk neutral pricing

# Closed form solution
Used for european call option w/ strike price $K$ and expiry $T$ is shown to be:

$C = S * N(d_1) - K e^{-rT} * N(d_2)$ <br>
where: <br>
$d_1 = \frac{ln(S/K) + (r+\frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$, $d_2 = d_1 - \sigma \sqrt{T}$
$N(*)$ = a cumulative normal distribution

# The Greeks
Partialderviatives ofprice formula that measure sensitivity

Delta: $\frac{\partial C}{\partial S}$ How much the option price moves per $ 1 move in the stock what you hedge with

Gamma: $\frac{\partial^2C}{\partial S^2}$ Rate of change delta, matters for dynamic hedging

Vega: $\frac{\partial C}{\partial \sigma}$ Sensitivity to volatility critical for project since $\sigma$ changes across different market regimes

Theta: $\frac{\partial C}{\partial t}$ time decay

# Effect of black-scholes on project

The black shcoles assumption of constant volitilty is exactly what breaks down in practice, markets have distinct regimes.
The HMM is detecting those regimes precisely so you can use a different volitility and drift in each one, making the optimization more realistic than just the standard black-shcoles eqn
