# Meeting 1 Outline

## Topics:
- Stochastic Processes
- SDE's
- Basic Financial/Trading Concepts
- Black Scholes Model
- Data Pipeline Starting

## Stochastic Processes
Main Idea: Predicting the future based on the past

Parameterized collection of random variables<br>
Written as ${x_t}$ $t \in T$

Can be discrete or continuous


E.g. <br>
Simple random walk <br>
Let $Y_i$ be indpendent identically distributed, and either equal 1 or -1 each with an even probability of 1/2. 

We can tell certain things about this as $t$ gets very large such that, the var = $t$ and that standard deviation = $\sqrt{t}$ we know that the walk will most likely be in the interval $[\sqrt{-t}, \sqrt{t}]$

1. $Ex_t = 0$
2. Mutually Indpedent (What happens in the beginning is irrelevant to the far future)

E.g <br>
Brownian Motion <br>
Intially it is the random motion of particles suspended in a medium<br>
Def: Fix $x \in \mathbb{R}^n$<br> $P(t,x,y) = (2 \pi t)^{\frac{-n}{2}} * \exp(\frac{- |x-y|^2}{2t})$


1. This follows a normal distribution with a mean of 0 and a variance that is proportinal to $\Delta t$

Key to project: This is because stock prices follow Geometric Brownian Motion

## SDE's

Main Idea: Allow for randomness within the coefficents <br>
Goal: Obtain a more realistic approach of the situation <br>
Connection to Project: The stock market is random we need a way to model this the most optimial appraoch is using SDE's such as the black scholes equation to do so. <br>


E.g. Simple Population Growth

$\frac{dN}{dt} = a(t)N(t)$ <br>
Where: 
$a(t) = r(t) +$ "noise"

## Basic Finance/Trading Concepts
American Option: This is a stock option that can be excercised at anytime up to the date that the option expires
- Continuous Mathematically <br>


European Option: A stock option that can only be excercised on a specific date
- Discrete Mathematically

Call option: A coupon to giving you the right to buy a stock at a fixed price, the more the stock is worth above the fixed price the more valuable the option.

- Real life example: If a stock is trading at $150 & you have the option to buy at $100 you will use and profit $50

## Black Scholes Model

### Assumptions:
1. Short term interest rate is known & constant
2. The stock price follows a random walk in continous time w/ a variance rate proportional to the square of the stock price. Therefore the distribution of possible stock prices at the end of any cycle is approximently normally distributed.
3. The option is european
4. Stock pays no dividends
5. No transaction costs in buying or selling the stock or the option
6. It is possible to borrow any fraction of the price of a security to buy or hold it at the short term interest rate
7. There is no penalty for short selling

### Assumptions Explained:
2. Stock Price random walk (Geometric Brownian Motion)
- $dS = \mu S dt + \sigma S dW_t$
<br>
Where: $S$ = Stock Price, $\sigma$ = volatility, $\mu$ = drift (expected return), $W_t$ = random noise

### Key Idea "Ito's Lemma"
Key math tool need to use because you can't use ordinary calculus to b/c stock prices are stochastic this adds in the 2nd order correction term to account for randomness: <br>
Term: $\frac{1}{2} \sigma^{2}S^2 \frac{\partial^{2}V}{\partial S^2} \to$ comes from randomness



### Black Scholes PDE:
When applying Ito's lemma to an option $V(s,t)$ and constucting the riskless hedging portfolio shows us: <br>
$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2S^2\frac{\partial^2V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$
<br>
Note: $\mu$ (expected return "drift") dropped out entirely $\to$ risk neutral pricing

### Closed form solution
Used for european call option w/ strike price $K$ and expiry $T$ is shown to be:

$C = S * N(d_1) - K e^{-rT} * N(d_2)$ <br>
where: <br>
$d_1 = \frac{ln(S/K) + (r+\frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}$, $d_2 = d_1 - \sigma \sqrt{T}$
$N(*)$ = a cumulative normal distribution

### The Greeks
Partial derviatives of price formula that measure sensitivity

Delta: $\frac{\partial C}{\partial S}$ How much the option price moves per $ 1 move in the stock what you hedge with

Gamma: $\frac{\partial^2C}{\partial S^2}$ Rate of change delta, matters for dynamic hedging

Vega: $\frac{\partial C}{\partial \sigma}$ Sensitivity to volatility critical for project since $\sigma$ changes across different market regimes

Theta: $\frac{\partial C}{\partial t}$ time decay

### Effect of black-scholes on project

The black shcoles assumption of constant volitilty is exactly what breaks down in practice, markets have distinct regimes.
The HMM is detecting those regimes precisely so you can use a different volitility and drift in each one, making the optimization more realistic than just the standard black-shcoles equation

## Data Pipeline

Asset Universe (Tickers, Asset Classes, Benchmarks) $\to$ Data Injestion (yfinance) Daily going back 15 years $\to$ Feature Engineering (log returns, rolling vol, correlation, handling)
