# Proof of Ito's Formula (Ito's Lemma)
## Motivation
Used crucially within Black Scholes Equation <br>
Considered chain rule in ito calculus

## Proof
Let $u$ and $v$ be elementary functions

$dX_t = udt + vdB_t$

$g(t, X_t) = g(0, X_0) + \int_{0}^{t} (\frac{\partial g}{\partial s}(s, X_s) + u_s \frac{\partial g}{\partial x} (s, X_s) + \frac{1}{2}v_s^2 \frac{\partial^2g}{\partial x^2}(s,X_s))ds + \int_0^t v_s \frac{\partial g}{\partial x}(s,X_s)dB_s$ <br>
Where: $u_s = u(s,w), v_s = v(s,w)$

Use taylors theorm

$g(t, X_t) = g(0, X_0) + \sum_j \Delta g(t_j, X_j)$ <br>

$= g(0,X_0) + \sum_j \frac{\partial g}{\partial t} \Delta t_j +  \sum_j \frac{\partial g}{\partial t} \Delta X_j + \frac{1}{2}  \sum_j \frac{\partial^2 g}{\partial t^2} (\Delta t_j)^2 + \sum_j \frac{\partial^2g}{\partial t \partial x}(\Delta t_j)(\Delta X_j) + \frac{1}{2} \sum_j \frac{\partial^2g}{\partial x^2}(\Delta X_j)^2 + \sum_j R_j$ <br> <br>
Where: $\Delta t_j = t_{j+1} - t_j, \Delta X_j = X_{t_{j+1}} - X_{t_{j}}, \Delta g(t_j, X_j) = g(t_{j+1}, X_{t_{j+1}} - g(t_j, X_j) R_j = o(|\Delta t_j|^2 + |\Delta X_j|^2) \forall_j$

## If $\Delta t_j \to 0$ <br>

$\sum_j \frac{\partial g}{\partial t} \Delta t_j = \sum_j \frac{\partial g}{\partial t}(t_j, X_j) \Delta t_j \to \int_0^t \frac{\partial g}{\partial t}(s,X_s)ds$

$\sum_j \frac{\partial g}{\partial x} \Delta X_j = \sum_j \frac{\partial g}{\partial x} (t_j, X_j) \Delta X_j \to \int_0^t \frac{\partial g}{\partial x} (s,X_s)dX_s$


Since $u$ and $v$ are elementary functions

$\sum_j \frac{\partial^2g}{\partial x^2} (\Delta X_j)^2 = \sum_j \frac{\partial^2g}{\partial x^2}u^2_j(\Delta t_j)^2 + 2 \sum_j \frac{\partial^2 g}{\partial x^2}u_jv_j(\Delta t_j)(\Delta B_j) + \sum_j \frac{\partial^2g}{\partial x^2}v_j^2(\Delta B_j)^2$ <br>

Where: $u_j = u(t_j, w), v_j = v(t_j, w)$

Now the first two terms in the summation shown above as $\Delta t_j \to 0$ We can illustrate this by showing the following

$E[(\sum \frac{\partial^2g}{\partial x^2}u_j v_j (\Delta t_j)(\Delta B_j))^2] = \sum_j E[(\frac{\partial^2 g}{\partial x^2} u_jv_j)^2](\Delta t_j)^3 \to 0$ as $\Delta t_j \to 0$ <br>

Now the last term in the summation shown as $\Delta t_j \to 0$ trends too:

$\int_0^t \frac{\partial^2g}{\partial x^2}v^2 ds$

This occurs from the following

$a(t) = \frac{\partial^2g}{\partial x^2} (t,X_t)v^2(t,w), a_j = a(t_j)$

$E[(\sum_ja_j(\Delta B_j)^2 - \sum_j a_j\Delta t_j)^2]$

$= \sum_{i,j}E[a_ia_j((\Delta B_i)^2-\Delta t_i)((\Delta B_j)^2- \Delta t_j))]$

If $i \ne j$ then all of the terms will vanish within this case and nothing will remain, so we are left with the following. 

$\sum_j E[a_j^2((\Delta B_j)^2 - \Delta t_j)^2] = \sum_j E[a_j^2] \cdot E[(\Delta B_j)^4 - 2(\Delta B_j)^2 \Delta t_j + \Delta (t_j)^2]$

$= \sum_j E[a_j^2] \cdot (3(\Delta t_j)^2 - 2(\Delta t_j)^2 + (\Delta t_j)^2) = 2 \sum_j E[a_j^2] \cdot (\Delta t_j)^2 \to 0$ as $\Delta t_j \to 0$

So, $\sum_j a_j (\Delta B_j)^2 \to \int_0^t a(s) ds$


