# Simulation_Heston_Stochastic_Volatility_Model
Simulation of the Heston Stochastic Volatility Model based on :
* Van der Stoep, A. W., Grzelak, L. A., & Oosterlee, C. W. (2014). The Heston stochastic-local volatility model: Efficient Monte Carlo simulation. International Journal of Theoretical and Applied Finance, 17(07), 1450045.
* Andersen, L. B. (2007). Efficient simulation of the Heston stochastic volatility model. Available at SSRN 946405.

## To execute
Execute the following command in folder _Surface volatility from data_ or _Heston model_ if you respectively want to compute the volatility surface based on raw data or if you want to simulate the path from Heston Model.

`g++ *.cpp ../*.cpp -o main`

to create executable file `main` and run it with `./main` in terminal.

## Implied Volatility Surface
In the market, for each asset, we observe a matrix of call prices for a given set of strikes and set of maturities $\{C^{Mkt}(T_i, K_j) \}_{i \in [[ 1, N ]], j  \in [[ 1, N ]]}$. Therefore, we can reverse the implied volatility matrix on the same set of strikes and maturities.

$$
\forall (i,j) \in [[ 1, M ]] \text{x} [[ 1, N ]], \sigma^*(T_i, K_j) = C^{-1}_{T_i, K_j} (C^{Mkt}(T_i, K_j))
$$

Then, we need to interpolate/extrapolate this volatility matrix to be able to request the implied volatility at any maturity and strike.

For each maturity, the volatility smile is often interpolated/extrapolated using natural cubic splines.

The interpolation/extrapolation along maturities has different conventions, but the one we will use is linear interpolation in total variance $(\sigma^*)² (T, K)T$ along a constant forward moneyness like

$$
k_{F_T} = \frac{K}{F_T} = \frac{K}{S_0 e^{\int_{0}^{T} r(s)ds}}
$$

### Interpolation/Extrapolation along the strikes

For each $ i \in [[ 1, M ]] $, we call the slice $(\sigma^* (T_i, K))_{K>0}$ the volatility smile at maturity $T_i$. As we only get a few points $\{ \sigma^* (T_i, K_1), ... , \sigma^* (T_i, K_N)\}$ from the market option prices, a first intuition would be to perform linear interpolation in between points.

A better idea is then to perform polynomial interpolation in between points as we need the implied volatility surface to be $C^{1,2}$, meaning once-differentiable towards the maturity variable T, and twice-differentiable towards the strike variable K.

Therefore, we use the **Natural Cubic Spline** approach.

Let's call $(x_j = K_j, y_j = \sigma^*(T_i, K_j))_{j \in [[ 1, N ]]}$ the set of points of the smile at maturity $T_i$.

$\forall j \in [[ 1, N-1 ]]$, we consider the cubic polynomial function $S_j$ defined on $[x_j, x_{j+1}]$ by :

$$
S_j(x) = \alpha_j(x - x_j)^3 + \Beta_j(x - x_j)^2 + \gamma_j(x - x_j) + \delta_j
$$

The spline interpolation function is therefore the piecewise combination of those cubic polynomials :

$$
\forall K \in [x_1, x_N], \text{ } \sigma^*(T_i, K) = S_j(K) \text{ if } K \in [x_j, x_{j+1}]
$$


Then, we use the conditions at points to find the 4 x (N-1) coefficients $\{ \alpha_j, \beta_j, \gamma_j, \delta_j \}$

** Conditions at points $x_j$ **

Firstly, let's use the fact that the spline function contains all the points given by the market :

$$
\forall j \in [[ 1, N-1 ]], S_j(x_j) = y_j \text{ and } S_j(x_{j+1}) = y_{j+1}
$$

Which fills 2(N-1) conditions.

Secondly, let's use the $C²$ property of the spline function :

$$
\forall j \in [[ 1, N-1 ]], S'_j(x_{j+1}) = S'_{j+1}(x_{j+1}) \text{ and } S''_j(x_{j+1}) = S''_{j+1}(x_{j+1})
$$

Which completes 2(N-2) conditions as well. Finally, there are 2 conditions left to make the system solvable. The most natural choice is to have **zero second order derivative** at extremities

$$
S''_1(x_1) = S''_{N-1}(x_N) = 0
$$

**Solving the conditions**

We note $\Delta x_j = x_{j+1} - x_j$, so from the first condition we have
$$
\delta_j = y_j
$$
$$
\alpha_j \Delta x³_j + \beta_j \Delta x²_j + \gamma_j \Delta x_j = y_{j+1} - y_j
$$

And the second condition gives

$$
3 \alpha_j \Delta x²_j + 2 \beta_j \Delta x_j = \gamma_{j+1} - \gamma_{j}
$$
$$
3 \alpha_j \Delta x_j = \beta_{j+1} - \beta_{j}
$$

In addition, the condition on zero second order derivative at extremities gives

$$
\beta_1 = 0
$$

$$
3 \alpha_{N-1} \Delta x_{N-1} + \beta_{N-1} = 0
$$

We can infer
$$
3 \alpha_j \Delta x^2_j = (\beta_{j+1} - \beta_{j}) \Delta x_j
\implies \\
\gamma_{j+1} - \gamma_j = (\beta_{j+1} - \beta_{j}) \Delta x_j
$$

Also, with $\alpha_j = \frac{\beta_{j+1} - \beta_{j}}{3 \Delta x}$

$$
(\beta_{j+1} + 2 \beta_{j}) \Delta x_j + 3 \gamma_j = 3 \frac{y_{j+1} - y_j}{\Delta x_j}
$$

We shift the index $j$ by $j+1$

$$
(\beta_{j+2} + 2 \beta_{j+1}) \Delta x_{j+1} + 3 \gamma_{j+1} = 3 \frac{y_{j+2} - y_{j+1}}{\Delta x_{j+1}}
$$

We obtain

$$
\forall j \in [[ 1, N-3 ]], \Beta_{j+2} \Delta x_{j+1} + 2 \beta_{j+1} (\Delta x_{j+1} + \Delta x_j) + \beta_j \Delta x_j = 3 \Bigg( \frac{y_{j+2} - y_{j+1}}{\Delta x_{j+1}} - \frac{y_{j+1} - y_j}{\Delta x_j} \Bigg)
$$

Using conditions $\beta_1 = 0$ and $\alpha_{N-1} = - \frac{\beta_{N-1}}{3 \Delta x_{N-1}}$, we deduce

$$
\Delta x_{N-2} \beta_{N-2} + 2 (\Delta x_{N-2} + \Delta x_{N-1}) \beta_{N-1} = 3 \Bigg( \frac{y_N - y_{N-1}}{\Delta x_{N-1}} - \frac{y_{N-1} - y_{N-2}}{\Delta x_{N-2}} \Bigg)
$$

**Matricial expression for the $\beta_j$**

We can rewrite the two last expressions in a matricial way :

$$
\begin{pmatrix}
  2(\Delta x_1 + \Delta x_2) & \Delta x_2 & 0 & 0 & 0 & 0 & 0\\
  \Delta x_2 & 2(\Delta x_1 + \Delta x_2) & \Delta x_3 & 0 & 0 & 0 & 0\\
  0 & ... & ... & ... & 0 & 0 & 0\\
  0 & 0 & ... & ... & ... & 0 & 0\\
  0 & 0 & 0 & ... & ... & ... & 0\\
  0 & 0 & 0 & 0 & \Delta x_{N-3} & 2 (\Delta x_{N-3} + \Delta x_{N-2}) & \Delta x_{N-2}\\
  0 & 0 & 0 & 0 & 0 & \Delta x_{N-2} & 2 (\Delta x_{N-2} + \Delta x_{N-1})\\
\end{pmatrix}
\begin{pmatrix}
  \beta_2\\
  \beta_3\\
  ...\\
  ...\\
  ...\\
  \beta_{N-2}\\
  \beta_{N-1}\\
\end{pmatrix}
$$

$$
= 3 \text{ x }
\begin{pmatrix}
  \frac{y_3 - y_2}{\Delta x_2} - \frac{y_2 - y_1}{\Delta x_1}\\
  \frac{y_4 - y_3}{\Delta x_3} - \frac{y_3 - y_2}{\Delta x_2}\\
  ...\\
  ...\\
  ...\\
  \frac{y_{N-1} - y_{N-2}}{\Delta x_{N-2}} - \frac{y_{N-2} - y_{N-3}}{\Delta x_{N-3}}\\
  \frac{y_N - y_{N-1}}{\Delta x_{N-1}} - \frac{y_{N-1} - y_{N-2}}{\Delta x_{N-2}}
\end{pmatrix}
$$

Let's denote A the squared tridiagonal symmetrical matrix of dimension (N-2), Z the unknown vector $\{Z_1 = \beta_2, ..., Z_{N-2} = \beta_{N-1}\}$ of size N-2 and $R = \{R_1, ..., R_{N-2}\}$ the right-hand side vector of the equation above.

This system is a typical application of the Thomas decomposition, given A is tridiagonal. The algorithm will be stable due to the fact that A is strictly diagonally dominant with real positive diagonal entries, therefore A is positive definite.

This algorithm allows us to find the coefficients $(\beta_j)_{j \in [[ 1, N-1 ]]}$, and therefore we can derive the coefficients $(\alpha_j)_{j \in [[ 1, N-1 ]]}$ and $(\gamma_j)_{j \in [[ 1, N-1 ]]}$

$$
\alpha_j  =
\begin{cases}
  \alpha_j = \frac{\beta_{j+1} - \beta_j}{3 \Delta x_j} \text{ if } j \in [[ 1, N-2 ]] \\
  \alpha_{N-1} = - \frac{\beta_{N-1}}{3 \Delta x_{N-1}} \text{ if } j = N-1
\end{cases}
$$

$$
\gamma_j = \frac{y_{j+1} - y_j}{\Delta x_j} - \alpha_j \Delta x²_j - \beta_j \Delta x_j
$$

**Extrapolation along the strike tails**

In the regions $K < K_1$ and $K > K_N$, we make the assumption of linear extrapolation (we prolongate the zero second-order derivative from the extreme points).

We first compute the Left and Right Derivatives:
$$
\begin{cases}
  D_L = S'(x_1) = \gamma_1\\
  D_R = S'_{N-1}(x_N) = 3 \alpha_{N-1} \Delta x²_{N-1} + 2 \beta_{N-1} \Delta x_{N-1} + \gamma_{N-1} 
\end{cases}
$$

The extrapolation formula in the tail regions become :

$$
\sigma^*(T_i, K) =
\begin{cases}
  \sigma^*(T_i, K_1) + D_L \text{ x } (K - K_1) \text{ if } K < K_1\\
  \sigma^*(T_i, K_N) + D_R \text{ x } (K - K_N) \text{ if } K > K_N
\end{cases}
$$

### Interpolation/Extrapolation along the maturities

The algorithm below assumes that all M smile functions $( \sigma^* (T_i, K))_{K \geq 0}$ have been computed by interpolation/extrapolation for all maturities $\{T_1, ..., T_M\}$ as done in the previous section.

**Interpolation along maturities**

The choice we are making for the interpolation along the maturities is a linear interpolation in variance following constant forward moneyness levels.

For a given maturity $T \in [T_i, T_{i+1}]$ and any strike $K$

1. We compute the forward moneyness level: $k_{F_T} = \frac{K}{F_T} = \frac{K}{S_0} e^{- \int^T_0 r(s) ds}$
2. We extract the strikes $K^{(i)}$ and $K^{(i+1)}$ corresponding to that forward moneyness for maturities $T_i$ and $T_{i+1}$:

$$
K^{(i)} = k_{F_T} \text{ x } S_0 e^{\int^{T_i}_0 r(s) ds}\\
K^{(i+1)} = k_{F_T} \text{ x } S_0 e^{\int^{T_{i+1}}_0 r(s) ds}
$$

3. We denote the variance quantity $v(T, k) = (\sigma^*)² (T,k \text{ x } S_0 e^{\int^T_0 r(s) ds}) \text{ x } T$
4. We get the value $v(T, k_{F_T})$ by linear interpolation of $v(T_i, k_{F_T})$ and $v(T_{i+1}, k_{F_T})$:

$$
v(T, k_{F_T}) = v(T_i, k_{F_T}) + \frac{v(T_{i+1}, k_{F_T}) - v(T_i, k_{F_T})}{T_{i+1} - T_i} \text{ x } (T - T_i)
$$

5. As a summary, the quantity $\sigma^*(T, K)$ is therefore computed by the following formula:

$$
\sigma^*(T, K) = \sqrt{\frac{1}{T} \text{ x } \Bigg( (\sigma^*)²(T_i, K^{(i)})T_i + \frac{(\sigma^*)²(T_{i+1}, K^{(i+1)})T_{i+1} - (\sigma^*)²-(T_i, K^{(i)})T_i}{T_{i+1} - T_i} (T - T_i) \Bigg)}
$$

where

$$
K^{(i)} = K e^{\int^{T_i - T}_0 r(s) ds}\\
K^{(i+1)} = K e^{\int^{T_{i+1} - T}_0 r(s) ds}
$$

6. All the quantities above are obtained thanks to the interpolation/extrapolation of all the smile functions at all maturities $\{T_i\}_{i \in [[ 1, M ]]}$

**Extrapolation along maturities**

The extrapolation of the implied volatility surface outside the range of the market input maturities can be assumed to be constant, still following a same level of forward moneyness from the extreme maturities

$$
\sigma^*(T, K) =
\begin{cases}
  \sigma^*(T_1, K^{(1)}) \text{ if } T < T_1\\
  \sigma^*(T_M, K^{(M)}) \text{ if } T > T_M
\end{cases}
$$

where

$$
K^{(1)} = K e^{\int^{T_1 - T}_0 r(s) ds}\\
K^{(M)} = K e^{\int^{T_M - T}_0 r(s) ds}
$$

## Description of the files & folder
- class_Matrix_header.h & class_Matrix.cpp: files containing the definition of matrix class and usual operations. Contains methods to perform Cholesky decomposition, (reduced) row echelon matrix transformation and algorithm to solve a Thomas system

- folder _Surface volatility from data_ : Folder containing the build of implied volatility surface
  - class_surface_vol_header.h & class_surface_vol.cpp: files containing the observation of volatility surface
  - class_tenor_cubicspline_header.h & class_tenor_cubicspline.cpp: files containing the interpolation/extrapolation along strikes and maturities as describe above using tenor cubicspline method
  - main.cpp: main file to compile which builds the implied volatility surface
  - folder data : folder containing raw data to observe
    - option_implied_vol.csv
    - option_prices.csv
    - tenor_vol_chain.csv
    - test_vol_curve.csv

- folder _Heston Model_ : Folder containing the simulation of the Heston model and its dependencies
  - class_Random_header.h & class_Random.cpp: files containing the definition of uniform and gaussian random variables
  - class_Model2F_header.h & class_Model2F.cpp: files containing the general definition of a 2 Factor model
  - class_PathSimulatorEuler2F_header.h & class_PathSimulatorEuler2F.cpp: files containing the definition of path simulator via Euler method for a 2 Factor model
  - class_HestonModel_header.h & class_HestonModel.cpp: files containing the definition of Heston model
  - main.cpp: main file to simulate the Heston model