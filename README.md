# Quantitative Regularization in Linear Regression

## Objective  
This notebook models and tests the **four key estimators** in linear regression:  
**Ordinary Least Squares (OLS)** ; **Ridge (L2)** ; **Lasso (L1)** ; **ElasticNet (L1 + L2)**.  
Starting from the classical OLS formulation, we progressively introduce regularization terms and study their **mathematical derivations**, **gradient minimization steps**, and **quantitative impacts**.  

While `scikit-learn` performs these optimizations internally, we explicitly derive the **cost functions**, **partial derivatives (∂)**, and **optimality conditions** to understand what happens *inside the black box*.

![alt text](scheme-1.png)
---

## Gradient Principle  

At the heart of all optimization problems lies the **additivity of derivatives**:  
if a cost function is composed of two differentiable parts,
L(θ) = f(θ) + g(θ)

then its gradient satisfies:

∂L(θ)/∂θ = ∂f(θ)/∂θ + ∂g(θ)/∂θ

This rule allows us to decompose each regularized regression objective into:
- a **data-fitting term** $f(\beta) = \|y - X\beta\|_2^2$
- and a **penalization term** $g(\beta) = \lambda \|\beta\|_p^p$,  
    where $p = 1$ (Lasso) or $p = 2$ (Ridge).

The total derivative is given by:

∂J(β)/∂β = ∂/∂β ‖y − Xβ‖² + ∂/∂β [ λ‖β‖ᵖᵖ ]
and the first-order optimality condition
∂J(β)/∂β = 0
defines the analytical or subgradient solutions for **OLS**, **Ridge**, **Lasso**, and **ElasticNet**.


---

## Mathematical Framework  

We consider the generalized penalized least squares problem:
$$\min_{\beta} \; J(\beta) = \|y - X\beta\|_2^2 + 
\lambda \big[(1-\gamma)\|\beta\|_2^2 + \gamma\|\beta\|_1\big]$$
where  
- $\lambda > 0$ controls the **regularization strength**,  
- $\gamma \in [0,1]$ interpolates between Ridge (γ=0) and Lasso (γ=1).

---

### 1. Ordinary Least Squares (OLS)
J(β) = ‖y − Xβ‖²  
Minimization condition:  
∂J/∂β = −2Xᵀ(y − Xβ) = 0  ⇒  β̂₍OLS₎ = (XᵀX)⁻¹Xᵀy  
OLS minimizes only the residual sum of squares → **unbiased** but **unstable** under multicollinearity.

---

### 2. Ridge Regression (L2 Regularization)
J(β) = ‖y − Xβ‖² + λ‖β‖²  
Setting the derivative to zero:  
∂J/∂β = −2Xᵀ(y − Xβ) + 2λβ = 0  
⇒ (XᵀX + λI)β̂₍Ridge₎ = Xᵀy  
Ridge adds an **L2 penalty** that **shrinks coefficients** and **reduces variance**.

---

### 3. Lasso Regression (L1 Regularization)
J(β) = ‖y − Xβ‖² + λ‖β‖₁  
Since |β| is not differentiable at zero, we use **subgradients**:  
∂J/∂βⱼ = −2xⱼᵀ(y − Xβ) + λ sign(βⱼ)  
At the minimum:  
0 ∈ ∂J/∂βⱼ  ⇒  some β̂ⱼ = 0  
Lasso performs **automatic variable selection** by setting some coefficients exactly to zero.

---

### 4. ElasticNet (L1 + L2 Regularization)
J(β) = ‖y − Xβ‖² + λ[(1 − γ)‖β‖² + γ‖β‖₁]  
Gradient with respect to β:  
∂J/∂β = −2Xᵀ(y − Xβ) + 2λ(1 − γ)β + λγ sign(β)  
ElasticNet blends **Lasso’s sparsity** and **Ridge’s stability**, making it **robust under correlated predictors**.

---

## Methods  
1. **Feature standardization** (mean = 0, std = 1) to ensure homogeneous penalization.  
2. **K-fold cross-validation** to find optimal $\lambda$ and $\gamma$.  
3. **Model comparison**: OLS, Ridge, Lasso, ElasticNet on both **synthetic** and **real (Diabetes)** datasets.  
4. **Analytical vs algorithmic** perspective:  
     - explicit gradient derivations (∂J/∂β = 0),  
     - empirical results using `scikit-learn`.  
5. **Evaluation metrics:** MSE, RMSE, and coefficient behavior under increasing regularization.

---

## Main Takeaway  
By moving from OLS ; Ridge ; Lasso ; ElasticNet,  we demonstrate mathematically and empirically how the **partial derivatives (∂J/∂β)** evolve with each penalty,   how **λ** and **γ** reshape the cost landscape,  and how regularization enhances **numerical stability**, **predictive robustness**, and **interpretability**.
