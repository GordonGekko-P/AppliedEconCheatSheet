import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Maximum Likelihood section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Maximum Likelihood Estimation</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers maximum likelihood estimation (MLE), a powerful method for estimating
    parameters in statistical models. We'll explore the theory, computation, and applications
    of MLE in econometrics.
    """)
    
    # Create a DataFrame for MLE concepts
    concepts = {
        "Concept": [
            "Likelihood Function",
            "Log-Likelihood",
            "Score Function",
            "Information Matrix",
            "MLE Properties",
            "Numerical Methods",
            "Standard Errors",
            "Hypothesis Tests"
        ],
        "Definition": [
            "L(θ;x) = ∏ᵢf(xᵢ|θ)",
            "ℓ(θ;x) = ∑ᵢlog f(xᵢ|θ)",
            "∂ℓ/∂θ = 0",
            "-E[∂²ℓ/∂θ²]",
            "Consistent, Efficient",
            "Newton-Raphson, BHHH",
            "√diag(I⁻¹)",
            "Wald, LR, LM tests"
        ],
        "Key Features": [
            "Joint density",
            "Monotonic transformation",
            "First-order condition",
            "Variance of estimator",
            "Asymptotic normality",
            "Iterative solutions",
            "Asymptotic approximation",
            "Asymptotically equivalent"
        ],
        "Applications": [
            "Parameter estimation",
            "Computational convenience",
            "Optimization condition",
            "Efficiency bounds",
            "Inference",
            "Finding maximum",
            "Confidence intervals",
            "Model comparison"
        ]
    }
    
    # Convert to DataFrame and display as table
    df = pd.DataFrame(concepts)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Key Concepts with Tabs
    st.markdown("### 🔑 Key Concepts")
    
    tabs = st.tabs(["Likelihood Functions", "Numerical Methods", "Hypothesis Testing"])
    
    with tabs[0]:
        st.markdown("#### Likelihood Functions and Properties")
        
        # Create visualization of likelihood function
        np.random.seed(42)
        n = 50
        true_mu = 2
        true_sigma = 1.5
        
        # Generate data
        data = np.random.normal(true_mu, true_sigma, n)
        
        # Create grid for parameters
        mu_grid = np.linspace(0, 4, 100)
        sigma_grid = np.linspace(0.5, 3, 100)
        mu_mesh, sigma_mesh = np.meshgrid(mu_grid, sigma_grid)
        
        # Calculate log-likelihood
        ll = np.zeros_like(mu_mesh)
        for i in range(len(mu_grid)):
            for j in range(len(sigma_grid)):
                ll[j,i] = np.sum(stats.norm.logpdf(data, mu_grid[i], sigma_grid[j]))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Contour plot
        cs = ax1.contour(mu_mesh, sigma_mesh, ll, levels=20)
        ax1.clabel(cs, inline=1, fontsize=8)
        ax1.plot(true_mu, true_sigma, 'r*', markersize=15, label='True Value')
        ax1.set_title('Log-Likelihood Contours')
        ax1.set_xlabel('μ')
        ax1.set_ylabel('σ')
        ax1.legend()
        
        # Profile likelihood for μ
        max_ll_mu = np.max(ll, axis=0)
        ax2.plot(mu_grid, max_ll_mu)
        ax2.axvline(true_mu, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Profile Likelihood for μ')
        ax2.set_xlabel('μ')
        ax2.set_ylabel('Profile Log-Likelihood')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Likelihood Function Properties:**
        
        1. **Construction**
           ```python
           L(θ;x) = ∏ᵢf(xᵢ|θ)
           ℓ(θ;x) = ∑ᵢlog f(xᵢ|θ)
           ```
           - Product of individual densities
           - Log transformation for computation
           - Maintains same maximum
        
        2. **Score Function**
           ```python
           s(θ) = ∂ℓ/∂θ = ∑ᵢ∂log f(xᵢ|θ)/∂θ
           ```
           - First derivative of log-likelihood
           - Zero at maximum
           - Information equality
        
        3. **Information Matrix**
           ```python
           I(θ) = -E[∂²ℓ/∂θ²]
           ```
           - Negative expected Hessian
           - Variance of score
           - Efficiency bound
        
        4. **Asymptotic Properties**
           - Consistency: θ̂ →ᵖ θ₀
           - Normality: √n(θ̂-θ₀) →ᵈ N(0,I⁻¹)
           - Efficiency: Achieves CR bound
           - Invariance to transformation
        """)
    
    with tabs[1]:
        st.markdown("#### Numerical Methods")
        
        # Create visualization of Newton-Raphson
        def f(x):
            return -0.5 * (x - 2)**2  # Negative quadratic (log-likelihood shape)
        
        def f_prime(x):
            return -(x - 2)  # Derivative
        
        # Create data for visualization
        x = np.linspace(0, 4, 100)
        x_nr = [0.5]  # Starting point
        
        for _ in range(3):
            x_new = x_nr[-1] - f_prime(x_nr[-1])/(-1)  # Newton-Raphson step
            x_nr.append(x_new)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot function and iterations
        ax1.plot(x, f(x), 'b-', label='Log-Likelihood')
        for i in range(len(x_nr)-1):
            # Plot point
            ax1.plot(x_nr[i], f(x_nr[i]), 'ro')
            # Plot tangent line
            x_tang = np.linspace(x_nr[i]-0.5, x_nr[i]+0.5, 10)
            y_tang = f(x_nr[i]) + f_prime(x_nr[i])*(x_tang - x_nr[i])
            ax1.plot(x_tang, y_tang, 'g--', alpha=0.5)
            # Plot vertical line to next point
            ax1.plot([x_nr[i], x_nr[i+1]], [f(x_nr[i]), f(x_nr[i+1])], 'r:')
        
        ax1.plot(x_nr[-1], f(x_nr[-1]), 'ro')
        ax1.set_title('Newton-Raphson Method')
        ax1.set_xlabel('Parameter')
        ax1.set_ylabel('Log-Likelihood')
        ax1.grid(True, alpha=0.3)
        
        # Convergence plot
        ax2.plot(range(len(x_nr)), np.abs(np.array(x_nr) - 2), 'bo-')
        ax2.set_title('Convergence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('|θₜ - θ*|')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Numerical Optimization Methods:**
        
        1. **Newton-Raphson**
           ```python
           θₜ₊₁ = θₜ - [∂²ℓ/∂θ²]⁻¹ · ∂ℓ/∂θ
           ```
           - Quadratic convergence
           - Requires Hessian
           - Sensitive to starting values
        
        2. **BHHH Algorithm**
           ```python
           θₜ₊₁ = θₜ + [∑ᵢsᵢsᵢ']⁻¹ · ∑ᵢsᵢ
           ```
           - Uses outer product of scores
           - More stable than Newton-Raphson
           - Linear convergence
        
        3. **Quasi-Newton Methods**
           ```python
           θₜ₊₁ = θₜ + λₜHₜgₜ
           ```
           - BFGS, DFP updates
           - No exact Hessian needed
           - Superlinear convergence
        
        4. **Implementation Issues**
           - Starting values
           - Step size (line search)
           - Convergence criteria
           - Multiple maxima
        """)
    
    with tabs[2]:
        st.markdown("#### Hypothesis Testing")
        
        # Create visualization of likelihood ratio test
        np.random.seed(42)
        n = 100
        
        # Generate data
        x = np.random.normal(1.5, 1, n)
        
        # Calculate likelihood ratio test statistic
        def ll_restricted(mu):
            return np.sum(stats.norm.logpdf(x, mu, 1))
        
        def ll_unrestricted(mu, sigma):
            return np.sum(stats.norm.logpdf(x, mu, sigma))
        
        # Grid for visualization
        mu_grid = np.linspace(0, 3, 100)
        ll_r = [ll_restricted(mu) for mu in mu_grid]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot likelihood ratio test
        ax1.plot(mu_grid, ll_r, 'b-', label='Restricted (σ=1)')
        ax1.axvline(np.mean(x), color='r', linestyle='--', 
                   label='Unrestricted MLE')
        ax1.axvline(1.5, color='g', linestyle='--',
                   label='True Value')
        ax1.set_title('Likelihood Ratio Test')
        ax1.set_xlabel('μ')
        ax1.set_ylabel('Log-Likelihood')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot chi-square distribution
        x_chi = np.linspace(0, 10, 100)
        ax2.plot(x_chi, stats.chi2.pdf(x_chi, df=1), 'b-')
        ax2.fill_between(x_chi[x_chi >= 3.84], 
                        stats.chi2.pdf(x_chi[x_chi >= 3.84], df=1),
                        alpha=0.3)
        ax2.text(4, 0.1, 'Critical\nValue\n(5%)', ha='left')
        ax2.set_title('χ² Distribution (df=1)')
        ax2.set_xlabel('Test Statistic')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Hypothesis Testing in MLE:**
        
        1. **Likelihood Ratio (LR) Test**
           ```python
           LR = 2[ℓ(θ̂) - ℓ(θ̃)]
           ```
           - Compare unrestricted vs restricted
           - Asymptotically χ² distributed
           - Invariant to parameterization
        
        2. **Wald Test**
           ```python
           W = (θ̂ - θ₀)'V̂⁻¹(θ̂ - θ₀)
           ```
           - Uses unrestricted estimates
           - Requires variance matrix
           - Not invariant to transformation
        
        3. **Lagrange Multiplier (LM) Test**
           ```python
           LM = s(θ̃)'I(θ̃)⁻¹s(θ̃)
           ```
           - Uses restricted estimates
           - Score principle
           - Locally most powerful
        
        4. **Testing Strategy**
           - All asymptotically equivalent
           - LR between Wald and LM
           - Choose based on computation
           - Multiple restrictions
        """)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Duration Model Estimation**
    
    Consider modeling unemployment duration:
    
    **Step 1: Model Specification**
    ```python
    # Weibull hazard model
    h(t|x) = λαt^(α-1)exp(xβ)
    
    # Log-likelihood function
    ℓ(θ) = ∑ᵢ[dᵢlog h(tᵢ|xᵢ) - H(tᵢ|xᵢ)]
    ```
    
    **Step 2: Estimation Results**
    ```python
    Parameters:
    - α (duration dependence): 0.85 (0.05)
    - β₁ (education): -0.12 (0.03)
    - β₂ (age): 0.02 (0.01)
    - β₃ (unemployment rate): 0.15 (0.04)
    
    Log-likelihood: -1234.5
    ```
    
    **Step 3: Hypothesis Tests**
    ```python
    1. No duration dependence (H₀: α = 1)
       - LR stat = 8.5
       - p-value = 0.004
    
    2. Joint significance of covariates
       - Wald χ²(3) = 45.2
       - p-value < 0.001
    ```
    
    **Step 4: Interpretation**
    - Negative duration dependence
    - Education reduces hazard
    - Higher unemployment rate increases duration
    - Model fits well
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Simulation Methods")
    
    st.markdown("""
    Maximum likelihood estimation connects to simulation methods through:
    - Simulated maximum likelihood
    - Bootstrap standard errors
    - Monte Carlo studies
    - Numerical integration
    
    The next section will explore these simulation methods.
    """) 