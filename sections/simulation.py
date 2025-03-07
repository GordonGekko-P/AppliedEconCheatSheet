import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Simulation Methods section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Simulation Methods</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers simulation methods in econometrics, including Monte Carlo simulation,
    bootstrap methods, and numerical integration techniques. These methods are essential for
    understanding estimator properties and conducting inference.
    """)
    
    # Create a DataFrame for simulation concepts
    concepts = {
        "Method": [
            "Monte Carlo",
            "Bootstrap",
            "Jackknife",
            "Importance Sampling",
            "Gibbs Sampling",
            "Metropolis-Hastings",
            "Simulated MLE",
            "Method of Simulated Moments"
        ],
        "Purpose": [
            "Study estimator properties",
            "Estimate standard errors",
            "Bias reduction",
            "Efficient integration",
            "Sample from conditionals",
            "General MCMC",
            "Complex likelihood",
            "Moment conditions"
        ],
        "Key Features": [
            "Repeated sampling",
            "Resampling data",
            "Leave-one-out",
            "Weighted sampling",
            "Conditional draws",
            "Accept-reject",
            "Unbiased simulator",
            "Simulated moments"
        ],
        "Applications": [
            "Size and power",
            "Confidence intervals",
            "Robust inference",
            "Rare event simulation",
            "Bayesian inference",
            "Complex posteriors",
            "Discrete choice",
            "GMM estimation"
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
    
    tabs = st.tabs(["Monte Carlo", "Bootstrap", "MCMC Methods"])
    
    with tabs[0]:
        st.markdown("#### Monte Carlo Simulation")
        
        # Create visualization of Monte Carlo simulation
        np.random.seed(42)
        
        # Generate multiple samples and calculate means
        n_samples = 1000
        sample_sizes = [5, 10, 30, 100]
        means = {n: [] for n in sample_sizes}
        
        for n in sample_sizes:
            for _ in range(n_samples):
                sample = np.random.normal(0, 1, n)
                means[n].append(np.mean(sample))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot sampling distributions
        for n in sample_sizes:
            ax1.hist(means[n], bins=30, density=True, alpha=0.3,
                    label=f'n={n}')
        x = np.linspace(-1, 1, 100)
        ax1.plot(x, stats.norm.pdf(x, 0, 1/np.sqrt(min(sample_sizes))),
                'r--', alpha=0.5)
        ax1.set_title('Sampling Distributions')
        ax1.set_xlabel('Sample Mean')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Plot convergence
        std_errors = [np.std(means[n]) for n in sample_sizes]
        ax2.plot(sample_sizes, std_errors, 'bo-')
        ax2.plot(sample_sizes, [1/np.sqrt(n) for n in sample_sizes],
                'r--', alpha=0.5, label='Theory')
        ax2.set_title('Standard Error Convergence')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Standard Error')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Monte Carlo Simulation Steps:**
        
        1. **Experimental Design**
           ```python
           # Set parameters
           n_sims = 1000
           sample_sizes = [100, 500, 1000]
           beta_true = 1.0
           
           # Storage for results
           results = {n: [] for n in sample_sizes}
           ```
        
        2. **Data Generation**
           ```python
           # Generate data
           X = np.random.normal(0, 1, n)
           Y = beta_true * X + np.random.normal(0, 1, n)
           
           # Estimate model
           beta_hat = np.sum(X * Y) / np.sum(X * X)
           ```
        
        3. **Results Analysis**
           ```python
           # Calculate properties
           bias = np.mean(betas) - beta_true
           mse = np.mean((betas - beta_true)**2)
           size = np.mean(p_values < 0.05)
           ```
        
        4. **Visualization**
           - Sampling distributions
           - Convergence rates
           - Size and power
           - Monte Carlo error
        """)
    
    with tabs[1]:
        st.markdown("#### Bootstrap Methods")
        
        # Create visualization of bootstrap
        np.random.seed(42)
        
        # Generate original sample
        n = 50
        original_data = np.random.normal(0, 1, n)
        original_mean = np.mean(original_data)
        
        # Bootstrap
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_data, size=n,
                                             replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot bootstrap distribution
        ax1.hist(bootstrap_means, bins=30, density=True, alpha=0.6)
        ax1.axvline(original_mean, color='r', linestyle='--',
                   label='Sample Mean')
        ax1.axvline(np.percentile(bootstrap_means, 2.5), color='g',
                   linestyle=':', label='95% CI')
        ax1.axvline(np.percentile(bootstrap_means, 97.5), color='g',
                   linestyle=':')
        ax1.set_title('Bootstrap Distribution')
        ax1.set_xlabel('Mean')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Plot single bootstrap sample
        bootstrap_sample = np.random.choice(original_data, size=n,
                                          replace=True)
        ax2.scatter(range(n), original_data, alpha=0.5,
                   label='Original')
        ax2.scatter(range(n), bootstrap_sample, alpha=0.5,
                   label='Bootstrap')
        ax2.set_title('Bootstrap Resampling')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Bootstrap Methods:**
        
        1. **Nonparametric Bootstrap**
           ```python
           # Resample with replacement
           bootstrap_sample = np.random.choice(data, size=n, replace=True)
           
           # Calculate statistic
           theta_boot = statistic(bootstrap_sample)
           ```
        
        2. **Parametric Bootstrap**
           ```python
           # Estimate model
           params = fit_model(data)
           
           # Generate new sample
           bootstrap_sample = generate_data(params, n)
           ```
        
        3. **Block Bootstrap**
           ```python
           # For time series
           blocks = [data[i:i+b] for i in range(0, n-b+1)]
           bootstrap_sample = np.concatenate(np.random.choice(blocks, size=n//b))
           ```
        
        4. **Wild Bootstrap**
           ```python
           # For heteroskedasticity
           weights = np.random.choice([-1, 1], size=n)
           bootstrap_residuals = residuals * weights
           ```
        """)
    
    with tabs[2]:
        st.markdown("#### MCMC Methods")
        
        # Create visualization of MCMC
        np.random.seed(42)
        
        # Generate Metropolis-Hastings example
        n_iter = 1000
        x = np.zeros(n_iter)
        x[0] = 0  # Starting value
        
        # Target distribution: N(0,1)
        def target(x):
            return np.exp(-0.5 * x**2)
        
        # Proposal: random walk
        for i in range(1, n_iter):
            proposal = x[i-1] + np.random.normal(0, 0.5)
            ratio = target(proposal) / target(x[i-1])
            if np.random.random() < ratio:
                x[i] = proposal
            else:
                x[i] = x[i-1]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot chain
        ax1.plot(x, alpha=0.6)
        ax1.set_title('MCMC Chain')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Parameter')
        ax1.grid(True, alpha=0.3)
        
        # Plot distribution
        ax2.hist(x[100:], bins=30, density=True, alpha=0.6)
        x_range = np.linspace(-3, 3, 100)
        ax2.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r--',
                label='Target')
        ax2.set_title('Posterior Distribution')
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **MCMC Methods:**
        
        1. **Metropolis-Hastings**
           ```python
           # Proposal and acceptance
           proposal = current + np.random.normal(0, sigma)
           ratio = target(proposal) / target(current)
           if np.random.random() < ratio:
               current = proposal
           ```
        
        2. **Gibbs Sampling**
           ```python
           # Sample from conditionals
           for i in range(n_params):
               theta[i] = sample_conditional(theta, i)
           ```
        
        3. **Hamiltonian Monte Carlo**
           ```python
           # Leapfrog steps
           for _ in range(L):
               momentum += 0.5 * gradient(position)
               position += momentum
               momentum += 0.5 * gradient(position)
           ```
        
        4. **Diagnostics**
           - Trace plots
           - Autocorrelation
           - Effective sample size
           - Gelman-Rubin statistic
        """)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Bootstrap Inference for Quantile Regression**
    
    Consider estimating confidence intervals for quantile regression:
    
    **Step 1: Original Estimation**
    ```python
    # Fit quantile regression
    model = QuantReg(y ~ x)
    result = model.fit(q=0.75)
    
    # Original estimates
    beta_hat = result.params
    ```
    
    **Step 2: Bootstrap Procedure**
    ```python
    n_boot = 1000
    boot_betas = np.zeros((n_boot, k))
    
    for i in range(n_boot):
        # Resample data
        idx = np.random.choice(n, size=n, replace=True)
        y_boot, x_boot = y[idx], x[idx]
        
        # Refit model
        boot_result = QuantReg(y_boot ~ x_boot).fit(q=0.75)
        boot_betas[i] = boot_result.params
    ```
    
    **Step 3: Confidence Intervals**
    ```python
    # Percentile method
    ci_lower = np.percentile(boot_betas, 2.5, axis=0)
    ci_upper = np.percentile(boot_betas, 97.5, axis=0)
    
    # Standard errors
    boot_se = np.std(boot_betas, axis=0)
    ```
    
    **Step 4: Results**
    - Point estimates with bootstrap SEs
    - Percentile confidence intervals
    - Visualization of bootstrap distribution
    - Diagnostic checks
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Nonparametric Methods")
    
    st.markdown("""
    Simulation methods connect to nonparametric methods through:
    - Bootstrap for kernel estimation
    - Simulation-based bandwidth selection
    - Resampling for density estimation
    - Monte Carlo cross-validation
    
    The next section will explore these nonparametric methods.
    """) 