import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Limited Dependent Variables section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Limited Dependent Variables</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers models for limited dependent variables, including binary choice models,
    censored and truncated regression, sample selection models, and count data models.
    """)
    
    # Create a DataFrame for model types
    models = {
        "Model": [
            "Probit",
            "Logit",
            "Tobit",
            "Heckman Selection",
            "Ordered Probit",
            "Multinomial Logit",
            "Poisson",
            "Negative Binomial"
        ],
        "Application": [
            "Binary outcomes (0/1)",
            "Binary outcomes with fat tails",
            "Censored/truncated data",
            "Sample selection bias",
            "Ordered categorical",
            "Unordered categorical",
            "Count data",
            "Overdispersed counts"
        ],
        "Key Features": [
            "Normal errors",
            "Logistic errors",
            "Corner solution",
            "Two-step estimation",
            "Multiple thresholds",
            "Independence of alternatives",
            "Mean = variance",
            "Variance > mean"
        ],
        "Estimation": [
            "MLE",
            "MLE",
            "MLE with censoring",
            "Two-step or MLE",
            "MLE with cutpoints",
            "MLE with normalization",
            "MLE",
            "MLE with dispersion"
        ]
    }
    
    # Convert to DataFrame and display as table
    df = pd.DataFrame(models)
    
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
    
    tabs = st.tabs(["Binary Choice", "Sample Selection", "Count Models"])
    
    with tabs[0]:
        st.markdown("#### Binary Choice Models")
        
        # Create visualization of probit vs logit
        np.random.seed(42)
        x = np.linspace(-4, 4, 100)
        
        # Calculate probabilities
        probit = stats.norm.cdf(x)
        logit = 1 / (1 + np.exp(-x))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot probability functions
        ax1.plot(x, probit, 'b-', label='Probit')
        ax1.plot(x, logit, 'r--', label='Logit')
        ax1.set_title('Probability Functions')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot difference
        ax2.plot(x, logit - probit, 'g-')
        ax2.set_title('Logit - Probit Difference')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Difference')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Binary Choice Models:**
        
        1. **Probit Model**
           ```python
           P(y = 1|x) = Φ(xβ)
           ```
           - Φ is standard normal CDF
           - Coefficients are harder to interpret
           - Marginal effects vary with x
        
        2. **Logit Model**
           ```python
           P(y = 1|x) = exp(xβ)/(1 + exp(xβ))
           ```
           - Logistic distribution
           - Odds ratios are constant
           - Better for extreme values
        
        3. **Interpretation**
           - Sign and significance
           - Marginal effects at means
           - Average marginal effects
           - Predicted probabilities
        
        4. **Model Choice**
           - Similar in middle range
           - Logit has heavier tails
           - Probit more common in economics
           - Consider link test
        """)
    
    with tabs[1]:
        st.markdown("#### Sample Selection Models")
        
        # Create visualization of selection bias
        np.random.seed(42)
        n = 200
        
        # Generate data
        X = np.random.normal(0, 1, n)
        error = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], n)
        Z = X + error[:, 0]
        Y_star = 2 * X + error[:, 1]
        
        # Selection rule
        selected = Z > 0
        Y = Y_star[selected]
        X_obs = X[selected]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Full sample
        ax1.scatter(X, Y_star, alpha=0.5, label='Full Sample')
        beta_full = np.polyfit(X, Y_star, 1)
        x_range = np.linspace(X.min(), X.max(), 100)
        ax1.plot(x_range, beta_full[0] * x_range + beta_full[1], 'r-', 
                label=f'β = {beta_full[0]:.2f}')
        ax1.set_title('Full Sample')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Selected sample
        ax2.scatter(X_obs, Y, alpha=0.5, label='Selected Sample')
        beta_sel = np.polyfit(X_obs, Y, 1)
        x_range = np.linspace(X_obs.min(), X_obs.max(), 100)
        ax2.plot(x_range, beta_sel[0] * x_range + beta_sel[1], 'r-',
                label=f'β = {beta_sel[0]:.2f}')
        ax2.set_title('Selected Sample')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Sample Selection Models:**
        
        1. **Heckman Two-Step**
           ```python
           Selection: z*ᵢ = wᵢγ + uᵢ
           Outcome:   yᵢ = xᵢβ + εᵢ
           ```
           - First step: Probit for selection
           - Second step: OLS with correction
           - Inverse Mills ratio as control
        
        2. **Maximum Likelihood**
           ```python
           L = ∏ᵢ[Φ(wᵢγ)ϕ(yᵢ-xᵢβ)/σ]ᵢ∈s × ∏ᵢ[1-Φ(wᵢγ)]ᵢ∉s
           ```
           - More efficient than two-step
           - Requires distributional assumptions
           - Can be sensitive to specification
        
        3. **Identification**
           - Exclusion restrictions
           - Nonlinearity of Mills ratio
           - Strong first stage
           - Test for selection bias
        """)
    
    with tabs[2]:
        st.markdown("#### Count Data Models")
        
        # Create visualization of count distributions
        np.random.seed(42)
        n = 1000
        
        # Generate data
        poisson_lambda = 2
        nb_r = 2  # Number of successes for negative binomial
        nb_p = 0.5  # Probability of success
        
        poisson_data = np.random.poisson(poisson_lambda, n)
        nb_data = np.random.negative_binomial(nb_r, nb_p, n)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Poisson distribution
        ax1.hist(poisson_data, bins=range(max(poisson_data)+2), density=True, 
                alpha=0.6, label='Data')
        ax1.set_title(f'Poisson(λ={poisson_lambda})\nMean = Variance = {poisson_lambda}')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Density')
        
        # Negative Binomial distribution
        ax2.hist(nb_data, bins=range(max(nb_data)+2), density=True,
                alpha=0.6, label='Data')
        ax2.set_title('Negative Binomial\nOverdispersion')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Count Data Models:**
        
        1. **Poisson Regression**
           ```python
           E[y|x] = exp(xβ)
           Var(y|x) = E[y|x]
           ```
           - Equidispersion assumption
           - Conditional mean parameterization
           - Exponential mean function
        
        2. **Negative Binomial**
           ```python
           Var(y|x) = E[y|x] + α·E[y|x]²
           ```
           - Allows for overdispersion
           - α is dispersion parameter
           - Nests Poisson when α = 0
        
        3. **Zero-Inflated Models**
           ```python
           P(y = 0) = π + (1-π)exp(-λ)
           ```
           - Mixture model approach
           - Separate process for zeros
           - Test for zero-inflation
        
        4. **Model Selection**
           - LR test for overdispersion
           - Vuong test for zero-inflation
           - AIC/BIC comparison
           - Residual analysis
        """)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Labor Force Participation**
    
    Consider modeling female labor force participation:
    
    **Step 1: Model Specification**
    ```python
    # Probit model
    P(Work = 1) = Φ(β₀ + β₁Education + β₂Age + β₃Children + β₄Income_spouse)
    
    # Sample selection for wages
    Participation: Work* = γ₀ + γ₁Z + u
    Wage equation: log(Wage) = β₀ + β₁X + ρσλ(γ'Z) + ε
    ```
    
    **Step 2: Estimation Results**
    ```python
    Probit Results:
    - Education: 0.15 (0.02)
    - Age: 0.02 (0.01)
    - Children: -0.25 (0.05)
    - Income_spouse: -0.10 (0.03)
    
    Heckman Selection:
    - Education (wage): 0.08 (0.01)
    - Experience: 0.03 (0.01)
    - Mills ratio: -0.25 (0.10)
    ```
    
    **Step 3: Interpretation**
    - Education increases participation probability
    - Children decrease participation
    - Negative selection bias in wages
    - 8% return to education (corrected)
    
    **Step 4: Diagnostics**
    - Link test for specification
    - LR test for selection
    - Predicted probabilities
    - Marginal effects analysis
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Maximum Likelihood")
    
    st.markdown("""
    Limited dependent variable models connect to maximum likelihood estimation through:
    - MLE as primary estimation method
    - Numerical optimization techniques
    - Specification testing
    - Standard error calculation
    
    The next section will explore these maximum likelihood methods.
    """) 