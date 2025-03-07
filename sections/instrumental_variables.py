import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Instrumental Variables section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Instrumental Variables</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers instrumental variables (IV) methods, a powerful approach for dealing with
    endogeneity in econometric analysis. We'll explore the theory, implementation, and diagnostic
    tests for IV estimation.
    """)
    
    # Create a DataFrame for IV concepts
    concepts = {
        "Concept": [
            "Endogeneity",
            "Instrumental Variable",
            "2SLS Estimation",
            "GMM Estimation",
            "LIML Estimation",
            "Weak Instruments",
            "Overidentification",
            "Local Average Treatment Effect"
        ],
        "Definition": [
            "Correlation between regressor and error term",
            "Variable correlated with X but not with ε",
            "Two-stage regression approach",
            "Efficient with heteroskedasticity",
            "Maximum likelihood alternative to 2SLS",
            "Instruments poorly correlated with X",
            "More instruments than endogenous variables",
            "Treatment effect for compliers"
        ],
        "Key Requirements": [
            "Cov(X,ε) ≠ 0",
            "Relevance and Exclusion",
            "First stage significance",
            "Moment conditions",
            "Normal distribution",
            "F-stat > 10",
            "J-test p-value > 0.05",
            "Monotonicity"
        ],
        "Tests/Diagnostics": [
            "Hausman test",
            "First-stage F-test",
            "Cragg-Donald F-stat",
            "J-test",
            "Anderson-Rubin test",
            "Stock-Yogo test",
            "Sargan-Hansen test",
            "Complier analysis"
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
    
    tabs = st.tabs(["IV Intuition", "2SLS Process", "Diagnostic Tests"])
    
    with tabs[0]:
        st.markdown("#### IV Intuition and Requirements")
        
        # Create visualization of IV intuition
        np.random.seed(42)
        n = 100
        
        # Generate data
        Z = np.random.normal(0, 1, n)  # Instrument
        X = 0.7 * Z + np.random.normal(0, 0.5, n)  # Endogenous variable
        epsilon = np.random.normal(0, 0.3, n)  # Error term
        Y = 2 * X + epsilon  # Outcome
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Z vs X (Relevance)
        ax1.scatter(Z, X, alpha=0.5)
        ax1.set_title('Instrument Relevance\nZ vs X')
        ax1.set_xlabel('Instrument (Z)')
        ax1.set_ylabel('Endogenous Variable (X)')
        
        # Add regression line
        z_range = np.linspace(Z.min(), Z.max(), 100)
        beta = np.polyfit(Z, X, 1)
        ax1.plot(z_range, beta[0] * z_range + beta[1], 'r-', alpha=0.7)
        
        # Plot Z vs epsilon (Exclusion)
        ax2.scatter(Z, epsilon, alpha=0.5)
        ax2.set_title('Exclusion Restriction\nZ vs ε')
        ax2.set_xlabel('Instrument (Z)')
        ax2.set_ylabel('Error Term (ε)')
        
        # Add horizontal line at y=0
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **IV Requirements:**
        
        1. **Relevance**: Instrument must be correlated with endogenous variable
           - First stage: X = π₀ + π₁Z + v
           - Test H₀: π₁ = 0 (F-test > 10)
        
        2. **Exclusion**: Instrument affects Y only through X
           - Cov(Z,ε) = 0
           - Not directly testable
        
        3. **Independence**: Instrument is as good as randomly assigned
           - Z ⊥ (Y₀,Y₁,X₀,X₁)
           - Check balance of covariates
        
        4. **Monotonicity**: Instrument affects X in one direction
           - If Z₁ > Z₀ → X₁ ≥ X₀ for all units
           - Important for LATE interpretation
        """)
    
    with tabs[1]:
        st.markdown("#### Two-Stage Least Squares (2SLS)")
        
        # Create visualization of 2SLS process
        np.random.seed(42)
        n = 100
        
        # Generate data for 2SLS example
        Z = np.random.normal(0, 1, n)
        X = 0.7 * Z + np.random.normal(0, 0.5, n)
        Y = 2 * X + np.random.normal(0, 0.3, n)
        
        # First stage
        X_hat = np.polyval(np.polyfit(Z, X, 1), Z)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # First stage
        ax1.scatter(Z, X, alpha=0.5, label='Observed')
        ax1.plot(Z, X_hat, 'r-', alpha=0.7, label='Fitted')
        ax1.set_title('First Stage\nX = π₀ + π₁Z + v')
        ax1.set_xlabel('Instrument (Z)')
        ax1.set_ylabel('Endogenous Variable (X)')
        ax1.legend()
        
        # Second stage
        ax2.scatter(X_hat, Y, alpha=0.5, label='Using X̂')
        beta_iv = np.polyfit(X_hat, Y, 1)
        x_range = np.linspace(X_hat.min(), X_hat.max(), 100)
        ax2.plot(x_range, beta_iv[0] * x_range + beta_iv[1], 'r-', alpha=0.7, label='IV Estimate')
        ax2.set_title('Second Stage\nY = β₀ + β₁X̂ + ε')
        ax2.set_xlabel('Fitted Values (X̂)')
        ax2.set_ylabel('Outcome (Y)')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **2SLS Process:**
        
        1. **First Stage**
           ```python
           X̂ = π₀ + π₁Z + v
           ```
           - Regress endogenous X on instrument Z
           - Save fitted values X̂
           - Check first-stage F-statistic
        
        2. **Second Stage**
           ```python
           Y = β₀ + β₁X̂ + ε
           ```
           - Regress Y on fitted values X̂
           - β₁ is the IV estimate
           - Correct standard errors
        
        3. **Alternative Methods**
           - GMM: Efficient with heteroskedasticity
           - LIML: Better with weak instruments
           - CUE: Continuous updating estimator
        """)
    
    with tabs[2]:
        st.markdown("#### Diagnostic Tests")
        
        # Create visualization of weak instruments
        np.random.seed(42)
        n = 100
        
        # Generate data for different instrument strengths
        Z_strong = np.random.normal(0, 1, n)
        Z_weak = np.random.normal(0, 1, n)
        
        X_strong = Z_strong + np.random.normal(0, 0.3, n)
        X_weak = 0.1 * Z_weak + np.random.normal(0, 1, n)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Strong instrument
        ax1.scatter(Z_strong, X_strong, alpha=0.5)
        beta_strong = np.polyfit(Z_strong, X_strong, 1)
        z_range = np.linspace(Z_strong.min(), Z_strong.max(), 100)
        ax1.plot(z_range, beta_strong[0] * z_range + beta_strong[1], 'r-', alpha=0.7)
        ax1.set_title('Strong Instrument\nF-stat > 10')
        ax1.set_xlabel('Instrument (Z)')
        ax1.set_ylabel('Endogenous Variable (X)')
        
        # Weak instrument
        ax2.scatter(Z_weak, X_weak, alpha=0.5)
        beta_weak = np.polyfit(Z_weak, X_weak, 1)
        z_range = np.linspace(Z_weak.min(), Z_weak.max(), 100)
        ax2.plot(z_range, beta_weak[0] * z_range + beta_weak[1], 'r-', alpha=0.7)
        ax2.set_title('Weak Instrument\nF-stat < 10')
        ax2.set_xlabel('Instrument (Z)')
        ax2.set_ylabel('Endogenous Variable (X)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Key Diagnostic Tests:**
        
        1. **Instrument Strength**
           - First-stage F-statistic > 10
           - Stock-Yogo critical values
           - Cragg-Donald F-statistic
           - Kleibergen-Paap rk statistic
        
        2. **Overidentification**
           - Sargan-Hansen J-test
           - Only with multiple instruments
           - H₀: All instruments valid
           - Check p-value > 0.05
        
        3. **Endogeneity**
           - Durbin-Wu-Hausman test
           - H₀: OLS is consistent
           - Compare OLS and IV estimates
           - Test endogeneity assumption
        
        4. **Robustness Checks**
           - Different instrument sets
           - Alternative specifications
           - Placebo tests
           - Sensitivity analysis
        """)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Returns to Education**
    
    Consider estimating returns to education using quarter of birth as an instrument:
    
    **Step 1: First Stage**
    ```python
    Education = π₀ + π₁Quarter + π₂Controls + v
    
    First-stage results:
    - F-statistic = 23.4
    - π₁ = 0.34 (SE = 0.07)
    - R² = 0.15
    ```
    
    **Step 2: Second Stage**
    ```python
    log(Wage) = β₀ + β₁Education̂ + β₂Controls + ε
    
    IV Results:
    - β₁ = 0.132 (SE = 0.025)
    - OLS estimate = 0.074
    - Hausman test p-value = 0.02
    ```
    
    **Step 3: Interpretation**
    - IV estimate larger than OLS
    - Suggests downward bias in OLS
    - LATE interpretation: effect for compliers
    - 13.2% return per year of education
    
    **Step 4: Robustness**
    - Alternative instruments
    - Control for region and cohort
    - Placebo tests with pre-treatment outcomes
    - Sensitivity to specification
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Limited Dependent Variables")
    
    st.markdown("""
    Instrumental variables methods connect to limited dependent variables through:
    - Binary endogenous regressors
    - Sample selection models
    - Treatment effects
    - Control function approaches
    
    The next section will explore these limited dependent variable methods.
    """) 