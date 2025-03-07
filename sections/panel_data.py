import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Panel Data Methods section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Panel Data Methods</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers key concepts and methods in panel data analysis, including fixed effects,
    random effects, and dynamic panel models. Panel data combines cross-sectional and time series
    dimensions, offering unique insights into economic relationships.
    """)
    
    # Create a DataFrame for the table of panel data models
    models = {
        "Model": [
            "Pooled OLS",
            "Fixed Effects (Within)",
            "Random Effects",
            "First Difference",
            "Between Effects",
            "Dynamic Panel",
            "System GMM",
            "Hausman-Taylor"
        ],
        "Equation": [
            "yᵢₜ = α + xᵢₜβ + εᵢₜ",
            "yᵢₜ = αᵢ + xᵢₜβ + εᵢₜ",
            "yᵢₜ = α + xᵢₜβ + (uᵢ + εᵢₜ)",
            "Δyᵢₜ = Δxᵢₜβ + Δεᵢₜ",
            "ȳᵢ = α + x̄ᵢβ + ūᵢ",
            "yᵢₜ = αᵢ + ρyᵢ,ₜ₋₁ + xᵢₜβ + εᵢₜ",
            "Uses lagged differences as instruments",
            "Combines FE and RE approaches"
        ],
        "Key Assumptions": [
            "No individual effects",
            "Fixed individual effects",
            "Random individual effects",
            "Strict exogeneity",
            "Between variation only",
            "Sequential exogeneity",
            "No serial correlation in εᵢₜ",
            "Some variables uncorrelated with effects"
        ],
        "When to Use": [
            "No unobserved heterogeneity",
            "Correlation with regressors",
            "No correlation with regressors",
            "Remove time-invariant effects",
            "Long-run relationships",
            "Persistence in dependent variable",
            "Endogenous regressors",
            "Time-invariant variables of interest"
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
    
    tabs = st.tabs(["Fixed vs Random Effects", "Model Selection", "Specification Tests"])
    
    with tabs[0]:
        st.markdown("#### Fixed Effects vs Random Effects")
        
        # Create comparison plot
        np.random.seed(42)
        n_individuals = 5
        n_time = 20
        
        # Generate individual effects
        individual_effects = np.random.normal(0, 2, n_individuals)
        
        # Create time variable
        t = np.linspace(0, 4, n_time)
        
        # Generate data
        X = np.random.normal(0, 1, (n_individuals, n_time))
        beta = 0.5
        epsilon = np.random.normal(0, 0.5, (n_individuals, n_time))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Fixed Effects
        for i in range(n_individuals):
            Y = individual_effects[i] + beta * X[i] + epsilon[i]
            ax1.plot(t, Y, '-', alpha=0.7, label=f'Individual {i+1}')
        
        ax1.set_title('Fixed Effects\nDifferent Intercepts')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Plot Random Effects
        random_effects = np.random.normal(0, 1, (n_individuals, n_time))
        for i in range(n_individuals):
            Y = beta * X[i] + random_effects[i]
            ax2.plot(t, Y, '-', alpha=0.7, label=f'Individual {i+1}')
        
        ax2.set_title('Random Effects\nRandom Deviations')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Fixed Effects (FE)**
        - Allows correlation between individual effects and regressors
        - Controls for time-invariant unobserved heterogeneity
        - Estimates individual-specific intercepts
        - Cannot estimate coefficients of time-invariant variables
        
        **Random Effects (RE)**
        - Assumes individual effects are uncorrelated with regressors
        - More efficient if assumption holds
        - Can estimate time-invariant variables
        - Uses GLS estimation
        """)
    
    with tabs[1]:
        st.markdown("#### Model Selection Process")
        
        # Create flowchart-like visualization
        st.markdown("""
        ```mermaid
        graph TD
            A[Start] --> B{Hausman Test}
            B -->|p < 0.05| C[Fixed Effects]
            B -->|p ≥ 0.05| D[Random Effects]
            C --> E{Time-invariant<br>variables?}
            E -->|Yes| F[Hausman-Taylor]
            E -->|No| G[Standard FE]
            D --> H{Heteroskedasticity?}
            H -->|Yes| I[Robust RE]
            H -->|No| J[Standard RE]
        ```
        
        **Decision Steps:**
        1. Run Hausman test to choose between FE and RE
        2. Check for time-invariant variables of interest
        3. Test for heteroskedasticity and autocorrelation
        4. Consider dynamic specification if needed
        5. Validate assumptions with specification tests
        """)
        
        # Model Selection Criteria
        selection_criteria = {
            "Test/Criterion": [
                "Hausman Test",
                "F-test for Fixed Effects",
                "Breusch-Pagan LM",
                "Wooldridge Test",
                "Modified Wald Test"
            ],
            "Null Hypothesis": [
                "RE is consistent and efficient",
                "No individual effects",
                "No random effects",
                "No first-order autocorrelation",
                "Homoskedasticity"
            ],
            "Decision Rule": [
                "Reject H₀ → Use FE",
                "Reject H₀ → Use FE over Pooled",
                "Reject H₀ → Use RE over Pooled",
                "Reject H₀ → Use robust SEs",
                "Reject H₀ → Use robust SEs"
            ]
        }
        
        # Display selection criteria table
        df_criteria = pd.DataFrame(selection_criteria)
        st.markdown(
            df_criteria.style.hide(axis="index")
            .to_html()
            .replace('<table', '<table class="concept-table"')
            .replace('<td>', '<td style="text-align: left; padding: 8px;">')
            .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
            unsafe_allow_html=True
        )
    
    with tabs[2]:
        st.markdown("#### Specification Tests")
        
        # Create visualization of residual patterns
        np.random.seed(42)
        n = 100
        
        # Generate example residuals
        homoskedastic = np.random.normal(0, 1, n)
        heteroskedastic = np.random.normal(0, np.linspace(0.5, 2, n), n)
        autocorrelated = np.cumsum(np.random.normal(0, 0.1, n))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot patterns
        ax1.scatter(range(n), homoskedastic, alpha=0.5)
        ax1.set_title('Homoskedastic\n(Desired)')
        ax1.set_xlabel('Observation')
        ax1.set_ylabel('Residual')
        
        ax2.scatter(range(n), heteroskedastic, alpha=0.5)
        ax2.set_title('Heteroskedastic\n(Problem)')
        ax2.set_xlabel('Observation')
        ax2.set_ylabel('Residual')
        
        ax3.scatter(range(n), autocorrelated, alpha=0.5)
        ax3.set_title('Autocorrelated\n(Problem)')
        ax3.set_xlabel('Observation')
        ax3.set_ylabel('Residual')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Common Specification Tests:**
        
        1. **Residual Analysis**
           - Plot residuals vs. fitted values
           - Test for normality
           - Check for patterns and outliers
        
        2. **Serial Correlation**
           - Wooldridge test for autocorrelation
           - Durbin-Watson test
           - ACF/PACF plots
        
        3. **Heteroskedasticity**
           - Modified Wald test
           - Breusch-Pagan test
           - White test
        
        4. **Cross-sectional Dependence**
           - Pesaran CD test
           - Frees test
           - Friedman test
        """)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Analyzing Firm Investment**
    
    Consider panel data on firm investment with variables:
    - Investment/Capital ratio (I/K)
    - Tobin's Q
    - Cash Flow/Capital ratio (CF/K)
    - Firm and Year fixed effects
    
    **Step 1: Model Selection**
    ```python
    # Hausman test results
    chi2(3) = 45.32
    p-value = 0.000
    → Choose Fixed Effects
    ```
    
    **Step 2: Fixed Effects Estimation**
    ```python
    (I/K)ᵢₜ = 0.15 + 0.08Qᵢₜ + 0.25(CF/K)ᵢₜ + αᵢ + γₜ + εᵢₜ
    
    R² within = 0.23
    F-test = 45.6 (p = 0.000)
    ```
    
    **Step 3: Diagnostic Tests**
    - Modified Wald test: χ²(234) = 12453.5 (p = 0.000)
    - Wooldridge test: F(1,233) = 15.4 (p = 0.000)
    → Use cluster-robust standard errors
    
    **Step 4: Final Specification**
    - Use two-way fixed effects with clustered SEs
    - Control for firm size and leverage
    - Include industry-year interactions
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Instrumental Variables")
    
    st.markdown("""
    Panel data methods connect to instrumental variables through:
    - Dynamic panel GMM estimation
    - Hausman-Taylor instrumental variables
    - Panel IV for endogenous regressors
    - System GMM for persistent series
    
    The next section will explore these instrumental variables methods.
    """) 