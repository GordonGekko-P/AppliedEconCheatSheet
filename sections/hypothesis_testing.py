import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Hypothesis Testing section."""
    
    # Section header with blue diamond
    st.markdown('<div class="blue-diamond">◆</div><h2 class="section-header">Hypothesis Testing & Statistical Inference</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers the fundamental concepts and procedures of hypothesis testing in econometrics,
    including test statistics, p-values, and different types of statistical tests commonly used in empirical research.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "Null Hypothesis ($H_0$)",
            "Alternative Hypothesis ($H_1$)",
            "Test Statistic",
            "p-value",
            "Type I Error (α)",
            "Type II Error (β)",
            "Power (1-β)",
            "Critical Value"
        ],
        "Formula/Definition": [
            "Statement of no effect or relationship",
            "Statement of effect or relationship",
            "$t = \\frac{\\hat{\\theta} - \\theta_0}{SE(\\hat{\\theta})}$",
            "$P(|T| > |t_{obs}| | H_0)$",
            "$P(\\text{Reject } H_0 | H_0 \\text{ true})$",
            "$P(\\text{Fail to reject } H_0 | H_0 \\text{ false})$",
            "$P(\\text{Reject } H_0 | H_0 \\text{ false})$",
            "$t_{\\alpha/2, df}$ or $z_{\\alpha/2}$"
        ],
        "Example/Usage": [
            "$H_0: \\beta_1 = 0$",
            "$H_1: \\beta_1 \\neq 0$",
            "t-statistic, z-statistic, F-statistic",
            "Compare to significance level α",
            "Usually set at 5% or 1%",
            "Depends on effect size and sample size",
            "Increases with sample size",
            "Compare test statistic against"
        ],
        "Decision Rule": [
            "Reject if test statistic > critical value",
            "Or if p-value < α",
            "Two-tailed vs. one-tailed tests",
            "Reject $H_0$ if p < α",
            "Balance with Type II error",
            "Reduce with larger sample size",
            "Aim for power ≥ 0.8",
            "Depends on α and degrees of freedom"
        ]
    }
    
    # Convert to DataFrame and display as table
    df = pd.DataFrame(data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Common Statistical Tests
    st.markdown("### 📊 Common Statistical Tests")
    
    tests_data = {
        "Test": [
            "t-test (one sample)",
            "t-test (two sample)",
            "Paired t-test",
            "F-test",
            "Chi-square test",
            "Wald test",
            "Likelihood Ratio test",
            "Lagrange Multiplier test"
        ],
        "Usage": [
            "Compare sample mean to hypothesized value",
            "Compare means of two independent samples",
            "Compare means of paired observations",
            "Compare variances or multiple restrictions",
            "Test independence in contingency tables",
            "Test parameter restrictions",
            "Compare nested models",
            "Test parameter restrictions without full model"
        ],
        "Test Statistic": [
            "$t = \\frac{\\bar{X} - \\mu_0}{s/\\sqrt{n}}$",
            "$t = \\frac{\\bar{X}_1 - \\bar{X}_2}{\\sqrt{s_1^2/n_1 + s_2^2/n_2}}$",
            "$t = \\frac{\\bar{D}}{s_D/\\sqrt{n}}$",
            "$F = \\frac{RSS_R - RSS_U}{RSS_U/(n-k)}$",
            "$\\chi^2 = \\sum \\frac{(O-E)^2}{E}$",
            "$W = (R\\hat{\\beta})^\\prime[R\\hat{V}(\\hat{\\beta})R^\\prime]^{-1}(R\\hat{\\beta})$",
            "$LR = -2(\\ln L_R - \\ln L_U)$",
            "$LM = n R^2$"
        ],
        "Distribution": [
            "$t_{n-1}$ under $H_0$",
            "$t_{n_1+n_2-2}$ under $H_0$",
            "$t_{n-1}$ under $H_0$",
            "$F_{k,n-k}$ under $H_0$",
            "$\\chi^2_{(r-1)(c-1)}$ under $H_0$",
            "$\\chi^2_q$ under $H_0$",
            "$\\chi^2_q$ under $H_0$",
            "$\\chi^2_q$ under $H_0$"
        ]
    }
    
    # Convert to DataFrame and display as table
    df_tests = pd.DataFrame(tests_data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df_tests.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Interactive Hypothesis Testing
    st.markdown("### 🔄 Interactive Hypothesis Testing")
    
    # Create tabs for different test types
    tabs = st.tabs(["One Sample t-test", "Two Sample t-test", "F-test"])
    
    with tabs[0]:
        st.markdown("#### One Sample t-test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters
            sample_mean = st.number_input("Sample Mean", value=10.0, step=0.1)
            sample_std = st.number_input("Sample Standard Deviation", value=2.0, step=0.1, min_value=0.1)
            sample_size = st.number_input("Sample Size", value=30, step=1, min_value=2)
            null_value = st.number_input("Null Hypothesis Value (μ₀)", value=9.0, step=0.1)
            alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
            
            # Calculate test statistic
            t_stat = (sample_mean - null_value) / (sample_std / np.sqrt(sample_size))
            df = sample_size - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Critical values
            t_crit = stats.t.ppf(1 - alpha/2, df)
            
            st.markdown(f"""
            **Test Results:**
            - t-statistic: {t_stat:.4f}
            - Degrees of freedom: {df}
            - p-value: {p_value:.4f}
            - Critical value (±): {t_crit:.4f}
            
            **Decision:**
            {
            "Reject H₀" if p_value < alpha else "Fail to reject H₀"
            } at α = {alpha}
            """)
        
        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate t-distribution
            x = np.linspace(-4, 4, 1000)
            y = stats.t.pdf(x, df)
            
            # Plot t-distribution
            ax.plot(x, y, 'b-', lw=2, label='t-distribution')
            
            # Add critical regions
            x_crit = np.linspace(-4, -t_crit, 100)
            ax.fill_between(x_crit, stats.t.pdf(x_crit, df), color='red', alpha=0.3)
            x_crit = np.linspace(t_crit, 4, 100)
            ax.fill_between(x_crit, stats.t.pdf(x_crit, df), color='red', alpha=0.3)
            
            # Add observed t-statistic
            ax.axvline(x=t_stat, color='g', linestyle='--', label=f't-stat = {t_stat:.2f}')
            
            # Add labels and title
            ax.set_title('t-distribution with Critical Regions')
            ax.set_xlabel('t-value')
            ax.set_ylabel('Density')
            ax.legend()
            
            st.pyplot(fig)
    
    with tabs[1]:
        st.markdown("#### Two Sample t-test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Group 1 parameters
            st.markdown("**Group 1**")
            mean1 = st.number_input("Mean 1", value=10.0, step=0.1)
            std1 = st.number_input("Standard Deviation 1", value=2.0, step=0.1, min_value=0.1)
            n1 = st.number_input("Sample Size 1", value=30, step=1, min_value=2)
            
            # Group 2 parameters
            st.markdown("**Group 2**")
            mean2 = st.number_input("Mean 2", value=9.0, step=0.1)
            std2 = st.number_input("Standard Deviation 2", value=2.0, step=0.1, min_value=0.1)
            n2 = st.number_input("Sample Size 2", value=30, step=1, min_value=2)
            
            # Test parameters
            alpha_2 = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1, key="alpha_2")
            
            # Calculate pooled standard error
            se = np.sqrt(std1**2/n1 + std2**2/n2)
            
            # Calculate test statistic
            t_stat_2 = (mean1 - mean2) / se
            df_2 = n1 + n2 - 2
            p_value_2 = 2 * (1 - stats.t.cdf(abs(t_stat_2), df_2))
            
            # Critical values
            t_crit_2 = stats.t.ppf(1 - alpha_2/2, df_2)
            
            st.markdown(f"""
            **Test Results:**
            - t-statistic: {t_stat_2:.4f}
            - Degrees of freedom: {df_2}
            - p-value: {p_value_2:.4f}
            - Critical value (±): {t_crit_2:.4f}
            
            **Decision:**
            {
            "Reject H₀" if p_value_2 < alpha_2 else "Fail to reject H₀"
            } at α = {alpha_2}
            """)
        
        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate normal distributions for both groups
            x = np.linspace(min(mean1, mean2) - 3*max(std1, std2),
                          max(mean1, mean2) + 3*max(std1, std2), 1000)
            y1 = stats.norm.pdf(x, mean1, std1)
            y2 = stats.norm.pdf(x, mean2, std2)
            
            # Plot distributions
            ax.plot(x, y1, 'b-', lw=2, label=f'Group 1 (μ={mean1})')
            ax.plot(x, y2, 'r-', lw=2, label=f'Group 2 (μ={mean2})')
            
            # Add vertical lines for means
            ax.axvline(x=mean1, color='b', linestyle='--')
            ax.axvline(x=mean2, color='r', linestyle='--')
            
            # Add labels and title
            ax.set_title('Comparison of Group Distributions')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            
            st.pyplot(fig)
    
    with tabs[2]:
        st.markdown("#### F-test for Multiple Restrictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters
            r_squared = st.slider("R² of Unrestricted Model", 0.0, 1.0, 0.3, step=0.01)
            n_obs = st.number_input("Number of Observations", value=100, step=1, min_value=10)
            k_vars = st.number_input("Number of Variables (including constant)", value=5, step=1, min_value=2)
            q_restr = st.number_input("Number of Restrictions", value=2, step=1, min_value=1)
            alpha_f = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1, key="alpha_f")
            
            # Calculate F-statistic
            r_squared_r = r_squared * 0.8  # Restricted model R² (simulated)
            f_stat = ((r_squared - r_squared_r) / q_restr) / ((1 - r_squared) / (n_obs - k_vars))
            df1 = q_restr
            df2 = n_obs - k_vars
            p_value_f = 1 - stats.f.cdf(f_stat, df1, df2)
            
            # Critical value
            f_crit = stats.f.ppf(1 - alpha_f, df1, df2)
            
            st.markdown(f"""
            **Test Results:**
            - F-statistic: {f_stat:.4f}
            - Degrees of freedom: ({df1}, {df2})
            - p-value: {p_value_f:.4f}
            - Critical value: {f_crit:.4f}
            
            **Decision:**
            {
            "Reject H₀" if p_value_f < alpha_f else "Fail to reject H₀"
            } at α = {alpha_f}
            """)
        
        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate F-distribution
            x = np.linspace(0, max(f_stat * 2, f_crit * 2), 1000)
            y = stats.f.pdf(x, df1, df2)
            
            # Plot F-distribution
            ax.plot(x, y, 'b-', lw=2, label='F-distribution')
            
            # Add critical region
            x_crit = np.linspace(f_crit, max(x), 100)
            ax.fill_between(x_crit, stats.f.pdf(x_crit, df1, df2), color='red', alpha=0.3)
            
            # Add observed F-statistic
            ax.axvline(x=f_stat, color='g', linestyle='--', label=f'F-stat = {f_stat:.2f}')
            
            # Add labels and title
            ax.set_title('F-distribution with Critical Region')
            ax.set_xlabel('F-value')
            ax.set_ylabel('Density')
            ax.legend()
            
            st.pyplot(fig)
    
    # Power Analysis
    st.markdown("### 📈 Power Analysis")
    
    st.markdown("""
    Power analysis helps determine the sample size needed to detect an effect of a given size
    with a specified level of confidence. It involves four key components:
    
    1. **Effect Size (d)**: The magnitude of the difference you want to detect
    2. **Sample Size (n)**: The number of observations
    3. **Significance Level (α)**: The probability of Type I error
    4. **Power (1-β)**: The probability of correctly rejecting a false null hypothesis
    """)
    
    # Interactive power analysis
    st.markdown("#### Sample Size Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input parameters
        effect_size = st.slider("Effect Size (d)", 0.1, 1.0, 0.5, step=0.1)
        desired_power = st.slider("Desired Power (1-β)", 0.7, 0.95, 0.8, step=0.05)
        alpha_power = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1, key="alpha_power")
        
        # Calculate required sample size
        z_alpha = stats.norm.ppf(1 - alpha_power/2)
        z_beta = stats.norm.ppf(desired_power)
        n_required = np.ceil(2 * ((z_alpha + z_beta)/effect_size)**2)
        
        st.markdown(f"""
        **Required Sample Size:**
        
        To detect an effect size of {effect_size:.2f} with {desired_power*100:.0f}% power
        at α = {alpha_power}, you need at least **{int(n_required)}** observations.
        
        **Interpretation:**
        - Larger effect sizes require smaller samples
        - Higher power requires larger samples
        - Lower α requires larger samples
        """)
    
    with col2:
        # Power curve visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate sample sizes
        n_range = np.linspace(10, int(n_required*2), 100)
        
        # Calculate power for each sample size
        power_curve = []
        for n in n_range:
            ncp = effect_size * np.sqrt(n/2)  # Non-centrality parameter
            crit_val = stats.norm.ppf(1 - alpha_power/2)
            power = 1 - stats.norm.cdf(crit_val - ncp)
            power_curve.append(power)
        
        # Plot power curve
        ax.plot(n_range, power_curve, 'b-', lw=2)
        
        # Add reference lines
        ax.axhline(y=desired_power, color='r', linestyle='--', label=f'Desired Power = {desired_power:.2f}')
        ax.axvline(x=n_required, color='g', linestyle='--', label=f'Required n = {int(n_required)}')
        
        # Add labels and title
        ax.set_title('Power Curve')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Power (1-β)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Testing the Effect of a Job Training Program**
    
    A researcher wants to evaluate whether a job training program increases wages. They have:
    - Treatment group (n₁ = 200): Mean wage = $15.50/hour, SD = $3.00
    - Control group (n₂ = 200): Mean wage = $14.75/hour, SD = $2.80
    
    **Hypothesis Test:**
    - H₀: μ₁ - μ₂ = 0 (no effect)
    - H₁: μ₁ - μ₂ ≠ 0 (program has an effect)
    
    **Solution:**
    1. Calculate test statistic:
       - Pooled SE = √(3.00²/200 + 2.80²/200) = 0.29
       - t = (15.50 - 14.75)/0.29 = 2.59
    
    2. Degrees of freedom = 200 + 200 - 2 = 398
    
    3. For α = 0.05:
       - Critical value = ±1.966
       - p-value = 0.010
    
    4. Decision: Reject H₀ at 5% level
    
    5. Interpretation: The job training program is associated with a statistically significant
       increase in hourly wages of $0.75 (p = 0.010).
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Time Series Analysis")
    
    st.markdown("""
    The hypothesis testing concepts covered here extend to time series analysis through:
    - Tests for stationarity (unit root tests)
    - Tests for serial correlation
    - Tests for cointegration
    - Tests for Granger causality
    
    The next section will explore these time series-specific tests and their applications.
    """) 