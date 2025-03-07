import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Probability Calculations & Risk Analysis section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Probability Calculations & Risk Analysis</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers key probability calculations and risk analysis concepts that are essential for 
    understanding economic decision-making under uncertainty and analyzing financial risks.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "Z-Score (Standardization)",
            "P(Y > threshold) Calculation",
            "Variance of Total Risk T (Insurance Pool)",
            "Variance of Average Risk $\\bar{Y}$",
            "Value at Risk (VaR)",
            "Conditional Value at Risk (CVaR)",
            "Hypothesis Testing (t-test)",
            "Type I and Type II Errors"
        ],
        "Formula": [
            "$Z = \\frac{X-\\mu}{\\sigma}$",
            "$P(Y > x) = 1 - P(Z < z)$",
            "$\\sigma_T^2 = n\\sigma_Y^2$",
            "$\\sigma_{\\bar{Y}}^2 = \\frac{\\sigma_Y^2}{n}$",
            "$P(X < VaR_\\alpha) = \\alpha$",
            "$CVaR_\\alpha = E[X | X < VaR_\\alpha]$",
            "$t = \\frac{\\bar{X} - \\mu_0}{s/\\sqrt{n}}$",
            "Type I: Reject $H_0$ when true<br>Type II: Accept $H_0$ when false"
        ],
        "Explanation": [
            "Converts any normal variable into standard normal form.",
            "Finds probability that a random variable exceeds a threshold.",
            "Total variance scales with number of policies.",
            "Larger sample sizes reduce individual risk.",
            "Maximum loss at a given confidence level.",
            "Expected loss given that loss exceeds VaR.",
            "Tests if sample mean differs from hypothesized value.",
            "Statistical decision errors in hypothesis testing."
        ],
        "Next Step": [
            "Find Probabilities Using Z-Tables ⬇️",
            "Risk Pooling in Insurance ⬇️",
            "Pooling Reduces Risk ⬇️",
            "Probability of Large Claims ($P(\\bar{Y} > 2000)$) ⬇️",
            "Risk Management Applications ⬇️",
            "Portfolio Optimization ⬇️",
            "Statistical Inference ⬇️",
            "Power Analysis ⬇️"
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
    
    # Z-Score and Probability Calculations
    st.markdown("### 📊 Z-Score and Probability Calculations")
    
    st.markdown("""
    Z-scores standardize random variables, making it possible to calculate probabilities using the standard normal distribution.
    
    The z-score tells us how many standard deviations an observation is from the mean:
    
    $$Z = \\frac{X - \\mu}{\\sigma}$$
    
    For a normal distribution, once we convert to z-scores, we can use standard normal tables or functions to find probabilities.
    """)
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Z-Score Interactive", "Risk Pooling", "Hypothesis Testing"])
    
    with tabs[0]:
        # Z-Score interactive visualization
        st.markdown("#### Z-Score and Probability Calculation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Let users input parameters
            mean = st.number_input("Mean (μ)", value=100.0, step=5.0)
            std_dev = st.number_input("Standard Deviation (σ)", value=15.0, step=1.0, min_value=0.1)
            threshold = st.number_input("Threshold (x)", value=120.0, step=5.0)
            
            # Calculate z-score
            z_score = (threshold - mean) / std_dev
            
            # Calculate probabilities
            p_less = stats.norm.cdf(z_score)
            p_greater = 1 - p_less
            
            st.markdown(f"""
            **Calculations:**
            - Z-score: $Z = \\frac{{{threshold} - {mean}}}{{{std_dev}}} = {z_score:.4f}$
            - P(X < {threshold}) = {p_less:.4f} ({p_less*100:.2f}%)
            - P(X > {threshold}) = {p_greater:.4f} ({p_greater*100:.2f}%)
            """)
            
            # Common probability questions
            st.markdown("**Common Probability Calculations:**")
            
            # P(within 1 sigma)
            p_within_1 = stats.norm.cdf(1) - stats.norm.cdf(-1)
            # P(within 2 sigma)
            p_within_2 = stats.norm.cdf(2) - stats.norm.cdf(-2)
            # P(within 3 sigma)
            p_within_3 = stats.norm.cdf(3) - stats.norm.cdf(-3)
            
            st.markdown(f"""
            - P(μ-σ < X < μ+σ) = {p_within_1:.4f} ({p_within_1*100:.2f}%)
            - P(μ-2σ < X < μ+2σ) = {p_within_2:.4f} ({p_within_2*100:.2f}%)
            - P(μ-3σ < X < μ+3σ) = {p_within_3:.4f} ({p_within_3*100:.2f}%)
            """)
        
        with col2:
            # Plot normal distribution with shaded areas
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate x values for plotting
            x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
            # Calculate normal PDF
            y = stats.norm.pdf(x, mean, std_dev)
            
            # Plot PDF
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Shade areas for P(X < threshold) and P(X > threshold)
            if threshold <= max(x):
                # Shade P(X < threshold)
                x_less = np.linspace(min(x), threshold, 500)
                y_less = stats.norm.pdf(x_less, mean, std_dev)
                ax.fill_between(x_less, y_less, color='skyblue', alpha=0.4, label=f'P(X < {threshold}) = {p_less:.4f}')
                
                # Shade P(X > threshold)
                x_greater = np.linspace(threshold, max(x), 500)
                y_greater = stats.norm.pdf(x_greater, mean, std_dev)
                ax.fill_between(x_greater, y_greater, color='salmon', alpha=0.4, label=f'P(X > {threshold}) = {p_greater:.4f}')
            
            # Add vertical lines for mean and threshold
            ax.axvline(x=mean, color='k', linestyle='--', label=f'Mean = {mean}')
            ax.axvline(x=threshold, color='r', linestyle='-', label=f'Threshold = {threshold}')
            
            # Add annotations for mean and standard deviations
            ax.annotate(f'μ = {mean}', xy=(mean, 0), xytext=(mean, -0.001), 
                        ha='center', fontsize=10)
            
            ax.annotate(f'μ+σ = {mean+std_dev:.1f}', xy=(mean+std_dev, 0), xytext=(mean+std_dev, -0.001), 
                        ha='center', fontsize=8)
            ax.annotate(f'μ-σ = {mean-std_dev:.1f}', xy=(mean-std_dev, 0), xytext=(mean-std_dev, -0.001), 
                        ha='center', fontsize=8)
            
            # Format plot
            ax.set_xlabel('X')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Normal Distribution N({mean}, {std_dev}²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.markdown(f"""
            **Z-score Interpretation:**
            - Z = {z_score:.2f} means the threshold is {abs(z_score):.2f} standard deviations 
              {'above' if z_score > 0 else 'below'} the mean.
            - {'This is an unusually high value.' if z_score > 2 else 'This is within the normal range.' if abs(z_score) <= 2 else 'This is an unusually low value.'}
            """)
        
    with tabs[1]:
        # Risk pooling visualization
        st.markdown("#### Risk Pooling in Insurance")
        
        st.markdown("""
        Risk pooling is a fundamental concept in insurance. By pooling many independent risks together,
        the variance of the average claim decreases, making total losses more predictable.
        
        This simulation demonstrates how increasing the number of policies in an insurance pool affects 
        the distribution of average claims.
        """)
        
        # Let user select parameters
        col1, col2 = st.columns(2)
        
        with col1:
            claim_mean = st.number_input("Mean Claim Amount (μ)", value=1000.0, step=100.0, min_value=1.0)
            claim_std = st.number_input("Std Dev of Claims (σ)", value=500.0, step=50.0, min_value=1.0)
            
        with col2:
            n_policies_1 = st.slider("Small Pool Size", 1, 50, 10)
            n_policies_2 = st.slider("Large Pool Size", 51, 1000, 100)
        
        # Number of simulations
        n_simulations = 10000
        
        # Generate claim data
        np.random.seed(42)
        
        # Generate individual claims
        individual_claims = np.random.normal(claim_mean, claim_std, n_simulations)
        
        # Generate average claims for different pool sizes
        small_pool_avg = np.array([
            np.mean(np.random.normal(claim_mean, claim_std, n_policies_1))
            for _ in range(n_simulations)
        ])
        
        large_pool_avg = np.array([
            np.mean(np.random.normal(claim_mean, claim_std, n_policies_2))
            for _ in range(n_simulations)
        ])
        
        # Calculate theoretical standard errors
        individual_se = claim_std
        small_pool_se = claim_std / np.sqrt(n_policies_1)
        large_pool_se = claim_std / np.sqrt(n_policies_2)
        
        # Calculate observed standard deviations
        individual_sd_obs = np.std(individual_claims, ddof=1)
        small_pool_sd_obs = np.std(small_pool_avg, ddof=1)
        large_pool_sd_obs = np.std(large_pool_avg, ddof=1)
        
        # Create plots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot histograms
        axs[0].hist(individual_claims, bins=30, alpha=0.6, density=True, color='salmon')
        axs[0].set_title(f'Individual Claims\nσ = {individual_sd_obs:.2f}')
        axs[0].set_xlabel('Claim Amount')
        axs[0].set_ylabel('Density')
        
        axs[1].hist(small_pool_avg, bins=30, alpha=0.6, density=True, color='skyblue')
        axs[1].set_title(f'Average Claim in Small Pool\n(n = {n_policies_1}, σ = {small_pool_sd_obs:.2f})')
        axs[1].set_xlabel('Average Claim Amount')
        
        axs[2].hist(large_pool_avg, bins=30, alpha=0.6, density=True, color='lightgreen')
        axs[2].set_title(f'Average Claim in Large Pool\n(n = {n_policies_2}, σ = {large_pool_sd_obs:.2f})')
        axs[2].set_xlabel('Average Claim Amount')
        
        # Add vertical lines for means
        for i, data in enumerate([individual_claims, small_pool_avg, large_pool_avg]):
            axs[i].axvline(x=np.mean(data), color='r', linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calculate probability of average claim exceeding threshold
        threshold = claim_mean * 1.2  # 20% above mean
        
        p_individual = np.mean(individual_claims > threshold)
        p_small_pool = np.mean(small_pool_avg > threshold)
        p_large_pool = np.mean(large_pool_avg > threshold)
        
        # Theoretical probabilities using normal approximation
        z_individual = (threshold - claim_mean) / individual_se
        z_small = (threshold - claim_mean) / small_pool_se
        z_large = (threshold - claim_mean) / large_pool_se
        
        p_individual_theory = 1 - stats.norm.cdf(z_individual)
        p_small_theory = 1 - stats.norm.cdf(z_small)
        p_large_theory = 1 - stats.norm.cdf(z_large)
        
        st.markdown(f"""
        **Risk Reduction via Pooling:**
        
        1. **Standard Error Comparison:**
           - Individual Claim: σ = {individual_se:.2f}
           - Small Pool (n={n_policies_1}): σ/√n = {small_pool_se:.2f}
           - Large Pool (n={n_policies_2}): σ/√n = {large_pool_se:.2f}
        
        2. **Probability of Exceeding {threshold:.2f} (20% above mean):**
           - Individual Claim: {p_individual:.4f} ({p_individual*100:.2f}%)
           - Small Pool (n={n_policies_1}): {p_small_pool:.4f} ({p_small_pool*100:.2f}%)
           - Large Pool (n={n_policies_2}): {p_large_pool:.4f} ({p_large_pool*100:.2f}%)
        
        This demonstrates the **Law of Large Numbers** in action. As the pool size increases:
        - The distribution of the average claim becomes more normal
        - The standard deviation decreases proportionally to 1/√n
        - The probability of extreme average outcomes becomes very small
        """)
        
    with tabs[2]:
        # Hypothesis testing visualization
        st.markdown("#### Hypothesis Testing and Statistical Errors")
        
        st.markdown("""
        Hypothesis testing involves making statistical decisions based on sample data.
        Two types of errors can occur:
        
        - **Type I Error (α)**: Rejecting the null hypothesis when it's actually true
        - **Type II Error (β)**: Failing to reject the null hypothesis when it's actually false
        
        This visualization explores the relationship between these errors in a z-test for a population mean.
        """)
        
        # Let user select parameters
        col1, col2 = st.columns(2)
        
        with col1:
            true_mean = st.number_input("True Population Mean (μ)", value=100.0, step=1.0)
            null_mean = st.number_input("Null Hypothesis Value (μ₀)", value=95.0, step=1.0)
            population_sd = st.number_input("Population Standard Deviation (σ)", value=15.0, step=1.0, min_value=0.1)
            
        with col2:
            sample_size = st.slider("Sample Size (n)", 5, 200, 30)
            alpha_level = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, step=0.01)
            
        # Calculate standard error
        se = population_sd / np.sqrt(sample_size)
        
        # Calculate critical value for alpha (two-sided test)
        critical_value = stats.norm.ppf(1 - alpha_level/2)
        
        # Calculate critical points in the original scale
        critical_low = null_mean - critical_value * se
        critical_high = null_mean + critical_value * se
        
        # Calculate Type II error (beta)
        # For the case where true mean > null mean
        if true_mean > null_mean:
            beta = stats.norm.cdf((critical_high - true_mean) / se)
        # For the case where true mean < null mean
        elif true_mean < null_mean:
            beta = 1 - stats.norm.cdf((critical_low - true_mean) / se)
        # For the case where true mean = null mean
        else:
            beta = 1 - alpha_level
        
        # Calculate power
        power = 1 - beta
        
        # Generate x values for plotting
        x_range = 6 * se  # Plot 6 standard errors in each direction
        x_null = np.linspace(null_mean - x_range, null_mean + x_range, 1000)
        x_alt = np.linspace(true_mean - x_range, true_mean + x_range, 1000)
        
        # Calculate PDF values
        y_null = stats.norm.pdf(x_null, null_mean, se)
        y_alt = stats.norm.pdf(x_alt, true_mean, se)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot PDFs
        ax.plot(x_null, y_null, 'b-', linewidth=2, label=f'Null: N({null_mean}, {se:.2f}²)')
        ax.plot(x_alt, y_alt, 'r-', linewidth=2, label=f'Alternative: N({true_mean}, {se:.2f}²)')
        
        # Shade rejection regions (Type I error)
        x_reject_low = np.linspace(null_mean - x_range, critical_low, 100)
        y_reject_low = stats.norm.pdf(x_reject_low, null_mean, se)
        ax.fill_between(x_reject_low, y_reject_low, color='blue', alpha=0.3)
        
        x_reject_high = np.linspace(critical_high, null_mean + x_range, 100)
        y_reject_high = stats.norm.pdf(x_reject_high, null_mean, se)
        ax.fill_between(x_reject_high, y_reject_high, color='blue', alpha=0.3)
        
        # Shade Type II error region (if applicable)
        if true_mean > null_mean:
            x_type2 = np.linspace(true_mean - x_range, critical_high, 100)
            y_type2 = stats.norm.pdf(x_type2, true_mean, se)
            ax.fill_between(x_type2, y_type2, color='red', alpha=0.3)
        elif true_mean < null_mean:
            x_type2 = np.linspace(critical_low, true_mean + x_range, 100)
            y_type2 = stats.norm.pdf(x_type2, true_mean, se)
            ax.fill_between(x_type2, y_type2, color='red', alpha=0.3)
        
        # Add vertical lines for critical values
        ax.axvline(x=critical_low, color='k', linestyle='--')
        ax.axvline(x=critical_high, color='k', linestyle='--')
        
        # Annotate critical values
        ax.annotate(f'Critical\nValue\n({critical_low:.2f})', xy=(critical_low, 0.01), xytext=(critical_low-2*se, 0.01),
                   ha='center', va='bottom', fontsize=8)
        ax.annotate(f'Critical\nValue\n({critical_high:.2f})', xy=(critical_high, 0.01), xytext=(critical_high+2*se, 0.01),
                   ha='center', va='bottom', fontsize=8)
        
        # Add legend and labels
        ax.legend()
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Probability Density')
        ax.set_title('Hypothesis Testing: Type I and Type II Errors')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown(f"""
        **Hypothesis Test Analysis:**
        
        Testing $H_0: \\mu = {null_mean}$ vs $H_1: \\mu \\neq {null_mean}$
        
        - **Standard Error (SE):** $\\frac{{\\sigma}}{{\\sqrt{{n}}}} = \\frac{{{population_sd}}}{{\\sqrt{{{sample_size}}}}} = {se:.4f}$
        
        - **Critical Values:**
          - For α = {alpha_level}, z-critical = ±{critical_value:.4f}
          - Reject $H_0$ if sample mean < {critical_low:.4f} or > {critical_high:.4f}
        
        - **Error Analysis:**
          - Type I Error (α): {alpha_level:.4f} (probability of rejecting $H_0$ when it's true)
          - Type II Error (β): {beta:.4f} (probability of not rejecting $H_0$ when $\\mu = {true_mean}$)
          - Power (1-β): {power:.4f} (probability of correctly rejecting $H_0$ when $\\mu = {true_mean}$)
        
        - **Improving the Test:**
          - Increasing sample size would decrease both types of errors
          - Increasing α would decrease Type II error but increase Type I error
        """)
    
    # Value at Risk (VaR) Section
    st.markdown("### 📉 Value at Risk (VaR) and Risk Management")
    
    st.markdown("""
    Value at Risk (VaR) is a key risk measure in financial economics that estimates the maximum potential loss 
    over a specific time period at a given confidence level.
    
    **Definition:** $VaR_\\alpha$ is the value such that $P(X < VaR_\\alpha) = \\alpha$, where X represents losses.
    
    For example, a one-day 95% VaR of $1 million means there is a 5% chance that losses will exceed $1 million over a one-day period.
    """)
    
    # Simple VaR calculator
    st.markdown("#### VaR Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=1000000.0, step=100000.0, min_value=1.0)
        annual_return = st.number_input("Expected Annual Return (%)", value=8.0, step=0.5) / 100
        annual_volatility = st.number_input("Annual Volatility (%)", value=15.0, step=0.5, min_value=0.1) / 100
        
    with col2:
        time_horizon = st.slider("Time Horizon (days)", 1, 252, 10)
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        
    # Convert annual parameters to the time horizon
    t_years = time_horizon / 252  # Assuming 252 trading days in a year
    expected_return = portfolio_value * (1 + annual_return) ** t_years
    scaled_volatility = annual_volatility * np.sqrt(t_years)
    
    # Calculate VaR
    z_score = stats.norm.ppf(1 - confidence_level)
    absolute_var = -portfolio_value * (expected_return / portfolio_value - 1 + z_score * scaled_volatility)
    relative_var = absolute_var / portfolio_value * 100
    
    # Calculate Conditional VaR (Expected Shortfall)
    z_cvar = stats.norm.pdf(z_score) / (1 - confidence_level)
    absolute_cvar = -portfolio_value * (expected_return / portfolio_value - 1 + z_cvar * scaled_volatility)
    relative_cvar = absolute_cvar / portfolio_value * 100
    
    # Create VaR visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate return data
    x = np.linspace(-0.3, 0.3, 1000)  # Return range
    mu = expected_return / portfolio_value - 1  # Expected return as percentage
    sigma = scaled_volatility  # Volatility for the time period
    
    # Calculate PDF
    y = stats.norm.pdf(x, mu, sigma)
    
    # Plot PDF
    ax.plot(x, y, 'b-', linewidth=2)
    
    # Find the VaR point
    var_return = -absolute_var / portfolio_value
    
    # Shade VaR region
    x_var = np.linspace(min(x), var_return, 500)
    y_var = stats.norm.pdf(x_var, mu, sigma)
    ax.fill_between(x_var, y_var, color='red', alpha=0.4, label=f'VaR Region ({confidence_level*100:.0f}%)')
    
    # Add vertical lines
    ax.axvline(x=mu, color='k', linestyle='--', label=f'Expected Return: {mu*100:.2f}%')
    ax.axvline(x=var_return, color='r', linestyle='-', label=f'VaR: {var_return*100:.2f}%')
    
    # Add annotations
    ax.annotate(f'VaR = ${absolute_var:,.2f}', xy=(var_return, 0), xytext=(var_return, 1),
               arrowprops=dict(facecolor='black', shrink=0.05),
               ha='center', va='bottom')
    
    # Format plot
    ax.set_xlabel('Return')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Value at Risk ({confidence_level*100:.0f}% Confidence, {time_horizon}-day Horizon)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Display results
    st.markdown(f"""
    **Value at Risk (VaR) Results:**
    
    For a ${portfolio_value:,.2f} portfolio over a {time_horizon}-day horizon at {confidence_level*100:.0f}% confidence:
    
    - **Absolute VaR:** ${absolute_var:,.2f}
    - **Relative VaR:** {relative_var:.2f}% of portfolio value
    - **Conditional VaR (Expected Shortfall):** ${absolute_cvar:,.2f} ({relative_cvar:.2f}% of portfolio)
    
    **Interpretation:**
    - There is a {(1-confidence_level)*100:.0f}% chance of losing more than ${absolute_var:,.2f} over the next {time_horizon} days.
    - If losses exceed VaR, the expected loss is ${absolute_cvar:,.2f} (Conditional VaR).
    """)
    
    # Application example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Insurance Risk Pool Analysis**
    
    An insurance company has 10,000 policies. Each policy has:
    - Expected annual claim amount: μ = $2,000
    - Standard deviation of claims: σ = $5,000
    
    **Questions:**
    1. What is the expected total annual claims for the entire pool?
    2. What is the standard deviation of total claims?
    3. What is the probability that average claim per policy exceeds $2,200?
    4. How much reserve should the company keep to cover claims with 99% confidence?
    
    **Solutions:**
    
    1. Expected total claims = n × μ = 10,000 × $2,000 = $20 million
    
    2. Standard deviation of total claims = σ_T = σ × √n = $5,000 × √10,000 = $500,000
    
    3. For average claim per policy:
       - Standard error of average = σ/√n = $5,000/√10,000 = $50
       - Z-score for $2,200 = (2,200 - 2,000)/50 = 4
       - P(avg > $2,200) = P(Z > 4) = 0.00003 (essentially zero)
    
    4. 99% confidence reserve:
       - Z-score for 99% confidence = 2.326
       - Reserve = Expected + Z × σ_T
       - Reserve = $20,000,000 + 2.326 × $500,000 = $21,163,000
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Hypothesis Testing & Confidence Intervals")
    
    st.markdown("""
    The probability concepts covered in this section provide the foundation for:
    - Formal hypothesis testing procedures for making statistical inferences
    - Construction and interpretation of confidence intervals
    - Testing specific hypotheses about population parameters
    - Understanding the tradeoffs between Type I and Type II errors
    
    The next sections will explore these topics in more detail, focusing on their application 
    in econometric analysis.
    """) 