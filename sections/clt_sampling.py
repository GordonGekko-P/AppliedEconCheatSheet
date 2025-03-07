import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Central Limit Theorem & Sampling Distributions section."""
    
    # Section header with blue diamond
    st.markdown('<div class="blue-diamond">◆</div><h2 class="section-header">Central Limit Theorem (CLT) & Sampling Distributions</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    The Central Limit Theorem is a cornerstone of statistical inference and econometrics.
    It provides the foundation for making inferences about population parameters from sample statistics.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "Sample Mean ($\\bar{Y}$)",
            "Variance of $\\bar{Y}$",
            "Standard Error of $\\bar{Y}$",
            "Normal Approximation via CLT",
            "Central Limit Theorem (CLT)",
            "Distribution of OLS Estimators",
            "Standard Error of Regression Coefficients"
        ],
        "Formula": [
            "$\\bar{Y} = \\frac{1}{n} \\sum Y_i$",
            "$\\text{Var}(\\bar{Y}) = \\frac{\\sigma^2}{n}$",
            "$SE(\\bar{Y}) = \\frac{\\sigma}{\\sqrt{n}}$",
            "$\\bar{Y} \\sim N(\\mu, \\frac{\\sigma^2}{n})$",
            "$\\frac{\\bar{Y} - \\mu}{\\sigma/\\sqrt{n}} \\sim N(0,1)$ as $n \\to \\infty$",
            "$\\hat{\\beta} \\sim N(\\beta, \\text{Var}(\\hat{\\beta}))$",
            "$SE(\\hat{\\beta}_j) = \\sqrt{\\frac{\\hat{\\sigma}^2}{\\sum(X_{ji}-\\bar{X}_j)^2(1-R_j^2)}}$"
        ],
        "Explanation": [
            "The mean of a sample drawn from a population.",
            "Shows that sample mean variance decreases as sample size increases.",
            "Measures how much $\\bar{Y}$ fluctuates from sample to sample.",
            "Sample means follow a normal distribution when $n$ is large.",
            "As sample size increases, the distribution of the standardized sample mean approaches standard normal distribution.",
            "OLS estimators are approximately normally distributed in large samples.",
            "Measures precision of regression coefficient estimates, accounting for correlations among predictors."
        ],
        "Next Step": [
            "Sampling Distribution of $\\bar{Y}$ ⬇️",
            "Standard Error of $\\bar{Y}$ ⬇️",
            "Normal Approximation via CLT ⬇️",
            "Probability Calculations for $\\bar{Y}$ ⬇️",
            "Confidence Intervals ⬇️",
            "Hypothesis Testing ⬇️",
            "Inference in Regression ⬇️"
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
    
    # Visual representation of concepts
    st.markdown("### 📊 Visual Representation of Key Concepts")
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Central Limit Theorem", "Sampling Distributions", "Sample Size Effect"])
    
    with tabs[0]:
        # CLT visualization
        st.markdown("#### Central Limit Theorem in Action")
        
        # Let user select sample size
        sample_size = st.slider("Select sample size (n)", 1, 100, 30)
        n_samples = 1000
        
        st.markdown(f"""
        This simulation draws {n_samples} samples of size n = {sample_size} from a non-normal distribution 
        (exponential distribution with λ = 1). The Central Limit Theorem predicts that the distribution 
        of sample means will become approximately normal as sample size increases.
        """)
        
        # Simulation
        np.random.seed(42)
        
        # Create an exponential distribution (non-normal)
        population = stats.expon(scale=1)
        population_mean = population.mean()
        population_std = population.std()
        
        # Generate samples and calculate sample means
        sample_means = np.array([
            np.mean(population.rvs(size=sample_size)) 
            for _ in range(n_samples)
        ])
        
        # Calculate observed standard error
        observed_se = np.std(sample_means, ddof=1)
        theoretical_se = population_std / np.sqrt(sample_size)
        
        # Create plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Parent distribution (exponential)
        x = np.linspace(0, 5, 1000)
        y = population.pdf(x)
        axs[0].plot(x, y, 'r-', linewidth=2)
        axs[0].fill_between(x, y, alpha=0.2, color='red')
        axs[0].set_title('Parent Distribution (Exponential)')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Density')
        
        # Plot 2: Distribution of sample means
        axs[1].hist(sample_means, bins=30, density=True, alpha=0.6, 
                   color='skyblue', label=f'Sample Means (n={sample_size})')
        
        # Overlay normal distribution with same mean and variance
        x_norm = np.linspace(min(sample_means), max(sample_means), 1000)
        y_norm = stats.norm.pdf(x_norm, population_mean, theoretical_se)
        axs[1].plot(x_norm, y_norm, 'r-', linewidth=2, 
                   label=f'Normal: N({population_mean:.2f}, {theoretical_se:.4f}²)')
        
        axs[1].axvline(x=population_mean, color='k', linestyle='--', 
                      label=f'Population Mean = {population_mean:.2f}')
        
        axs[1].set_title('Sampling Distribution of Sample Means')
        axs[1].set_xlabel('Sample Mean Value')
        axs[1].set_ylabel('Density')
        axs[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Comparison of theoretical vs. observed standard error
        st.markdown(f"""
        **Theoretical vs. Observed Standard Error:**
        - Theoretical SE = σ/√n = {theoretical_se:.4f}
        - Observed SE from simulation = {observed_se:.4f}
        
        As sample size increases, the sampling distribution becomes more normal and the standard error decreases.
        """)
        
    with tabs[1]:
        # Sampling distribution visualizations
        st.markdown("#### Sampling Distributions of Different Statistics")
        
        # Let user select distribution type
        dist_type = st.selectbox(
            "Select parent distribution type:",
            ["Normal", "Uniform", "Exponential", "Bimodal"]
        )
        
        sample_size_2 = st.slider("Select sample size", 1, 100, 30, key="sample_size_2")
        n_samples_2 = 1000
        
        # Generate parent distribution
        np.random.seed(42)
        x = np.linspace(-4, 4, 1000)
        
        if dist_type == "Normal":
            population = stats.norm(0, 1)
            parent_label = "Normal: N(0, 1)"
        elif dist_type == "Uniform":
            population = stats.uniform(-1.73, 3.46)  # Centered at 0 with variance 1
            parent_label = "Uniform: U(-1.73, 1.73)"
        elif dist_type == "Exponential":
            population = stats.expon(scale=1)
            parent_label = "Exponential: Exp(1)"
        else:  # Bimodal
            def bimodal(x):
                return 0.5 * stats.norm(-1, 0.5).pdf(x) + 0.5 * stats.norm(1, 0.5).pdf(x)
            
            y = np.array([bimodal(xi) for xi in x])
            parent_label = "Bimodal Mixture"
            
            # Create custom bimodal samples
            samples = np.concatenate([
                np.random.normal(-1, 0.5, size=(n_samples_2, sample_size_2 // 2)),
                np.random.normal(1, 0.5, size=(n_samples_2, sample_size_2 - sample_size_2 // 2))
            ], axis=1)
            
        # Generate samples
        if dist_type != "Bimodal":
            samples = population.rvs(size=(n_samples_2, sample_size_2))
        
        # Calculate statistics
        sample_means = np.mean(samples, axis=1)
        sample_medians = np.median(samples, axis=1)
        sample_stds = np.std(samples, axis=1, ddof=1)
        
        # Create plots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Parent distribution
        if dist_type != "Bimodal":
            y = population.pdf(x)
            axs[0, 0].plot(x, y, 'r-', linewidth=2)
            axs[0, 0].fill_between(x, y, alpha=0.2, color='red')
        else:
            x_bimodal = np.linspace(-3, 3, 1000)
            y_bimodal = [bimodal(xi) for xi in x_bimodal]
            axs[0, 0].plot(x_bimodal, y_bimodal, 'r-', linewidth=2)
            axs[0, 0].fill_between(x_bimodal, y_bimodal, alpha=0.2, color='red')
            
        axs[0, 0].set_title(f'Parent Distribution ({parent_label})')
        axs[0, 0].set_xlabel('Value')
        axs[0, 0].set_ylabel('Density')
        
        # Plot 2: Distribution of sample means
        axs[0, 1].hist(sample_means, bins=30, density=True, alpha=0.6, color='skyblue')
        axs[0, 1].set_title(f'Sampling Distribution of Sample Mean (n={sample_size_2})')
        axs[0, 1].set_xlabel('Sample Mean')
        axs[0, 1].set_ylabel('Density')
        
        # Add normal curve to mean plot
        mean_mean = np.mean(sample_means)
        mean_std = np.std(sample_means, ddof=1)
        x_norm = np.linspace(min(sample_means), max(sample_means), 1000)
        y_norm = stats.norm.pdf(x_norm, mean_mean, mean_std)
        axs[0, 1].plot(x_norm, y_norm, 'r-', linewidth=2)
        
        # Plot 3: Distribution of sample medians
        axs[1, 0].hist(sample_medians, bins=30, density=True, alpha=0.6, color='lightgreen')
        axs[1, 0].set_title(f'Sampling Distribution of Sample Median (n={sample_size_2})')
        axs[1, 0].set_xlabel('Sample Median')
        axs[1, 0].set_ylabel('Density')
        
        # Plot 4: Distribution of sample standard deviations
        axs[1, 1].hist(sample_stds, bins=30, density=True, alpha=0.6, color='salmon')
        axs[1, 1].set_title(f'Sampling Distribution of Sample Standard Deviation (n={sample_size_2})')
        axs[1, 1].set_xlabel('Sample Standard Deviation')
        axs[1, 1].set_ylabel('Density')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"""
        This visualization shows the sampling distributions of different statistics
        (mean, median, standard deviation) based on samples of size n = {sample_size_2}
        from a {dist_type.lower()} parent distribution.
        
        **Key Observations:**
        - The sampling distribution of the mean is approximately normal (CLT in action)
        - The sampling distribution of the median is more variable than the mean
        - The sampling distribution of the standard deviation is right-skewed
        
        As sample size increases, all these distributions become narrower, indicating more precise estimation.
        """)
        
    with tabs[2]:
        # Sample size effect visualization
        st.markdown("#### Effect of Sample Size on Sampling Distributions")
        
        # Create comparison of different sample sizes
        sample_sizes = [5, 30, 100]
        n_samples_3 = 1000
        
        # Generate samples from an exponential distribution
        np.random.seed(42)
        population = stats.expon(scale=1)
        
        # Prepare plot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = np.array([
                np.mean(population.rvs(size=n))
                for _ in range(n_samples_3)
            ])
            
            # Calculate theoretical standard error
            theoretical_se = population.std() / np.sqrt(n)
            
            # Plot histogram
            axs[i].hist(sample_means, bins=30, density=True, alpha=0.6, 
                       color='skyblue', label=f'Sample Means')
            
            # Overlay normal distribution
            x_norm = np.linspace(min(sample_means), max(sample_means), 1000)
            y_norm = stats.norm.pdf(x_norm, population.mean(), theoretical_se)
            axs[i].plot(x_norm, y_norm, 'r-', linewidth=2, 
                       label=f'N({population.mean():.2f}, {theoretical_se:.4f}²)')
            
            axs[i].axvline(x=population.mean(), color='k', linestyle='--')
            axs[i].set_title(f'n = {n}')
            axs[i].set_xlabel('Sample Mean')
            
            if i == 0:
                axs[i].set_ylabel('Density')
                
            # Add SE annotation
            axs[i].annotate(f'SE = {theoretical_se:.4f}', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        fig.suptitle('Effect of Sample Size on the Sampling Distribution of the Mean', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        st.pyplot(fig)
        
        st.markdown("""
        **Observations:**
        
        As sample size (n) increases:
        1. The sampling distribution becomes more normal (even for non-normal parent distributions)
        2. The standard error decreases (proportional to 1/√n)
        3. Sample means cluster more tightly around the true population mean
        
        This demonstrates why larger samples provide more precise estimates and why the Central Limit Theorem
        is so important for statistical inference.
        """)
    
    # The Central Limit Theorem in Econometrics
    st.markdown("### 🔑 The Central Limit Theorem in Econometrics")
    
    st.markdown("""
    The Central Limit Theorem (CLT) is crucial for econometric analysis for several reasons:
    
    1. **Enables Inference:** Even when the underlying population distribution is unknown or non-normal, 
    the CLT allows us to make inferences using t and z statistics.
    
    2. **Justifies OLS Asymptotics:** The CLT provides the foundation for the asymptotic properties 
    of OLS estimators, allowing us to perform hypothesis tests in large samples.
    
    3. **Confidence Intervals:** The CLT allows us to construct reliable confidence intervals for
    population parameters based on sample statistics.
    
    4. **Normality of Regression Coefficients:** The sampling distributions of regression coefficients 
    (β̂) are approximately normal in large samples, enabling valid hypothesis testing.
    """)
    
    # Confidence Intervals
    st.markdown("### 📏 Confidence Intervals")
    
    ci_data = {
        "Parameter": [
            "Population Mean (σ known)",
            "Population Mean (σ unknown)",
            "Regression Coefficient"
        ],
        "Confidence Interval": [
            "$\\bar{Y} \\pm z_{\\alpha/2} \\cdot \\frac{\\sigma}{\\sqrt{n}}$",
            "$\\bar{Y} \\pm t_{n-1, \\alpha/2} \\cdot \\frac{s}{\\sqrt{n}}$",
            "$\\hat{\\beta}_j \\pm t_{n-k-1, \\alpha/2} \\cdot SE(\\hat{\\beta}_j)$"
        ],
        "Interpretation": [
            "We are (1-α)×100% confident that the true population mean lies in this interval.",
            "Uses t-distribution to account for estimating σ with s.",
            "We are (1-α)×100% confident that the true regression coefficient lies in this interval."
        ]
    }
    
    # Convert to DataFrame and display as table
    df_ci = pd.DataFrame(ci_data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df_ci.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Application example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Estimating Average Income**
    
    A researcher sampled 100 households and found:
    - Sample mean income: $\\bar{Y} = \\$50,000$
    - Sample standard deviation: $s = \\$15,000$
    
    **Tasks:**
    1. Find the standard error of the sample mean
    2. Construct a 95% confidence interval for the population mean income
    3. Test if the population mean income is different from $\\$48,000$
    
    **Solutions:**
    
    1. $SE(\\bar{Y}) = \\frac{s}{\\sqrt{n}} = \\frac{15,000}{\\sqrt{100}} = \\$1,500$
    
    2. 95% CI: $\\bar{Y} \\pm t_{99, 0.025} \\cdot SE(\\bar{Y})$
       With n = 100, $t_{99, 0.025} ≈ 1.984$
       CI: $\\$50,000 \\pm 1.984 \\times \\$1,500 = [\\$47,024, \\$52,976]$
    
    3. $H_0: \\mu = \\$48,000$ vs. $H_1: \\mu \\neq \\$48,000$
       t-statistic: $t = \\frac{\\bar{Y} - \\mu_0}{SE(\\bar{Y})} = \\frac{50,000 - 48,000}{1,500} = 1.33$
       p-value = 0.186
       Since p-value > 0.05, fail to reject $H_0$
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Probability Calculations & Risk Analysis")
    
    st.markdown("""
    The Central Limit Theorem and sampling distributions provide the foundation for:
    - Converting raw observations to standardized z-scores
    - Calculating probabilities using normal distributions
    - Assessing and managing risk through statistical approaches
    - Insurance models that rely on the law of large numbers
    
    The next section explores these probability calculations and risk analysis in more detail.
    """) 