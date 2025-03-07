import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm

def display_section():
    """Display the Regression Analysis section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Regression Analysis: Measuring Relationships Between Variables</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Regression analysis is a fundamental tool in econometrics for studying relationships between variables.
    This section covers the key concepts, formulas, and interpretations of regression models.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "OLS Regression Model",
            "Estimate $\\beta_1$ (Slope Coefficient)",
            "Standard Error of $\\hat{\\beta}_1$",
            "t-Statistic",
            "p-Value",
            "Confidence Interval for $\\beta_1$",
            "$R^2$ (Goodness of Fit)",
            "Standard Error of Regression (SER)",
            "Adjusted $R^2$",
            "Multiple Regression"
        ],
        "Formula": [
            "$Y = \\beta_0 + \\beta_1 X + u$",
            "$\\hat{\\beta}_1 = \\frac{\\sum(X_i-\\bar{X})(Y_i-\\bar{Y})}{\\sum(X_i-\\bar{X})^2}$",
            "$SE(\\hat{\\beta}_1) = \\frac{\\sigma_u}{\\sqrt{\\sum(X_i-\\bar{X})^2}}$",
            "$t = \\frac{\\hat{\\beta}_1-\\beta_0}{SE(\\hat{\\beta}_1)}$",
            "p = 2P(T > |t|)",
            "$\\hat{\\beta}_1 \\pm t_{\\alpha/2}SE(\\hat{\\beta}_1)$",
            "$R^2 = 1 - \\frac{SSR}{SST} = \\frac{ESS}{TSS}$",
            "$SER = \\sqrt{\\frac{\\sum u_i^2}{n-k-1}}$",
            "$\\bar{R}^2 = 1 - \\frac{n-1}{n-k-1}(1-R^2)$",
            "$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_k X_k + u$"
        ],
        "Explanation": [
            "Explains $Y$ (dependent variable) using $X$ (independent variable) with error term $u$.",
            "Measures how much $Y$ changes when $X$ changes by 1 unit.",
            "Measures the precision of $\\hat{\\beta}_1$; smaller SE means more precise estimate.",
            "Tests if $\\beta_1$ is significantly different from 0.",
            "Probability of observing the sample result if the null hypothesis ($\\beta_1=0$) is true.",
            "If CI excludes 0, reject $H_0$; if CI includes 0, fail to reject $H_0$.",
            "Measures how much variation in $Y$ is explained by $X$ (ranges from 0 to 1).",
            "Measures spread of residuals (smaller means better model fit).",
            "Adjusted $R^2$ penalizes for additional variables; useful for comparing models.",
            "Extends simple regression to include multiple independent variables."
        ],
        "Next Step": [
            "Estimate $\\beta_1$ using OLS ⬇️",
            "Standard Error of $\\hat{\\beta}_1$ ⬇️",
            "t-Test for $H_0: \\beta_1 = 0$ ⬇️",
            "p-Value ⬇️",
            "Interpretation ⬇️",
            "Model Fit ($R^2$ & SER) ⬇️",
            "Standard Error of Regression (SER) ⬇️",
            "Comparing Models (Adjusted $R^2$) ⬇️",
            "Multiple Regression ⬇️",
            "OLS Assumptions ⬇️"
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
    
    # OLS Assumptions
    st.markdown("### 📋 OLS Assumptions (CLRM)")
    
    assumptions_data = {
        "Assumption": [
            "1. Linear in Parameters",
            "2. Random Sampling",
            "3. No Perfect Multicollinearity",
            "4. Zero Conditional Mean",
            "5. Homoskedasticity",
            "6. Normality of Errors"
        ],
        "Mathematical Form": [
            "$Y = \\beta_0 + \\beta_1 X_1 + ... + \\beta_k X_k + u$",
            "$(X_i, Y_i)$ are randomly sampled",
            "No exact linear relationships among independent variables",
            "$E[u|X] = 0$",
            "$Var(u|X) = \\sigma^2$ (constant)",
            "$u \\sim N(0, \\sigma^2)$"
        ],
        "Violation Consequence": [
            "Model is misspecified, biased estimates",
            "Sample not representative of population",
            "Cannot estimate unique coefficients",
            "Biased and inconsistent estimates",
            "Standard errors are incorrect, inefficient estimation",
            "t and F tests are invalid in small samples"
        ],
        "Diagnostic/Solution": [
            "Specification tests, transformations",
            "Ensure proper sampling techniques",
            "Remove redundant variables, use regularization",
            "Add omitted variables, use instrumental variables",
            "Use robust standard errors, weighted least squares",
            "Use larger sample, bootstrap methods"
        ]
    }
    
    # Convert to DataFrame and display as table
    df_assumptions = pd.DataFrame(assumptions_data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df_assumptions.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Visual representation of concepts
    st.markdown("### 📊 Visual Representation of Key Concepts")
    
    # Create tabs for different visualizations
    tabs = st.tabs(["OLS Regression", "Residual Analysis", "Hypothesis Testing"])
    
    with tabs[0]:
        # OLS Regression visualization
        st.markdown("#### Simple Linear Regression: $Y = \\beta_0 + \\beta_1 X + u$")
        
        # Generate sample data
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 + 1.5 * x + np.random.normal(0, 2, 100)
        
        # Fit model
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data points
        ax.scatter(x, y, alpha=0.6, label='Data points')
        
        # Plot regression line
        ax.plot(x, model.predict(), 'r-', linewidth=2, label=f'OLS: y = {model.params[0]:.2f} + {model.params[1]:.2f}x')
        
        # Add confidence intervals
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        _, lower, upper = wls_prediction_std(model)
        ax.fill_between(x, lower, upper, color='red', alpha=0.1, label='95% confidence interval')
        
        # Styling
        ax.set_xlabel('Independent Variable (X)')
        ax.set_ylabel('Dependent Variable (Y)')
        ax.set_title('Simple Linear Regression Example')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Display regression results
        st.markdown("#### Regression Results Summary")
        
        # Create table of key results
        results_data = {
            "Statistic": ["Coefficient (β₀)", "Coefficient (β₁)", "Standard Error of β₁", "t-statistic", "p-value", "R²"],
            "Value": [
                f"{model.params[0]:.4f}",
                f"{model.params[1]:.4f}",
                f"{model.bse[1]:.4f}",
                f"{model.tvalues[1]:.4f}",
                f"{model.pvalues[1]:.4f}",
                f"{model.rsquared:.4f}"
            ]
        }
        
        st.table(pd.DataFrame(results_data))
        
    with tabs[1]:
        # Residual analysis
        st.markdown("#### Residual Analysis")
        
        # Create subplots for different residual visualizations
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        axs[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
        axs[0, 0].axhline(y=0, color='r', linestyle='--')
        axs[0, 0].set_xlabel('Fitted values')
        axs[0, 0].set_ylabel('Residuals')
        axs[0, 0].set_title('Residuals vs Fitted')
        
        # Histogram of residuals
        axs[0, 1].hist(model.resid, bins=15, alpha=0.6, density=True)
        
        # Add a normal distribution curve
        from scipy import stats
        xmin, xmax = axs[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(model.resid), np.std(model.resid))
        axs[0, 1].plot(x, p, 'k', linewidth=2)
        
        axs[0, 1].set_xlabel('Residuals')
        axs[0, 1].set_ylabel('Density')
        axs[0, 1].set_title('Histogram of Residuals')
        
        # Q-Q plot
        QQ = ProbPlot(model.resid)
        QQ.qqplot(line='45', alpha=0.6, ax=axs[1, 0])
        axs[1, 0].set_title('Q-Q Plot')
        
        # Scale-Location Plot (sqrt of abs. residuals vs. fitted)
        axs[1, 1].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.6)
        axs[1, 1].set_xlabel('Fitted values')
        axs[1, 1].set_ylabel('√|Residuals|')
        axs[1, 1].set_title('Scale-Location Plot')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        Residual analysis helps check OLS assumptions:
        
        1. **Residuals vs Fitted:** Should show no pattern (checks linearity, zero conditional mean)
        2. **Histogram of Residuals:** Should be approximately normal (checks normality assumption)
        3. **Q-Q Plot:** Points should follow the diagonal line (checks normality)
        4. **Scale-Location:** Should show constant spread (checks homoskedasticity)
        """)
        
    with tabs[2]:
        # Hypothesis testing
        st.markdown("#### Hypothesis Testing in Regression")
        
        st.markdown("""
        ##### Testing the Slope Coefficient ($\\beta_1$)
        
        Most common hypothesis test in regression:
        
        $H_0: \\beta_1 = 0$ (No effect)  
        $H_1: \\beta_1 \\neq 0$ (There is an effect)
        
        **Decision Rule:**
        - If |t-statistic| > critical value: Reject $H_0$
        - If p-value < significance level (α): Reject $H_0$
        
        **Interpretation:**
        - Rejecting $H_0$ means X has a statistically significant effect on Y
        - Failing to reject $H_0$ means insufficient evidence that X affects Y
        """)
        
        # Confidence interval visualization
        ci_lower = model.params[1] - 1.96 * model.bse[1]
        ci_upper = model.params[1] + 1.96 * model.bse[1]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot the distribution
        x_ci = np.linspace(model.params[1] - 4*model.bse[1], model.params[1] + 4*model.bse[1], 1000)
        y_ci = stats.norm.pdf(x_ci, model.params[1], model.bse[1])
        ax.plot(x_ci, y_ci, 'b-', linewidth=2)
        
        # Shade the confidence interval
        ci_x = np.linspace(ci_lower, ci_upper, 100)
        ci_y = stats.norm.pdf(ci_x, model.params[1], model.bse[1])
        ax.fill_between(ci_x, ci_y, color='blue', alpha=0.2)
        
        # Add vertical lines
        ax.axvline(x=model.params[1], color='r', linestyle='-', label=f'β₁ = {model.params[1]:.2f}')
        ax.axvline(x=ci_lower, color='g', linestyle='--', label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
        ax.axvline(x=ci_upper, color='g', linestyle='--')
        ax.axvline(x=0, color='k', linestyle=':', label='H₀: β₁ = 0')
        
        ax.set_title('95% Confidence Interval for β₁')
        ax.set_xlabel('β₁ value')
        ax.set_ylabel('Probability density')
        ax.legend()
        
        st.pyplot(fig)
        
        st.markdown(f"""
        **Interpretation:**
        
        In this example:
        - Estimated β₁: {model.params[1]:.4f}
        - Standard Error: {model.bse[1]:.4f}
        - 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
        - t-statistic: {model.tvalues[1]:.4f}
        - p-value: {model.pvalues[1]:.4f}
        
        Since the confidence interval does not include zero and p-value < 0.05, we reject H₀.
        This means X has a statistically significant effect on Y.
        """)
    
    # Extensions of regression analysis
    st.markdown("### 🔄 Extensions of Regression Analysis")
    
    extensions_data = {
        "Model Type": [
            "Multiple Regression", 
            "Log-Linear Models", 
            "Polynomial Regression", 
            "Interaction Terms",
            "Fixed Effects",
            "Random Effects",
            "Instrumental Variables"
        ],
        "Form": [
            "$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_k X_k + u$",
            "$\\ln(Y) = \\beta_0 + \\beta_1 X + u$",
            "$Y = \\beta_0 + \\beta_1 X + \\beta_2 X^2 + ... + \\beta_k X^k + u$",
            "$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\beta_3 X_1 X_2 + u$",
            "$Y_{it} = \\alpha_i + \\beta X_{it} + u_{it}$",
            "$Y_{it} = \\beta X_{it} + (\\alpha_i + u_{it})$",
            "$Y = \\beta_0 + \\beta_1 \\hat{X} + u$, where $\\hat{X}$ is fitted from $X = \\pi_0 + \\pi_1 Z + v$"
        ],
        "Interpretation": [
            "β₁ is effect of X₁ on Y, holding all other X's constant",
            "β₁ is % change in Y for a 1-unit change in X",
            "Allows for nonlinear relationships",
            "Effect of X₁ on Y depends on X₂",
            "Controls for time-invariant unobserved heterogeneity",
            "Treats individual effects as random",
            "Addresses endogeneity by using instrument Z"
        ]
    }
    
    # Convert to DataFrame and display as table
    df_ext = pd.DataFrame(extensions_data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df_ext.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Application example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Returns to Education**
    
    Model: $\\ln(wage) = \\beta_0 + \\beta_1 education + \\beta_2 experience + \\beta_3 experience^2 + u$
    
    **Interpretation:**
    - $\\beta_1 = 0.08$ means a 1-year increase in education is associated with an 8% increase in wages, holding experience constant.
    - $\\beta_2 > 0$ and $\\beta_3 < 0$ indicate that returns to experience increase but at a decreasing rate.
    
    **Potential issues:**
    - Omitted variable bias: Ability might affect both education and wages
    - Endogeneity: Education might be chosen based on expected wage returns
    - Selection bias: We only observe wages for employed individuals
    
    **Solutions:**
    - Control for proxies of ability
    - Use instrumental variables (e.g., distance to college)
    - Use Heckman selection model
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Central Limit Theorem & Sampling Distributions")
    st.markdown("""
    Regression analysis relies heavily on:
    - Sampling distributions of the estimators $\\hat{\\beta}_0$ and $\\hat{\\beta}_1$
    - Central Limit Theorem for hypothesis testing and confidence intervals
    - Understanding the uncertainty in our estimates through standard errors
    
    The next section will explore these concepts in more detail.
    """) 