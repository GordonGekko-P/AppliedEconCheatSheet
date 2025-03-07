import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def display_section():
    """Display the Time Series Analysis section."""
    
    # Section header with orange diamond
    st.markdown('<div class="orange-diamond">◆</div><h2 class="section-header">Time Series Analysis</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers key concepts and techniques in time series analysis, including stationarity,
    autocorrelation, and various time series models used in econometric analysis.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "Stationarity",
            "Autocorrelation",
            "AR(p) Model",
            "MA(q) Model",
            "ARMA(p,q)",
            "ARIMA(p,d,q)",
            "Unit Root",
            "Cointegration"
        ],
        "Formula/Definition": [
            "E[Yₜ] = μ, Var(Yₜ) = σ², Cov(Yₜ,Yₜ₋ₖ) = f(k)",
            "$\\rho_k = \\frac{Cov(Y_t, Y_{t-k})}{\\sqrt{Var(Y_t)Var(Y_{t-k})}}$",
            "$Y_t = c + \\sum_{i=1}^p \\phi_i Y_{t-i} + \\epsilon_t$",
            "$Y_t = c + \\epsilon_t + \\sum_{i=1}^q \\theta_i \\epsilon_{t-i}$",
            "$Y_t = c + \\sum_{i=1}^p \\phi_i Y_{t-i} + \\epsilon_t + \\sum_{i=1}^q \\theta_i \\epsilon_{t-i}$",
            "ARMA model on differenced series",
            "$Y_t = \\rho Y_{t-1} + \\epsilon_t$ where $\\rho = 1$",
            "Linear combination of non-stationary series is stationary"
        ],
        "Tests/Diagnostics": [
            "ADF, KPSS tests",
            "Ljung-Box Q-test",
            "t-tests on AR coefficients",
            "t-tests on MA coefficients",
            "AIC, BIC for model selection",
            "ACF, PACF plots",
            "Dickey-Fuller test",
            "Johansen test"
        ],
        "Application": [
            "Required for valid inference",
            "Identify serial dependence",
            "Model persistent series",
            "Model finite dependence",
            "Complex dynamics",
            "Non-stationary series",
            "Test for non-stationarity",
            "Long-run relationships"
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
    
    # Time Series Components
    st.markdown("### 📊 Time Series Components")
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Components", "Stationarity", "ACF/PACF"])
    
    with tabs[0]:
        st.markdown("#### Decomposition of Time Series")
        
        # Generate sample time series
        np.random.seed(42)
        t = np.linspace(0, 4, 200)
        
        # Trend
        trend = 0.5 * t
        
        # Seasonal component
        seasonal = 0.5 * np.sin(2 * np.pi * t)
        
        # Cyclical component
        cyclical = 0.3 * np.sin(2 * np.pi * t / 4)
        
        # Random component
        random = np.random.normal(0, 0.1, len(t))
        
        # Combined series
        y = trend + seasonal + cyclical + random
        
        # Create plot
        fig, axs = plt.subplots(5, 1, figsize=(12, 12))
        
        # Plot original series
        axs[0].plot(t, y, 'b-')
        axs[0].set_title('Original Time Series')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        
        # Plot trend
        axs[1].plot(t, trend, 'r-')
        axs[1].set_title('Trend Component')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Value')
        
        # Plot seasonal component
        axs[2].plot(t, seasonal, 'g-')
        axs[2].set_title('Seasonal Component')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Value')
        
        # Plot cyclical component
        axs[3].plot(t, cyclical, 'purple')
        axs[3].set_title('Cyclical Component')
        axs[3].set_xlabel('Time')
        axs[3].set_ylabel('Value')
        
        # Plot random component
        axs[4].plot(t, random, 'gray')
        axs[4].set_title('Random Component')
        axs[4].set_xlabel('Time')
        axs[4].set_ylabel('Value')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        Time series can be decomposed into four main components:
        1. **Trend**: Long-term movement in the series
        2. **Seasonal**: Regular patterns of fluctuation
        3. **Cyclical**: Irregular but systematic variation
        4. **Random**: Unpredictable fluctuations
        """)
    
    with tabs[1]:
        st.markdown("#### Stationarity Visualization")
        
        # Create different types of non-stationary series
        t = np.linspace(0, 10, 500)
        
        # Random walk
        np.random.seed(42)
        random_walk = np.cumsum(np.random.normal(0, 0.1, len(t)))
        
        # Trend stationary
        trend_stationary = 0.5 * t + np.random.normal(0, 0.5, len(t))
        
        # Heteroskedastic
        heteroskedastic = np.random.normal(0, 0.1 * (1 + t), len(t))
        
        # Stationary
        stationary = np.random.normal(0, 1, len(t))
        
        # Create plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot random walk
        axs[0, 0].plot(t, random_walk, 'b-')
        axs[0, 0].set_title('Random Walk\n(Non-stationary in mean and variance)')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Value')
        
        # Plot trend stationary
        axs[0, 1].plot(t, trend_stationary, 'r-')
        axs[0, 1].set_title('Trend Stationary\n(Non-stationary in mean)')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Value')
        
        # Plot heteroskedastic
        axs[1, 0].plot(t, heteroskedastic, 'g-')
        axs[1, 0].set_title('Heteroskedastic\n(Non-stationary in variance)')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('Value')
        
        # Plot stationary
        axs[1, 1].plot(t, stationary, 'purple')
        axs[1, 1].set_title('Stationary Series\n(Constant mean and variance)')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Stationarity Requirements:**
        1. Constant mean over time
        2. Constant variance over time
        3. Constant autocovariance over time
        
        **Common Types of Non-stationarity:**
        - Random Walk: Unit root non-stationarity
        - Trend Stationarity: Deterministic trend
        - Heteroskedasticity: Time-varying variance
        """)
    
    with tabs[2]:
        st.markdown("#### Autocorrelation and Partial Autocorrelation")
        
        # Generate AR(1) process
        np.random.seed(42)
        n = 200
        phi = 0.7
        ar1 = np.zeros(n)
        for t in range(1, n):
            ar1[t] = phi * ar1[t-1] + np.random.normal(0, 1)
        
        # Calculate ACF and PACF
        max_lag = 20
        acf = np.array([1] + [np.corrcoef(ar1[:-i], ar1[i:])[0,1] for i in range(1, max_lag+1)])
        
        # Create plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot time series
        axs[0].plot(ar1, 'b-')
        axs[0].set_title('AR(1) Process: Yₜ = 0.7Yₜ₋₁ + εₜ')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Value')
        
        # Plot ACF
        lags = range(len(acf))
        axs[1].vlines(lags, [0], acf)
        axs[1].plot(lags, [0]*len(lags), 'k--', alpha=0.5)
        # Add confidence intervals
        ci = 1.96/np.sqrt(n)
        axs[1].fill_between(lags, -ci, ci, alpha=0.2, color='blue')
        axs[1].set_title('Autocorrelation Function (ACF)')
        axs[1].set_xlabel('Lag')
        axs[1].set_ylabel('ACF')
        
        # Plot PACF (simplified)
        pacf = np.zeros(max_lag+1)
        pacf[0] = 1
        pacf[1] = acf[1]
        for i in range(2, max_lag+1):
            # Use Yule-Walker equations (simplified)
            r = acf[:i]
            R = np.zeros((i-1, i-1))
            for j in range(i-1):
                for k in range(i-1):
                    R[j,k] = acf[abs(j-k)]
            pacf[i] = np.linalg.solve(R, r[1:])[i-2]
        
        axs[2].vlines(lags, [0], pacf)
        axs[2].plot(lags, [0]*len(lags), 'k--', alpha=0.5)
        axs[2].fill_between(lags, -ci, ci, alpha=0.2, color='blue')
        axs[2].set_title('Partial Autocorrelation Function (PACF)')
        axs[2].set_xlabel('Lag')
        axs[2].set_ylabel('PACF')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **ACF and PACF Patterns:**
        
        1. **AR(p) process:**
           - ACF: Decays exponentially
           - PACF: Cuts off after lag p
        
        2. **MA(q) process:**
           - ACF: Cuts off after lag q
           - PACF: Decays exponentially
        
        3. **ARMA(p,q) process:**
           - ACF: Decays exponentially after lag q
           - PACF: Decays exponentially after lag p
        """)
    
    # Model Selection and Diagnostics
    st.markdown("### 📈 Model Selection and Diagnostics")
    
    model_selection = {
        "Criterion": [
            "Akaike Information Criterion (AIC)",
            "Bayesian Information Criterion (BIC)",
            "Ljung-Box Q-test",
            "Jarque-Bera test",
            "Durbin-Watson test"
        ],
        "Formula": [
            "AIC = -2ln(L) + 2k",
            "BIC = -2ln(L) + k·ln(n)",
            "Q = n(n+2)∑(ρ²ₖ/(n-k))",
            "JB = n[(S²/6) + ((K-3)²/24)]",
            "DW = ∑(eₜ - eₜ₋₁)²/∑eₜ²"
        ],
        "Purpose": [
            "Model selection (penalizes complexity)",
            "Model selection (stronger penalty for complexity)",
            "Test for autocorrelation in residuals",
            "Test for normality of residuals",
            "Test for first-order autocorrelation"
        ],
        "Decision Rule": [
            "Choose model with lowest AIC",
            "Choose model with lowest BIC",
            "Reject H₀ if Q > χ²(α,h)",
            "Reject H₀ if JB > χ²(α,2)",
            "Compare to critical values"
        ]
    }
    
    # Convert to DataFrame and display as table
    df_model = pd.DataFrame(model_selection)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df_model.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Application Example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example: Analyzing GDP Growth Rates**
    
    Consider quarterly GDP growth rates for a country:
    
    **Step 1: Check for Stationarity**
    - ADF test: t-stat = -3.5, p = 0.008
    - Conclusion: Series is stationary
    
    **Step 2: Examine ACF/PACF**
    - ACF: Significant at lags 1 and 4
    - PACF: Significant at lags 1 and 4
    - Suggests: AR(4) or ARMA(4,0) with seasonal component
    
    **Step 3: Model Estimation**
    ```
    Model: Yₜ = 0.3 + 0.4Yₜ₋₁ + 0.2Yₜ₋₄ + εₜ
    
    Diagnostics:
    - R² = 0.35
    - AIC = -234.5
    - Q(4) = 3.2 (p = 0.52)
    ```
    
    **Step 4: Forecasting**
    - One-quarter ahead: Ŷₜ₊₁ = 0.3 + 0.4Yₜ + 0.2Yₜ₋₃
    - Calculate prediction intervals using error variance
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Panel Data Methods")
    
    st.markdown("""
    Time series analysis connects to panel data methods through:
    - Fixed effects in time dimension
    - Serial correlation in panel context
    - Dynamic panel models
    - Time-varying parameters
    
    The next section will explore these panel data concepts and methods.
    """) 