import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, latex
import networkx as nx

def display_section():
    """Display the Fundamental Concepts of Data & Probability section."""
    
    # Section header with blue diamond
    st.markdown('<div class="blue-diamond">◆</div><h2 class="section-header">Fundamental Concepts of Data & Probability</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This section covers the fundamental statistical concepts that form the basis of econometric analysis.
    Understanding these concepts is crucial for interpreting econometric results correctly.
    """)
    
    # Create a DataFrame for the table
    data = {
        "Concept": [
            "Mean<br>(Expected Value, E[X])",
            "Variance (σ²)",
            "Standard Deviation (σ)",
            "Covariance<br>(Cov(X,Y))",
            "Correlation (ρ)",
            "Coefficient of Variation",
            "Probability Density Function (PDF)",
            "Cumulative Distribution Function (CDF)"
        ],
        "Formula": [
            '<span style="font-family: Times New Roman;">μ</span> = E[X] = Σ<sub>i=1</sub><sup>n</sup> x<sub>i</sub>P(x<sub>i</sub>)',
            '<span style="font-family: Times New Roman;">σ</span>² = E[(X - <span style="font-family: Times New Roman;">μ</span>)²] = Σ<sub>i=1</sub><sup>n</sup> (x<sub>i</sub> - <span style="font-family: Times New Roman;">μ</span>)²P(x<sub>i</sub>)',
            '<span style="font-family: Times New Roman;">σ</span> = √(<span style="font-family: Times New Roman;">σ</span>²)',
            'Cov(X,Y) = E[(X - <span style="font-family: Times New Roman;">μ</span><sub>X</sub>)(Y - <span style="font-family: Times New Roman;">μ</span><sub>Y</sub>)]',
            '<span style="font-family: Times New Roman;">ρ</span> = Cov(X,Y)/(<span style="font-family: Times New Roman;">σ</span><sub>X</sub><span style="font-family: Times New Roman;">σ</span><sub>Y</sub>)',
            'CV = <span style="font-family: Times New Roman;">σ</span>/<span style="font-family: Times New Roman;">μ</span>',
            'f(x) = d/dx F(x)',
            'F(x) = P(X ≤ x) = ∫<sub>-∞</sub><sup>x</sup> f(t)dt'
        ],
        "Explanation": [
            "The long-run average of a random variable.",
            "Measures spread by calculating the average squared deviation from the mean.",
            "The square root of variance, giving spread in original units.",
            "Measures whether two variables move together (positive = same direction, negative = opposite).",
            "Standardized measure of association between two variables, ranging from −1 to +1.",
            "Relative measure of dispersion, allows comparison between distributions with different means.",
            "Gives the relative likelihood of a continuous random variable taking a specific value.",
            "Gives the probability that a random variable is less than or equal to a specific value."
        ],
        "Next Step": [
            "Variance (<span style='font-family: Times New Roman;'>σ</span>²) ⬇️",
            "Standard Deviation (<span style='font-family: Times New Roman;'>σ</span>) ⬇️",
            "Coefficient of Variation (Relative Spread) ⬇️",
            "Correlation (<span style='font-family: Times New Roman;'>ρ</span>) ⬇️",
            "Regression Analysis ⬇️",
            "Interpretation of dispersion ⬇️",
            "Probability calculations ⬇️",
            "Hypothesis testing ⬇️"
        ]
    }
    
    # Convert to DataFrame and display as table
    df = pd.DataFrame(data)
    
    # Use st.markdown to display the table with HTML formatting
    st.markdown(
        df.style.hide(axis="index")
        .to_html()
        .replace('<table', '<table class="concept-table"')
        .replace('<td>', '<td style="text-align: left; padding: 8px; font-family: -apple-system, system-ui, BlinkMacSystemFont, \'Segoe UI\', Roboto, \'Helvetica Neue\', Arial, sans-serif;">')
        .replace('<th>', '<th style="text-align: left; background-color: #f2f2f2; padding: 8px;">'),
        unsafe_allow_html=True
    )
    
    # Add Concept Relationship Diagram
    st.markdown("### 🔄 Concept Relationships")
    
    # Create relationship diagram using networkx and matplotlib
    G = nx.DiGraph()
    
    # Add nodes with labels
    nodes = {
        "Data": "Raw observations",
        "Mean": "Central tendency",
        "Variance": "Spread measure",
        "StdDev": "Root of variance",
        "Covariance": "Joint variation",
        "Correlation": "Standardized covariance",
        "PDF": "Density function",
        "CDF": "Cumulative probability"
    }
    
    # Add nodes to graph
    for node, desc in nodes.items():
        G.add_node(node, description=desc)
    
    # Add edges with relationships
    edges = [
        ("Data", "Mean", "summarizes"),
        ("Data", "Variance", "describes spread"),
        ("Variance", "StdDev", "square root"),
        ("Mean", "Variance", "reference point"),
        ("StdDev", "Correlation", "standardizes"),
        ("Covariance", "Correlation", "standardizes"),
        ("PDF", "CDF", "integrates to"),
        ("Mean", "Covariance", "centers data")
    ]
    
    # Add edges to graph
    for source, target, relationship in edges:
        G.add_edge(source, target, relationship=relationship)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw node labels with Greek letters
    node_labels = {
        "Data": "Data",
        "Mean": "Mean (μ)",
        "Variance": "Variance (σ²)",
        "StdDev": "Std Dev (σ)",
        "Covariance": "Cov(X,Y)",
        "Correlation": "Corr (ρ)",
        "PDF": "PDF f(x)",
        "CDF": "CDF F(x)"
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    
    # Draw edges and edge labels
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Relationships Between Statistical Concepts", pad=20)
    plt.axis('off')
    
    st.pyplot(fig)
    
    st.markdown("""
    **Key Relationships:**
    - **Mean (μ) → Variance (σ²)**: The mean is used as the reference point for calculating variance
    - **Variance (σ²) → Standard Deviation (σ)**: Standard deviation is the square root of variance
    - **Covariance → Correlation (ρ)**: Correlation standardizes covariance using standard deviations
    - **PDF → CDF**: CDF is the integral of PDF
    - **Mean & Standard Deviation → Normal Distribution**: These parameters fully define a normal distribution
    """)
    
    # Visual representation of concepts
    st.markdown("### 📊 Visual Representation of Key Concepts")
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Normal Distribution", "Correlation Examples", "Law of Large Numbers"])
    
    with tabs[0]:
        # Normal distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(-4, 4, 1000)
        y = 1/(np.sqrt(2*np.pi)) * np.exp(-x**2/2)
        
        ax.plot(x, y, 'b-', linewidth=2)
        ax.fill_between(x[(x >= -1) & (x <= 1)], y[(x >= -1) & (x <= 1)], color='skyblue', alpha=0.5)
        ax.fill_between(x[(x >= -2) & (x <= 2)], y[(x >= -2) & (x <= 2)], color='lightblue', alpha=0.3)
        ax.fill_between(x[(x >= -3) & (x <= 3)], y[(x >= -3) & (x <= 3)], color='lightgray', alpha=0.2)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.45)
        ax.set_title('Standard Normal Distribution')
        ax.set_xlabel('Z-score (Standard Deviations from Mean)')
        ax.set_ylabel('Probability Density')
        
        # Add annotations
        ax.annotate('68.2% within 1σ', xy=(0, 0.1), xytext=(0, 0.1), 
                   ha='center', fontsize=10, color='blue')
        ax.annotate('95.4% within 2σ', xy=(0, 0.05), xytext=(0, 0.05), 
                   ha='center', fontsize=10, color='blue')
        ax.annotate('99.7% within 3σ', xy=(0, 0.02), xytext=(0, 0.02), 
                   ha='center', fontsize=10, color='blue')
        
        st.pyplot(fig)
        
        st.markdown("""
        **The Normal Distribution** is central to econometrics because:
        - Many economic variables are approximately normally distributed
        - The Central Limit Theorem ensures sample means approach normal distribution
        - Statistical tests often assume normality
        """)
        
    with tabs[1]:
        # Correlation examples
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Positive correlation
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        y1 = x1 * 0.8 + np.random.normal(0, 0.5, 100)
        axs[0].scatter(x1, y1, alpha=0.7)
        axs[0].set_title('Positive Correlation (ρ ≈ 0.8)')
        axs[0].set_xlabel('Variable X')
        axs[0].set_ylabel('Variable Y')
        
        # No correlation
        np.random.seed(42)
        x2 = np.random.normal(0, 1, 100)
        y2 = np.random.normal(0, 1, 100)
        axs[1].scatter(x2, y2, alpha=0.7)
        axs[1].set_title('No Correlation (ρ ≈ 0)')
        axs[1].set_xlabel('Variable X')
        axs[1].set_ylabel('Variable Y')
        
        # Negative correlation
        np.random.seed(42)
        x3 = np.random.normal(0, 1, 100)
        y3 = x3 * -0.8 + np.random.normal(0, 0.5, 100)
        axs[2].scatter(x3, y3, alpha=0.7)
        axs[2].set_title('Negative Correlation (ρ ≈ -0.8)')
        axs[2].set_xlabel('Variable X')
        axs[2].set_ylabel('Variable Y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Correlation** measures the strength and direction of linear relationship between two variables:
        - ρ = +1: Perfect positive correlation
        - ρ = 0: No correlation
        - ρ = -1: Perfect negative correlation
        - **Important:** Correlation does not imply causation!
        """)
        
    with tabs[2]:
        # Law of large numbers visualization
        st.markdown("""
        ### Law of Large Numbers
        
        The Law of Large Numbers states that as the sample size increases, the sample mean approaches the population mean.
        
        This interactive widget demonstrates this concept by simulating coin flips (with 0.5 probability of heads).
        """)
        
        # Simulation parameters
        n_flips = st.slider("Number of coin flips", 10, 1000, 100)
        
        # Run simulation
        np.random.seed(42)  # For reproducibility
        flips = np.random.binomial(1, 0.5, n_flips)
        running_mean = np.cumsum(flips) / np.arange(1, n_flips + 1)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, n_flips + 1), running_mean, 'b-')
        ax.axhline(y=0.5, color='r', linestyle='--', label='True Probability (0.5)')
        ax.set_xlabel('Number of Flips')
        ax.set_ylabel('Running Average (Proportion of Heads)')
        ax.set_title('Law of Large Numbers: Coin Flip Simulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.markdown("""
        This simulation shows how the sample average converges to the true population value as the sample size increases.
        This principle is fundamental to statistical inference in econometrics.
        """)
    
    # Key properties and relationships
    st.markdown("### 🔑 Key Properties and Relationships")
    
    st.markdown("""
    - **Variance Properties:**
      - Var(aX + b) = a²Var(X)
      - Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
      - If X and Y are independent: Var(X + Y) = Var(X) + Var(Y)
    
    - **Covariance Properties:**
      - Cov(X, X) = Var(X)
      - Cov(aX, bY) = ab·Cov(X,Y)
      - Cov(X₁ + X₂, Y) = Cov(X₁, Y) + Cov(X₂, Y)
    
    - **Correlation Properties:**
      - -1 ≤ ρ ≤ 1
      - ρ = 0 means linear independence (but not necessarily statistical independence)
      - |ρ| = 1 means perfect linear relationship
    """)
    
    # Application example
    st.markdown("### 📝 Application Example")
    
    st.markdown("""
    **Example:** A stock has historical annual returns with mean μ = 8% and standard deviation σ = 15%.
    
    1. **Probability of Positive Return**: Assuming returns are normally distributed, P(Return > 0) = ?
    2. **Variance Reduction through Diversification**: How does combining stocks reduce portfolio risk?
    
    **Solution:**
    
    1. Convert to z-score: z = (0 - 8) / 15 = -0.533
       P(Return > 0) = P(Z > -0.533) = 0.703 or 70.3%
    
    2. If returns of stocks are uncorrelated, the variance of an equally weighted portfolio of n stocks is:
       σ²ᵖ = σ²/n
       
       So a portfolio of 25 stocks would have roughly 1/25 or 4% of the variance of a single stock.
    """)
    
    # Connection to next section
    st.markdown("### ⬇️ Connection to Regression Analysis")
    st.markdown("""
    Understanding these fundamental concepts is essential for regression analysis, which:
    - Uses expected values to estimate parameters
    - Employs variance and standard deviation to assess uncertainty
    - Utilizes correlation and covariance to analyze relationships
    """) 