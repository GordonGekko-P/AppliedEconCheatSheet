# Applied Econometrics Cheatsheet

An interactive web application that serves as a comprehensive cheatsheet for applied econometrics concepts. Built with Streamlit, this application provides:

- Interactive visualizations of key statistical concepts
- Comprehensive formulas and explanations
- Real-world examples and applications
- Interactive calculators for probability and risk analysis

## Features

- **Fundamental Concepts of Data & Probability**
  - Mean, variance, standard deviation
  - Correlation and covariance
  - Interactive normal distribution visualization
  - Law of Large Numbers demonstration

- **Regression Analysis**
  - OLS regression concepts and formulas
  - Interactive regression visualization
  - Residual analysis
  - Hypothesis testing visualization

- **Central Limit Theorem & Sampling Distributions**
  - Interactive CLT demonstrations
  - Sampling distribution visualizations
  - Effect of sample size on distributions
  - Confidence interval calculations

- **Probability Calculations & Risk Analysis**
  - Z-score calculator
  - Risk pooling demonstrations
  - Value at Risk (VaR) calculator
  - Interactive hypothesis testing visualization

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/appliedeconcheatsheet.git
   cd appliedeconcheatsheet
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to navigate between different sections of the cheatsheet

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Statistical visualizations powered by [Matplotlib](https://matplotlib.org/)
- Statistical computations using [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/)
- Econometric models using [statsmodels](https://www.statsmodels.org/) 