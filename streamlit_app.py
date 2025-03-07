import streamlit as st
from sections import (
    data_probability,
    regression_analysis,
    clt_sampling,
    probability_risk,
    time_series,
    panel_data,
    instrumental_variables,
    limited_dependent,
    maximum_likelihood,
    simulation
)

def main():
    """Main function to run the Econometrics Cheatsheet app."""
    
    # Set page config
    st.set_page_config(
        page_title="Econometrics Cheatsheet",
        page_icon="📊",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .orange-diamond {
        color: #FF9F1C;
        font-size: 24px;
        margin-right: 10px;
        display: inline;
    }
    .section-header {
        color: #2C3E50;
        display: inline;
    }
    .concept-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 15px;
    }
    .concept-table th, .concept-table td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
        vertical-align: middle;
        line-height: 1.5;
    }
    .concept-table th {
        background-color: #f2f2f2;
        font-weight: 600;
    }
    .concept-table td {
        font-family: -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    /* Math formatting */
    sub, sup {
        font-size: 75%;
        line-height: 0;
        position: relative;
        vertical-align: baseline;
    }
    sup {
        top: -0.5em;
    }
    sub {
        bottom: -0.25em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("📚 Interactive Econometrics Cheatsheet")
    
    st.markdown("""
    Welcome to the Interactive Econometrics Cheatsheet! This tool provides a comprehensive
    overview of key econometric concepts, methods, and applications. Use the navigation
    below to explore different topics.
    
    Each section includes:
    - 📝 Key concepts and definitions
    - 📊 Interactive visualizations
    - 🧮 Formula explanations
    - 💻 Code examples
    - 📈 Applications
    """)
    
    # Create sections
    sections = {
        "Data & Probability": data_probability.display_section,
        "Regression Analysis": regression_analysis.display_section,
        "Central Limit Theorem": clt_sampling.display_section,
        "Probability Calculations": probability_risk.display_section,
        "Time Series Analysis": time_series.display_section,
        "Panel Data Methods": panel_data.display_section,
        "Instrumental Variables": instrumental_variables.display_section,
        "Limited Dependent Variables": limited_dependent.display_section,
        "Maximum Likelihood": maximum_likelihood.display_section,
        "Simulation Methods": simulation.display_section
    }
    
    # Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Choose a section:", list(sections.keys()))
    
    # Display selected section
    sections[section]()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Cheatsheet**
    
    This interactive cheatsheet is designed to help students and practitioners understand
    key concepts in econometrics. It combines theoretical explanations with practical
    examples and interactive visualizations.
    
    Built with:
    - Streamlit
    - Python
    - NumPy, Pandas, SciPy
    - Matplotlib
    
    For questions or suggestions, please raise an issue on the GitHub repository.
    """)

if __name__ == "__main__":
    main() 