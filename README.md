# Quantitative Backtesting Platform 🚀

A comprehensive Streamlit-based quantitative analysis and backtesting platform for financial data analysis, factor research, and portfolio optimization.

## Quick Start

### Method 1: Using pip (Recommended for quick setup)

```bash
# 0. Python 3.10+

# 1. Clone the repository
git clone https://github.com/nfxfinance/ascent_research_platform
cd ascent_research_platform

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the application
streamlit run main_app.py --server.port 12851 --server.address 0.0.0.0
```


### Mount Historical Analysis

```bash
mkdir saved_analyses
# 密码是通用密码
sshfs -o port=10022 marlon@h.adpolitan.com:/mnt/datas/saved_analyses ./saved_analyses
```

