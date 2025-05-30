# This is sentiment analysis and financial analysis for stock 

This project performs exploratory data analysis (EDA), sentiment analysis, and financial analysis on stock analyst ratings and related news headlines. The goal is to uncover trends, patterns, and insights from raw analyst ratings data, including publisher activity, headline content, and temporal publication trends.

## Features

- **Data Loading & Cleaning:** Robust loading of raw analyst ratings data with error handling for missing or empty files.
- **Exploratory Data Analysis:** 
  - Dataset profiling (shape, missing values, column info)
  - Descriptive statistics for headline lengths and publisher activity
  - Time series analysis of article publication trends (by day, hour, weekday, month, year)
- **Text Analysis:** 
  - Preprocessing of news headlines (lowercasing, punctuation removal, stopword filtering)
  - Extraction of top keywords and phrases using n-grams
- **Publisher Analysis:** 
  - Identification of top publishers and their activity
  - Extraction of publisher domains (for email-based publishers)
- **Visualization:** 
  - Plots for publication trends and top publishers

## Project Structure

```
stock_analysis_week1/
│
├── notebooks/
│   └── eda_raw_analysts_rating.ipynb   # Main analysis notebook
├── data/
│   └── raw_analyst_ratings.csv         # Raw data (not tracked in git)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files and folders to ignore in git
├── .github/
│   └── workflows/
│       └── unittests.yml               # GitHub Actions workflow (if present)
└── README.md                           # Project documentation
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd stock_analysis_week1
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate   # On Windows
   # or
   source venv/bin/activate   # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Place your data file:**
   - Add `raw_analyst_ratings.csv` to the `data/` directory.

5. **Run the analysis:**
   - Open `notebooks/eda_raw_analysts_rating.ipynb` in Jupyter or VS Code and run the cells.

## Notes

- The data file is not included in the repository. Please provide your own `raw_analyst_ratings.csv` in the `data/` folder.
- The `.gitignore` ensures that data files, virtual environments, and notebook checkpoints are not tracked by git.

## License

This project is for educational and research purposes.

---

*Feel free to modify this README to better fit your project’s specifics!*
