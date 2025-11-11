#  Covid-19 Data Analysis Project 

A complete **Covid-19 Data Analysis Capstone Project** built using **Python** and **Jupyter Notebook**, showcasing data cleaning, exploration, visualization, and feature engineering using real-world Covid-19 data.

---

## 📖 Overview
This project performs an in-depth analysis of the global Covid-19 dataset. It follows a full **data science pipeline** — from raw data ingestion and preprocessing to exploratory analysis, statistical insights, and rich visualizations.

---

## 📂 Repository Structure

```
├── covid-data.csv                        # Dataset used for analysis
├── Covid_Analysis_Capstone_Solution.ipynb # Main Jupyter notebook
├── covid_capstone_solution.py            # Equivalent Python script
├── plots/                                # Folder for generated plots
└── df_groupby.csv                        # Aggregated output file
```

---

## 🚀 Project Workflow

### 1️⃣ Data Loading
- Reads dataset from local file or fallback URL

### 2️⃣ High-Level Understanding
- Prints data types, structure, and descriptive stats

### 3️⃣ Low-Level Analysis
- Unique country count, continent-wise frequency, statistical summaries

### 4️⃣ Data Cleaning
- Removes duplicates, handles missing values, cleans invalid entries

### 5️⃣ Date Handling
- Converts date column to datetime format and extracts month info

### 6️⃣ Aggregation
- Groups by continent to compute maximum values for metrics

### 7️⃣ Feature Engineering
- Creates `total_deaths_to_total_cases` ratio

### 8️⃣ Visualization
- **Univariate Analysis**: GDP per capita distribution
- **Scatter Plot**: Total cases vs GDP per capita
- **Pairplot**: Relationships between grouped metrics
- **Bar Plot**: Total cases by continent

### 9️⃣ Export Results
- Saves cleaned dataset and grouped summary as `df_groupby.csv`

---

## 🧰 Tools & Libraries Used

| Library | Purpose |
|----------|----------|
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Seaborn** | Data visualization |
| **Matplotlib** | Plot rendering |
| **Jupyter Notebook** | Interactive environment |

---

## 💡 Key Insights
- Relationship between **GDP per capita** and **Covid-19 case impact**
- **Continent-level** differences in total cases and deaths
- How **Human Development Index (HDI)** relates to health outcomes

---

## 🧭 How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/<your-username>/covid19-data-analysis.git
   cd covid19-data-analysis
   ```

2. Ensure `covid-data.csv` is present in the root folder.

3. Run either:
   - Jupyter Notebook version:
     ```bash
     jupyter notebook Covid_Analysis.ipynb
     ```
   - Python script version:
     ```bash
     python Covid_Analysis_script.py
     ```

4. View results in:
   - `plots/` → all generated graphs  
   - `df_groupby.csv` → aggregated output data

---

## 📊 Sample Visuals

- **GDP per Capita Distribution**
- **Total Cases vs GDP per Capita**
- **Continent-Wise Bar Plot of Cases**
- **Pairplot for Correlation Analysis**

(Plots are automatically saved in the `plots/` folder after execution.)

---

## 📜 License
This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with proper attribution.

---

## ⭐ Acknowledgements
- Dataset source: [Our World in Data](https://ourworldindata.org/coronavirus)
- Project inspired by data analytics and visualization best practices.

---

⭐ **If you find this project useful, please give it a star on GitHub!** ⭐
