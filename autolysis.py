import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# Set your API key from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Function to load and display the dataset info
def load_data(filename):
    data = pd.read_csv(filename)
    print("Dataset Loaded.")
    print(data.info())  # Display info to understand types and missing values
    return data

# Perform summary statistics and missing value analysis
def analyze_data(data):
    # Summary statistics
    summary_stats = data.describe(include='all')
    print("Summary Statistics:\n", summary_stats)
    
    # Missing values count
    missing_values = data.isnull().sum()
    print("Missing Values:\n", missing_values)
    
    # Correlation matrix
    correlation_matrix = data.corr()
    print("Correlation Matrix:\n", correlation_matrix)
    
    return summary_stats, missing_values, correlation_matrix

# Function to detect and visualize outliers
def detect_outliers(data):
    # Boxplot for outlier detection
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title("Boxplot for Outlier Detection")
    plt.savefig('outliers.png')
    plt.close()
    print("Outlier Detection Plot Saved as outliers.png")

# Function to generate a correlation heatmap
def plot_correlation(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Correlation Heatmap Saved as correlation_heatmap.png")

# Function to request a summary story from LLM
def generate_story(summary_stats, missing_values, correlation_matrix, filename):
    # Preparing the context for the LLM
    context = {
        'filename': filename,
        'summary_stats': summary_stats.to_dict(),
        'missing_values': missing_values.to_dict(),
        'correlation_matrix': correlation_matrix.to_dict()
    }
    
    # LLM Request: Ask for a story summary
    prompt = f"""
    Analyze the dataset {context['filename']} with the following summary statistics, missing values, and correlation matrix:
    
    Summary Statistics:
    {context['summary_stats']}
    
    Missing Values:
    {context['missing_values']}
    
    Correlation Matrix:
    {context['correlation_matrix']}
    
    Based on this analysis, generate a narrative story about the data. Explain:
    1. What the data represents
    2. Key insights from the analysis
    3. What actions or recommendations can be made based on these insights.
    """
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        temperature=0.7
    )
    
    story = response.choices[0].text.strip()
    print("Generated Story:\n", story)
    
    # Write the story to a README.md file
    with open('README.md', 'w') as f:
        f.write(f"# Data Analysis Report for {filename}\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"Filename: {filename}\n")
        f.write("## Analysis Summary\n")
        f.write("### Summary Statistics\n")
        f.write(str(summary_stats))
        f.write("\n### Missing Values\n")
        f.write(str(missing_values))
        f.write("\n### Correlation Matrix\n")
        f.write(str(correlation_matrix))
        f.write("\n## Insights and Story\n")
        f.write(story)
    
    print("Story written to README.md")

# Main function to run the entire analysis
def main():
    import sys
    if len(sys.argv) < 2:
        print("Please provide a CSV filename as an argument.")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Load the data
    data = load_data(filename)
    
    # Perform the analysis
    summary_stats, missing_values, correlation_matrix = analyze_data(data)
    
    # Visualize the results
    detect_outliers(data)
    plot_correlation(correlation_matrix)
    
    # Generate a narrative story from the analysis
    generate_story(summary_stats, missing_values, correlation_matrix, filename)

if __name__ == "__main__":
    main()
To create a Python script `autolysis.py` that satisfies the requirements of the task, we'll go step-by-step through the necessary actions: data analysis, visualization, and narration.

### 1. **Install necessary libraries**

To begin, you'll need libraries for data analysis, visualization, and interacting with the LLM (via the `openai` library). Install them using pip:

```bash
pip install pandas numpy seaborn matplotlib openai
```

### 2. **Script Overview**

We'll build the `autolysis.py` script which:
- Loads a CSV file.
- Performs generic exploratory data analysis (EDA) like summary statistics, missing value counting, correlations, and outlier detection.
- Creates visualizations (heatmaps, histograms, etc.).
- Sends results to an LLM (OpenAI GPT-3 or equivalent) for further analysis and story narration.
- Saves the outputs as `README.md` and charts in PNG format.

### 3. **The Python Script (`autolysis.py`)**

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import openai

# Set your API key from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

# Function to load and display the dataset info
def load_data(filename):
    data = pd.read_csv(filename)
    print("Dataset Loaded.")
    print(data.info())  # Display info to understand types and missing values
    return data

# Perform summary statistics and missing value analysis
def analyze_data(data):
    # Summary statistics
    summary_stats = data.describe(include='all')
    print("Summary Statistics:\n", summary_stats)
    
    # Missing values count
    missing_values = data.isnull().sum()
    print("Missing Values:\n", missing_values)
    
    # Correlation matrix
    correlation_matrix = data.corr()
    print("Correlation Matrix:\n", correlation_matrix)
    
    return summary_stats, missing_values, correlation_matrix

# Function to detect and visualize outliers
def detect_outliers(data):
    # Boxplot for outlier detection
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title("Boxplot for Outlier Detection")
    plt.savefig('outliers.png')
    plt.close()
    print("Outlier Detection Plot Saved as outliers.png")

# Function to generate a correlation heatmap
def plot_correlation(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print("Correlation Heatmap Saved as correlation_heatmap.png")

# Function to request a summary story from LLM
def generate_story(summary_stats, missing_values, correlation_matrix, filename):
    # Preparing the context for the LLM
    context = {
        'filename': filename,
        'summary_stats': summary_stats.to_dict(),
        'missing_values': missing_values.to_dict(),
        'correlation_matrix': correlation_matrix.to_dict()
    }
    
    # LLM Request: Ask for a story summary
    prompt = f"""
    Analyze the dataset {context['filename']} with the following summary statistics, missing values, and correlation matrix:
    
    Summary Statistics:
    {context['summary_stats']}
    
    Missing Values:
    {context['missing_values']}
    
    Correlation Matrix:
    {context['correlation_matrix']}
    
    Based on this analysis, generate a narrative story about the data. Explain:
    1. What the data represents
    2. Key insights from the analysis
    3. What actions or recommendations can be made based on these insights.
    """
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        temperature=0.7
    )
    
    story = response.choices[0].text.strip()
    print("Generated Story:\n", story)
    
    # Write the story to a README.md file
    with open('README.md', 'w') as f:
        f.write(f"# Data Analysis Report for {filename}\n\n")
        f.write("## Dataset Overview\n")
        f.write(f"Filename: {filename}\n")
        f.write("## Analysis Summary\n")
        f.write("### Summary Statistics\n")
        f.write(str(summary_stats))
        f.write("\n### Missing Values\n")
        f.write(str(missing_values))
        f.write("\n### Correlation Matrix\n")
        f.write(str(correlation_matrix))
        f.write("\n## Insights and Story\n")
        f.write(story)
    
    print("Story written to README.md")

# Main function to run the entire analysis
def main():
    import sys
    if len(sys.argv) < 2:
        print("Please provide a CSV filename as an argument.")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Load the data
    data = load_data(filename)
    
    # Perform the analysis
    summary_stats, missing_values, correlation_matrix = analyze_data(data)
    
    # Visualize the results
    detect_outliers(data)
    plot_correlation(correlation_matrix)
    
    # Generate a narrative story from the analysis
    generate_story(summary_stats, missing_values, correlation_matrix, filename)

if __name__ == "__main__":
    main()
```

### 4. **Explanation of the Script**

- **Data Loading**: 
  - The `load_data` function loads the CSV file using `pandas.read_csv()` and prints basic info about the dataset.

- **Data Analysis**:
  - The `analyze_data` function computes summary statistics, missing values, and the correlation matrix for the dataset.
  
- **Outlier Detection**:
  - The `detect_outliers` function generates a boxplot to detect potential outliers in the dataset.

- **Correlation Heatmap**:
  - The `plot_correlation` function generates and saves a heatmap for the correlation matrix to visualize relationships between variables.

- **Story Generation**:
  - The `generate_story` function sends a detailed prompt to the LLM (via OpenAI’s API), asking it to generate a narrative story based on the summary statistics, missing values, and correlation matrix.
  - The result is saved in `README.md`.

- **Execution**:
  - When you run the script, it accepts a CSV file, processes it, generates visualizations (outliers and correlation heatmap), and writes a detailed markdown report to `README.md`.

### 5. **Running the Script**

To run the script, make sure to set the environment variable for your AI proxy token, then run the script from the command line:

```bash
export AIPROXY_TOKEN='your-api-key-here'
uv run autolysis.py dataset.csv
```

This will generate:
- `README.md`: The markdown file with analysis and insights.
- `outliers.png`: The plot for outlier detection.
- `correlation_heatmap.png`: The heatmap of the correlation matrix.

### 6. **Visualization and Markdown**

- The visualizations will be saved as PNG images, and the markdown file will include references to these images. This approach allows easy integration of both the raw data insights and visual support into a single readable report.

---

This script performs the essential functions of data analysis, visualization, and storytelling in an automated fashion, while leveraging an LLM for generating insights and narrative summaries.