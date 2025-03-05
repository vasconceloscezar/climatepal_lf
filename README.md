Live demo at 

https://climate-pal.namastex.ai/flow/e5a6ccad-1aa6-458b-abcc-ed9b9809b5ff/folder/54aad242-ee9a-4a53-960f-3fe4f6177497

Accounts with:

OpenAI (for an API key)
Groq (Free tier for API key)

Supabase (for a database)

1. Install Langflow:

Follow the official installation instructions: https://github.com/langflow-ai/langflow

Recommended installation: uv pip install langflow

Verify installation: langflow --version

2. Launch Langflow:

Run: langflow

Access Langflow in your browser (usually http://127.0.0.1:7860).

3. Import the Flow:

Download Dataset Retrieval - v1.0.3.json from the flow/ directory.

In Langflow, create a new flow and import the downloaded JSON file.

4. API Key Setup:

For the following components, enter your API keys and Supabase details in the component settings:

OpenAIModel (OpenAI API Key)

GroqModel (Groq API Key)

Supabase Save Configuration (Supabase URL, Supabase Key, Table Name)

5. Create a Supabase Server:

Sign up at https://supabase.com/ and create a new project.

Note your "Project URL" and the "service_role" API key from the Supabase dashboard's API settings.

In the Supabase SQL Editor, run the SQL script from sql/create_feedback_results.sql to create the sentiment_feedback table.

Check the flow Notes for more detailed instructions.

With these steps you can open the playground and talk to the Climate Pal agent.

## Running the Evaluation Script

The evaluation script allows you to test the Climate Pal agent against a set of predefined queries and measure its accuracy in extracting climate data information.

### Prerequisites

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install them individually:
```bash
pip install pandas requests tqdm python-dotenv
```

3. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
LANGFLOW_API_KEY=your_langflow_api_key
LANGFLOW_URL=https://api.langflow.tech/api/v1/predict
```

### Preparing Evaluation Data

Create a CSV file at `data/evaluation_queries.csv` with the following columns:
- `query`: The query to test
- `scenario`: The expected scenario/experiment (e.g., "ssp119")
- `variable`: The expected variable (e.g., "tas")
- `year_range`: The expected year range (e.g., "2015-2050")
- `file_path`: The expected file path URL

You can use the provided sample file `data/evaluation_queries_sample.csv` as a template. Example CSV structure:

```csv
query,scenario,variable,year_range,file_path
"I need near-surface air temperature data for the ssp119 CO2 emissions scenario from January 2015 to December 2050",ssp119,tas,2015-2050,https://example.com/data/tas_Amon_MODEL_ssp119_r1i1p1f1_gn_201501-205012.nc
```

### Basic Usage

To run the evaluation script:

```bash
python evaluate.py
```

### Command-line Options

The script supports several command-line options:

- `--samples N`: Process only N samples from the CSV file (default: process all)
- `--output-dir DIR`: Directory to store evaluation results (default: "evaluation_results")
- `--debug`: Enable debug mode to save raw API responses
- `--max-retries N`: Maximum number of API retry attempts (default: 3)
- `--retry-delay N`: Delay between retry attempts in seconds (default: 5)
- `--max-workers N`: Maximum number of parallel workers (default: 4)

### Interrupting the Evaluation

The evaluation script now supports graceful interruption using Ctrl+C. When you press Ctrl+C:

1. The script will stop accepting new queries
2. It will wait for currently running queries to complete
3. It will save partial results to a file named `evaluation_results_partial_TIMESTAMP.csv`
4. A summary of the processed queries will be displayed

If you press Ctrl+C a second time, the script will exit immediately without waiting for running queries to complete.

### Examples

Process all queries with 8 parallel workers:
```bash
python evaluate.py --max-workers 8
```

Process only the first 10 queries with debug mode enabled:
```bash
python evaluate.py --samples 10 --debug
```

Process all queries with increased retry attempts:
```bash
python evaluate.py --max-retries 5 --retry-delay 10
```

### Results

The script will generate a CSV file with the evaluation results and display a summary of the accuracy metrics:

- Scenario Accuracy: Percentage of correctly identified scenarios
- Variable Accuracy: Percentage of correctly identified variables
- Year Range Accuracy: Percentage of correctly identified year ranges
- File Path Accuracy: Percentage of correctly identified file paths
- Overall Accuracy: Percentage of queries where all four elements were correctly identified

In debug mode, the script will also save the raw API responses to the debug directory for further analysis.

## Troubleshooting

Here are some common issues you might encounter when running the evaluation script:

### API Connection Issues

- **Problem**: The script fails to connect to the API or returns connection errors.
- **Solution**: 
  - Verify your API keys in the `.env` file are correct
  - Check if the LangFlow service is running and accessible
  - Increase the `--max-retries` and `--retry-delay` parameters

### Missing Dependencies

- **Problem**: Import errors or missing module errors.
- **Solution**:
  - Ensure all required packages are installed: `pip install -r requirements.txt`
  - Check for any Python version compatibility issues (Python 3.8+ is recommended)

### Performance Issues

- **Problem**: The script is running too slowly.
- **Solution**:
  - Adjust the `--max-workers` parameter to increase parallel processing
  - For large datasets, use the `--samples` parameter to test with a smaller subset first

### Incorrect Results

- **Problem**: The script is not correctly identifying the expected values.
- **Solution**:
  - Enable debug mode with `--debug` to inspect the raw API responses
  - Check that your CSV file is formatted correctly
  - Verify that the expected values in your CSV match exactly what the API should return

### File Path Matching Issues

- **Problem**: File paths are not matching even though they seem correct.
- **Solution**:
  - Ensure the file paths in your CSV exactly match the format returned by the API
  - Check for differences in URL encoding, relative vs. absolute paths, or trailing slashes