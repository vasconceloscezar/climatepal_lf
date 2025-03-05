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

The evaluation script allows you to test the Climate Pal API with a set of predefined queries and measure its accuracy. The script supports parallel processing to speed up evaluation of multiple queries.

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
   OPENAI_API_KEY=your_openai_api_key_here
   LANGFLOW_URL=https://climate-pal.namastex.ai/api/v1/run/nasa-dataset-selector-02-1-1-1
   LANGFLOW_API_KEY=your_langflow_api_key_here
   ```

### Preparing Evaluation Data

Create a CSV file at `data/evaluation_queries.csv` with the following columns:
- `query`: The query to test
- `scenario`: The expected scenario/experiment (e.g., "ssp119")
- `variable`: The expected variable (e.g., "tas")
- `year_range`: The expected year range (e.g., "2015-2050")
- `file_path`: The expected file path URL

You can use the provided sample file `data/evaluation_queries_sample.csv` as a template:

```csv
,query,scenario,variable,year_range,file_path
0,Show me near-surface air temperature data for a ssp119 CO2 emissions scenario for 201501-205012,ssp119,tas,2015-2050,https://portal.nccs.nasa.gov/datashare/giss_cmip6/CMIP6/ScenarioMIP/NASA-GISS/GISS-E2-1-G/ssp119/r3i1p1f2/Amon/tas/gn/v20200115/tas_Amon_GISS-E2-1-G_ssp119_r3i1p1f2_gn_201501-205012.nc
...
```

### Running the Evaluation

Basic usage:
```bash
python evaluate.py
```

This will process all queries in the CSV file.

### Command Line Options

The script supports several command line options:

- `--samples N`: Process only the first N samples (default: 0, which means all samples)
- `--output-dir DIR`: Directory to store evaluation results (default: "evaluation_results")
- `--debug`: Enable debug mode to save raw API responses
- `--max-retries N`: Maximum number of retries for failed API requests (default: 3)
- `--retry-delay N`: Delay in seconds between retry attempts (default: 5)
- `--max-workers N`: Maximum number of parallel workers (default: 4)

Examples:

```bash
# Process only 10 samples
python evaluate.py --samples 10

# Use 8 parallel workers
python evaluate.py --max-workers 8

# Enable debug mode to save API responses
python evaluate.py --debug

# Combine multiple options
python evaluate.py --samples 5 --debug --max-workers 2 --output-dir custom_results
```

### Understanding the Results

The script will generate a CSV file with the evaluation results in the specified output directory. The file includes:

- Expected vs. predicted values for scenario, variable, year range, and file path
- Match indicators for each field
- Success/error information

The script also calculates and displays summary statistics:
- Total number of queries processed
- Accuracy percentages for scenario, variable, year range, and file path

If debug mode is enabled, the raw API responses will be saved in the `debug` subdirectory of the output directory.

### Troubleshooting

Here are some common issues and their solutions:

1. **API Connection Issues**:
   - Verify your API keys in the `.env` file
   - Check that the LangFlow service is running
   - Increase the `--retry-delay` if you're experiencing rate limiting

2. **Missing Dependencies**:
   - Make sure all required packages are installed: `pip install -r requirements.txt`
   - If you encounter "Unable to import 'requests'" error, install it separately: `pip install requests`

3. **Performance Issues**:
   - Adjust the `--max-workers` parameter based on your system's capabilities
   - For large datasets, use the `--samples` parameter to test with a smaller subset first

4. **Incorrect Results**:
   - Enable `--debug` mode to inspect the raw API responses
   - Check the format of your evaluation_queries.csv file
   - Verify that the expected values match the format returned by the API

5. **File Path Matching Issues**:
   - Ensure that the file paths in your CSV use the same format as the API responses
   - Some APIs may return relative paths while others return absolute URLs