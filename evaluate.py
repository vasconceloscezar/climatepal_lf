import csv
import os
import json
import argparse
import random
import string
import requests
import time
import re
import concurrent.futures
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFLOW_URL = os.getenv("LANGFLOW_URL")
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it in a .env file or in your environment."
    )

if not LANGFLOW_URL or not LANGFLOW_API_KEY:
    raise ValueError(
        "LANGFLOW_URL or LANGFLOW_API_KEY environment variables are not set. Please set them in a .env file or in your environment."
    )


def extract_url_from_text(text):
    """Extract URL from text using regex pattern."""
    url_pattern = r"https?://\S+"
    urls = re.findall(url_pattern, text)
    return urls[0] if urls else ""


def extract_data_from_response(response_json):
    """Extract relevant data from the API response."""
    result = {"scenario": "", "variable": "", "year_range": "", "file_path": ""}

    # Check if outputs exists in the response
    if "outputs" not in response_json:
        return result

    # Look for messages with specific sender_name values
    for output_group in response_json["outputs"]:
        if "outputs" not in output_group:
            continue

        for output in output_group["outputs"]:
            # Look for extracted_urls which contains the file path
            if "results" in output and "extracted_urls" in output["results"]:
                url_data = output["results"]["extracted_urls"]["raw"]
                if isinstance(url_data, list) and len(url_data) > 0:
                    if "data" in url_data[0] and "url" in url_data[0]["data"]:
                        result["file_path"] = url_data[0]["data"]["url"]

            # Look for messages with specific sender_name values
            if "results" in output and "message" in output["results"]:
                message = output["results"]["message"]
                if isinstance(message, dict) and "sender_name" in message:
                    sender_name = message.get("sender_name", "")
                    text = message.get("text", "")

                    if sender_name == "Experiment":
                        result["scenario"] = text
                    elif sender_name == "Variable":
                        result["variable"] = text
                    elif sender_name == "Date Range":
                        result["year_range"] = text

                    # If we found a URL in the text, save it
                    if "http" in text:
                        url = extract_url_from_text(text)
                        if url and not result["file_path"]:
                            result["file_path"] = url

    # If we still don't have a file path, look for saved_data
    if not result["file_path"]:
        for output_group in response_json["outputs"]:
            if (
                "artifacts" in output_group
                and "saved_data" in output_group["artifacts"]
            ):
                saved_data = output_group["artifacts"]["saved_data"]["raw"]
                if isinstance(saved_data, dict) and "url" in saved_data:
                    result["file_path"] = saved_data["url"]

    return result


def generate_unique_run_id():
    """Generate a unique run ID using timestamp and random string."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{timestamp}_{random_string}"


# Generate a unique run ID at the start of the script
UNIQUE_RUN_ID = generate_unique_run_id()


def process_query(row, index, debug_dir=None, max_retries=3, retry_delay=5):
    """Process a single query and return the results."""
    # Prepare the API request payload
    payload = {
        "session_id": str(index) + "-" + UNIQUE_RUN_ID,
        "output_type": "debug",
        "input_type": "chat",
        "input_value": row["query"],
    }

    headers = {"Content-Type": "application/json", "x-api-key": LANGFLOW_API_KEY}

    try:
        # Make the API request with retries
        response = None
        retry_count = 0

        while retry_count <= max_retries:
            try:
                response = requests.post(
                    LANGFLOW_URL,
                    headers=headers,
                    json=payload,
                    params={"stream": "false"},
                    timeout=60,  # Add a timeout to prevent hanging requests
                )
                response.raise_for_status()
                break  # If successful, exit the retry loop
            except (
                requests.exceptions.RequestException,
                requests.exceptions.HTTPError,
            ) as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise  # Re-raise the exception if we've exceeded max retries
                print(
                    f"API request failed (attempt {retry_count}/{max_retries}): {str(e)}"
                )
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        # Parse the response
        result = response.json()

        # Save the raw response for debugging if debug_dir is provided
        if debug_dir:
            debug_file = os.path.join(debug_dir, f"query_{index}_response.json")
            with open(debug_file, "w") as f:
                json.dump(result, f, indent=2)

        # Extract predictions from the result
        extracted_data = extract_data_from_response(result)

        predicted_scenario = extracted_data["scenario"]
        predicted_variable = extracted_data["variable"]
        predicted_year_range = extracted_data["year_range"]
        predicted_file_path = extracted_data["file_path"]

        # Compare predictions with expected values
        scenario_match = predicted_scenario.lower() == row["scenario"].lower()
        variable_match = predicted_variable.lower() == row["variable"].lower()
        year_range_match = predicted_year_range.lower() == row["year_range"].lower()
        file_path_match = predicted_file_path.lower() == row["file_path"].lower()

        # Return the results
        return {
            "query_id": row.get(
                "", index
            ),  # Use the index column if available, otherwise use the dataframe index
            "query": row["query"],
            "expected_scenario": row["scenario"],
            "expected_variable": row["variable"],
            "expected_year_range": row["year_range"],
            "expected_file_path": row["file_path"],
            "predicted_scenario": predicted_scenario,
            "predicted_variable": predicted_variable,
            "predicted_year_range": predicted_year_range,
            "predicted_file_path": predicted_file_path,
            "scenario_match": scenario_match,
            "variable_match": variable_match,
            "year_range_match": year_range_match,
            "file_path_match": file_path_match,
            "success": True,
            "error": None,
        }

    except Exception as e:
        # Return error information
        return {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_scenario": row["scenario"],
            "expected_variable": row["variable"],
            "expected_year_range": row["year_range"],
            "expected_file_path": row["file_path"],
            "predicted_scenario": f"ERROR: {str(e)}",
            "predicted_variable": "",
            "predicted_year_range": "",
            "predicted_file_path": "",
            "scenario_match": False,
            "variable_match": False,
            "year_range_match": False,
            "file_path_match": False,
            "success": False,
            "error": str(e),
        }


def run_evaluation(
    max_samples=0,
    results_dir="evaluation_results",
    debug=False,
    max_retries=3,
    retry_delay=5,
    max_workers=4,
):
    # Create a directory to store results if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Create a timestamp for the results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.csv")

    # Create a debug directory if debug mode is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(results_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

    # Read the evaluation queries
    df = pd.read_csv("data/evaluation_queries.csv")

    # Limit the number of samples if max_samples is set
    if max_samples > 0:
        df = df.head(max_samples)
        print(f"Processing {max_samples} samples out of {len(df)} total samples")
    else:
        print(f"Processing all {len(df)} samples")

    # Prepare results dataframe
    results_df = pd.DataFrame(
        columns=[
            "query_id",
            "query",
            "expected_scenario",
            "expected_variable",
            "expected_year_range",
            "expected_file_path",
            "predicted_scenario",
            "predicted_variable",
            "predicted_year_range",
            "predicted_file_path",
            "scenario_match",
            "variable_match",
            "year_range_match",
            "file_path_match",
        ]
    )

    # Process queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                process_query, row, index, debug_dir, max_retries, retry_delay
            ): index
            for index, row in df.iterrows()
        }

        # Process results as they complete with a progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(df),
            desc="Processing queries",
        ):
            index = future_to_index[future]
            try:
                result = future.result()
                # Add to results dataframe
                results_df = results_df._append(result, ignore_index=True)

                # Save intermediate results
                results_df.to_csv(results_file, index=False)

                # Print result summary
                if result["success"]:
                    print(
                        f"  Query {index}: Scenario: {'✓' if result['scenario_match'] else '✗'} | "
                        f"Variable: {'✓' if result['variable_match'] else '✗'} | "
                        f"Year Range: {'✓' if result['year_range_match'] else '✗'} | "
                        f"File Path: {'✓' if result['file_path_match'] else '✗'}"
                    )
                else:
                    print(f"  Query {index}: ERROR: {result['error']}")

            except Exception as e:
                print(f"Error processing query {index}: {str(e)}")

    # Calculate and print summary statistics
    total_queries = len(results_df)
    scenario_accuracy = results_df["scenario_match"].mean() * 100
    variable_accuracy = results_df["variable_match"].mean() * 100
    year_range_accuracy = results_df["year_range_match"].mean() * 100
    file_path_accuracy = results_df["file_path_match"].mean() * 100

    print("\nEvaluation Summary:")
    print(f"Total Queries: {total_queries}")
    print(f"Scenario Accuracy: {scenario_accuracy:.2f}%")
    print(f"Variable Accuracy: {variable_accuracy:.2f}%")
    print(f"Year Range Accuracy: {year_range_accuracy:.2f}%")
    print(f"File Path Accuracy: {file_path_accuracy:.2f}%")

    # Save final results
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation on climate data queries"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples to process (0 for all samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to store evaluation results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save raw API responses",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API requests",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay in seconds between retry attempts",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers",
    )
    args = parser.parse_args()

    # Run evaluation with the specified parameters
    run_evaluation(
        max_samples=args.samples,
        results_dir=args.output_dir,
        debug=args.debug,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_workers=args.max_workers,
    )
