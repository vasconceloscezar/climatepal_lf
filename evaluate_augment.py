import os
import json
import time
import uuid
import argparse
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
import signal
import re
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Generate a unique ID for this run
UNIQUE_RUN_ID = str(uuid.uuid4())

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY")
LANGFLOW_URL = os.getenv(
    "LANGFLOW_URL",
    "https://climate-pal.namastex.ai/api/v1/run/nasa-dataset-selector-02-1-1-1",
)

# Global flag to indicate if the process is being interrupted
interrupted = False


def signal_handler(sig, frame):
    """Handle keyboard interrupt signal."""
    global interrupted
    if not interrupted:
        print("\nKeyboard interrupt received. Gracefully shutting down...")
        interrupted = True
    else:
        print("\nSecond interrupt received. Forcing exit...")
        sys.exit(1)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def extract_data_from_response(response_json):
    """Extract relevant data from the API response."""
    try:
        # Get the response content
        # First, check if the response has the expected structure
        if "outputs" in response_json and isinstance(response_json["outputs"], list):
            # Extract all the text from the response for easier searching
            all_text = json.dumps(response_json)

            # Initialize variables
            extracted_variable = ""

            # Look for the variable in the response
            for output in response_json["outputs"]:
                # Look for variable in response results
                if "results" in output and "response" in output["results"]:
                    content = output["results"]["response"].get("text", "")
                    if content:
                        print(f"Found variable in response: {content}")
                        # Don't return yet, continue searching for other data
                        extracted_variable = content
                        break

                # Check for other possible locations of the variable
                if "artifacts" in output and "response" in output["artifacts"]:
                    if "text" in output["artifacts"]["response"]:
                        content = output["artifacts"]["response"]["text"]
                        if content:
                            print(f"Found variable in artifacts: {content}")
                            extracted_variable = content
                            break
                    elif (
                        isinstance(output["artifacts"]["response"], dict)
                        and "raw" in output["artifacts"]["response"]
                    ):
                        content = output["artifacts"]["response"]["raw"].get("text", "")
                        if content:
                            print(f"Found variable in raw: {content}")
                            extracted_variable = content
                            break

                # Try to find agent-related information
                if (
                    "component_display_name" in output
                    and output["component_display_name"] == "VariableSelector"
                ):
                    if "artifacts" in output and "response" in output["artifacts"]:
                        agent_data = output["artifacts"]["response"].get("raw", {})
                        if isinstance(agent_data, dict):
                            # Look for model information in the response
                            if "content_blocks" in agent_data:
                                for block in agent_data.get("content_blocks", []):
                                    for content in block.get("contents", []):
                                        if (
                                            content.get("type") == "tool_use"
                                            and "output" in content
                                        ):
                                            # May contain variable information
                                            for item in content.get("output", []):
                                                if "variable" in item:
                                                    extracted_variable = item[
                                                        "variable"
                                                    ]
                                                    print(
                                                        f"Found variable in agent output: {extracted_variable}"
                                                    )

            # Look for date range in the response structure
            date_range = ""
            # Search for date range pattern like "201501-205012" in all text
            date_range_match = re.search(r"(\d{6})-(\d{6})", all_text)
            if date_range_match:
                full_date_range = date_range_match.group(0)
                print(f"Found date range: {full_date_range}")
                date_range = full_date_range

                # Extract start and end years from the range
                start_year = (
                    date_range_match.group(1)[:4] if date_range_match.group(1) else ""
                )
                end_year = (
                    date_range_match.group(2)[:4] if date_range_match.group(2) else ""
                )
            else:
                # Try to find other formats like "2015-2050"
                alt_date_match = re.search(r"(\d{4})-(\d{4})", all_text)
                if alt_date_match:
                    start_year = alt_date_match.group(1)
                    end_year = alt_date_match.group(2)
                    print(
                        f"Found date range in alternate format: {start_year}-{end_year}"
                    )
                else:
                    start_year = ""
                    end_year = ""

            # Extract variable using regex patterns
            variable_match = re.search(
                r'"text":\s*"(tas|pr|tos|sfcWind|huss|psl|ts|ua|va|zg|ta)"', all_text
            )
            if variable_match:
                variable = variable_match.group(1)
                print(f"Found variable using regex: {variable}")
            else:
                # Look for variable in an "Agent Steps" section with tool output
                agent_steps_match = re.search(
                    r'"title":\s*"Agent Steps".*?"variable":\s*"(tas|pr|tos|sfcWind|huss|psl|ts|ua|va|zg|ta)"',
                    all_text,
                    re.DOTALL,
                )
                if agent_steps_match:
                    variable = agent_steps_match.group(1)
                    print(f"Found variable in Agent Steps: {variable}")
                else:
                    variable = extracted_variable

            # Extract temporal resolution
            temporal_resolution = ""
            resolution_match = re.search(r'"text":\s*"(mon|day|hr|fx)"', all_text)
            if resolution_match:
                temporal_resolution = resolution_match.group(1)
                print(f"Found temporal resolution: {temporal_resolution}")

            # Extract experiment and MIP
            experiment = ""
            exp_match = re.search(
                r'"text":\s*"(ssp119|ssp126|ssp245|ssp370|ssp434|ssp460|ssp534-over|ssp585|historical|piControl|1pctCO2|abrupt-4xCO2|amip)"',
                all_text,
            )
            if exp_match:
                experiment = exp_match.group(1)
                print(f"Found experiment: {experiment}")

            mip = ""
            mip_match = re.search(r'"text":\s*"(CMIP|ScenarioMIP)"', all_text)
            if mip_match:
                mip = mip_match.group(1)
                print(f"Found MIP: {mip}")

            return {
                "variable": variable,
                "temporal_resolution": temporal_resolution,
                "start_year": start_year,
                "end_year": end_year,
                "mip": mip,
                "experiment": experiment,
            }

        # If we couldn't find structured data, search the entire response string
        response_str = json.dumps(response_json)

        # Extract variable
        variable = ""
        variable_match = re.search(
            r'"text":\s*"(tas|pr|tos|sfcWind|huss|psl|ts|ua|va|zg|ta)"', response_str
        )
        if variable_match:
            variable = variable_match.group(1)
            print(f"Found variable using regex: {variable}")

        # Extract date range
        start_year = ""
        end_year = ""
        date_range_match = re.search(r"(\d{6})-(\d{6})", response_str)
        if date_range_match:
            # Extract year from YYYYMM format
            start_year = date_range_match.group(1)[:4]
            end_year = date_range_match.group(2)[:4]
            print(f"Found date range: {start_year}-{end_year}")
        else:
            # Try to find other formats like "2015-2050"
            alt_date_match = re.search(r"(\d{4})-(\d{4})", response_str)
            if alt_date_match:
                start_year = alt_date_match.group(1)
                end_year = alt_date_match.group(2)
                print(f"Found date range in alternate format: {start_year}-{end_year}")

        # Extract temporal resolution
        temporal_resolution = ""
        resolution_match = re.search(r'"text":\s*"(mon|day|hr|fx)"', response_str)
        if resolution_match:
            temporal_resolution = resolution_match.group(1)
            print(f"Found temporal resolution: {temporal_resolution}")

        # Extract experiment and MIP
        experiment = ""
        exp_match = re.search(
            r'"text":\s*"(ssp119|ssp126|ssp245|ssp370|ssp434|ssp460|ssp534-over|ssp585|historical|piControl|1pctCO2|abrupt-4xCO2|amip)"',
            response_str,
        )
        if exp_match:
            experiment = exp_match.group(1)
            print(f"Found experiment: {experiment}")

        mip = ""
        mip_match = re.search(r'"text":\s*"(CMIP|ScenarioMIP)"', response_str)
        if mip_match:
            mip = mip_match.group(1)
            print(f"Found MIP: {mip}")

        return {
            "variable": variable,
            "temporal_resolution": temporal_resolution,
            "start_year": start_year,
            "end_year": end_year,
            "mip": mip,
            "experiment": experiment,
        }

    except Exception as e:
        print(f"Error extracting data from response: {str(e)}")
        traceback.print_exc()
        return {
            "variable": "",
            "temporal_resolution": "",
            "start_year": "",
            "end_year": "",
            "mip": "",
            "experiment": "",
        }


def process_query(row, index, debug_dir=None, max_retries=3, retry_delay=5):
    """Process a single query and evaluate the result."""
    # Access the global interrupted flag
    global interrupted

    print(f"\nProcessing query {index}: {row['query']}")

    # Make sure we have the auth key
    LANGFLOW_API_KEY = os.environ.get("LANGFLOW_API_KEY")
    LANGFLOW_URL = os.environ.get("LANGFLOW_URL")

    if not LANGFLOW_API_KEY or not LANGFLOW_URL:
        print(
            "Error: LANGFLOW_API_KEY or LANGFLOW_URL environment variables are not set."
        )
        return {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_variable": row["variable"],
            "expected_temporal_resolution": (
                ""
                if pd.isna(row.get("temporal resolution", ""))
                else row.get("temporal resolution", "")
            ),
            "expected_start_year": (
                ""
                if pd.isna(row.get("start year", 0))
                else (
                    str(int(row.get("start year", 0)))
                    if isinstance(row.get("start year", 0), (int, float))
                    and not pd.isna(row.get("start year", 0))
                    else row.get("start year", 0)
                )
            ),
            "expected_end_year": (
                ""
                if pd.isna(row.get("end year", 0))
                else (
                    str(int(row.get("end year", 0)))
                    if isinstance(row.get("end year", 0), (int, float))
                    and not pd.isna(row.get("end year", 0))
                    else row.get("end year", 0)
                )
            ),
            "expected_mip": "" if pd.isna(row.get("MIP", "")) else row.get("MIP", ""),
            "expected_experiment": (
                "" if pd.isna(row.get("experiment", "")) else row.get("experiment", "")
            ),
            "predicted_variable": "",
            "predicted_temporal_resolution": "",
            "predicted_start_year": "",
            "predicted_end_year": "",
            "predicted_mip": "",
            "predicted_experiment": "",
            "variable_match": False,
            "temporal_resolution_match": False,
            "start_year_match": False,
            "end_year_match": False,
            "mip_match": False,
            "experiment_match": False,
            "success": False,
            "error": "API key or URL not set",
        }

    # Prepare the API request payload
    payload = {
        "session_id": str(index) + "-" + datetime.now().strftime("%Y%m%d_%H%M%S"),
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
                # Check if the current thread has been interrupted
                if threading.current_thread().daemon or interrupted:
                    # This is a way to check if the thread should exit
                    # If the thread is marked as daemon, it means it should terminate
                    return {
                        "query_id": row.get("", index),
                        "query": row["query"],
                        "expected_variable": row["variable"],
                        "expected_temporal_resolution": (
                            ""
                            if pd.isna(row.get("temporal resolution", ""))
                            else row.get("temporal resolution", "")
                        ),
                        "expected_start_year": (
                            ""
                            if pd.isna(row.get("start year", 0))
                            else (
                                str(int(row.get("start year", 0)))
                                if isinstance(row.get("start year", 0), (int, float))
                                and not pd.isna(row.get("start year", 0))
                                else row.get("start year", 0)
                            )
                        ),
                        "expected_end_year": (
                            ""
                            if pd.isna(row.get("end year", 0))
                            else (
                                str(int(row.get("end year", 0)))
                                if isinstance(row.get("end year", 0), (int, float))
                                and not pd.isna(row.get("end year", 0))
                                else row.get("end year", 0)
                            )
                        ),
                        "expected_mip": (
                            "" if pd.isna(row.get("MIP", "")) else row.get("MIP", "")
                        ),
                        "expected_experiment": (
                            ""
                            if pd.isna(row.get("experiment", ""))
                            else row.get("experiment", "")
                        ),
                        "predicted_variable": "INTERRUPTED",
                        "predicted_temporal_resolution": "",
                        "predicted_start_year": "",
                        "predicted_end_year": "",
                        "predicted_mip": "",
                        "predicted_experiment": "",
                        "variable_match": False,
                        "temporal_resolution_match": False,
                        "start_year_match": False,
                        "end_year_match": False,
                        "mip_match": False,
                        "experiment_match": False,
                        "success": False,
                        "error": "Task interrupted",
                    }

                response = requests.post(LANGFLOW_URL, json=payload, headers=headers)
                response.raise_for_status()  # Raise an error if the request failed
                break
            except requests.exceptions.RequestException as e:
                retry_count += 1
                print(f"API request failed (attempt {retry_count}): {str(e)}")
                if retry_count <= max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Maximum retries reached. Giving up.")
                    return {
                        "query_id": row.get("", index),
                        "query": row["query"],
                        "expected_variable": row["variable"],
                        "expected_temporal_resolution": "",
                        "expected_start_year": "",
                        "expected_end_year": "",
                        "expected_mip": row.get("MIP", ""),
                        "expected_experiment": row.get("experiment", ""),
                        "predicted_variable": "",
                        "predicted_temporal_resolution": "",
                        "predicted_start_year": "",
                        "predicted_end_year": "",
                        "predicted_mip": "",
                        "predicted_experiment": "",
                        "variable_match": False,
                        "temporal_resolution_match": False,
                        "start_year_match": False,
                        "end_year_match": False,
                        "mip_match": False,
                        "experiment_match": False,
                        "success": False,
                        "error": str(e),
                    }

        # Parse the response
        response_json = response.json()

        # Save the debug information if requested
        if debug_dir:
            debug_file = os.path.join(debug_dir, f"query_{index}_response.json")
            with open(debug_file, "w") as f:
                json.dump(response_json, f, indent=2)

        # Extract data from the response
        extracted_data = extract_data_from_response(response_json)

        # Convert start_year and end_year to int for comparison, if they are not empty
        expected_start_year = row.get("start year", 0)
        expected_end_year = row.get("end year", 0)

        # Handle NaN values and convert to string for comparison
        if isinstance(expected_start_year, (int, float)):
            # Check if it's NaN before converting to int
            if pd.isna(expected_start_year):
                expected_start_year = ""
            else:
                expected_start_year = str(int(expected_start_year))

        if isinstance(expected_end_year, (int, float)):
            # Check if it's NaN before converting to int
            if pd.isna(expected_end_year):
                expected_end_year = ""
            else:
                expected_end_year = str(int(expected_end_year))

        # Normalize predicted years if not empty
        predicted_start_year = extracted_data["start_year"]
        predicted_end_year = extracted_data["end_year"]

        # Compare the expected and predicted values
        variable_match = (
            extracted_data["variable"].lower() == row["variable"].lower()
            if extracted_data["variable"] and row["variable"]
            else False
        )

        # For temporal resolution, MIP, and experiment, check if empty strings match
        expected_temporal_resolution = row.get("temporal resolution", "")
        if pd.isna(expected_temporal_resolution):
            expected_temporal_resolution = ""

        temporal_resolution_match = (
            extracted_data["temporal_resolution"].lower()
            == expected_temporal_resolution.lower()
            if extracted_data["temporal_resolution"] or expected_temporal_resolution
            else True  # Match if both are empty
        )

        # For year comparison, treat empty or zero values as a special case
        start_year_match = False
        if (
            not expected_start_year
            or expected_start_year == "0"
            or expected_start_year == 0
        ):
            start_year_match = not predicted_start_year  # Match if both are empty/zero
        else:
            start_year_match = predicted_start_year == expected_start_year

        end_year_match = False
        if not expected_end_year or expected_end_year == "0" or expected_end_year == 0:
            end_year_match = not predicted_end_year  # Match if both are empty/zero
        else:
            end_year_match = predicted_end_year == expected_end_year

        # For MIP and experiment
        expected_mip = row.get("MIP", "")
        if pd.isna(expected_mip):
            expected_mip = ""

        mip_match = (
            extracted_data["mip"].lower() == expected_mip.lower()
            if extracted_data["mip"] or expected_mip
            else True  # Match if both are empty
        )

        expected_experiment = row.get("experiment", "")
        if pd.isna(expected_experiment):
            expected_experiment = ""

        experiment_match = (
            extracted_data["experiment"].lower() == expected_experiment.lower()
            if extracted_data["experiment"] or expected_experiment
            else True  # Match if both are empty
        )

        # Compute success as True if the variable was found
        success = True if extracted_data["variable"] else False

        result = {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_variable": row["variable"],
            "expected_temporal_resolution": expected_temporal_resolution,
            "expected_start_year": expected_start_year,
            "expected_end_year": expected_end_year,
            "expected_mip": expected_mip,
            "expected_experiment": expected_experiment,
            "predicted_variable": extracted_data["variable"],
            "predicted_temporal_resolution": extracted_data["temporal_resolution"],
            "predicted_start_year": predicted_start_year,
            "predicted_end_year": predicted_end_year,
            "predicted_mip": extracted_data["mip"],
            "predicted_experiment": extracted_data["experiment"],
            "variable_match": variable_match,
            "temporal_resolution_match": temporal_resolution_match,
            "start_year_match": start_year_match,
            "end_year_match": end_year_match,
            "mip_match": mip_match,
            "experiment_match": experiment_match,
            "success": success,
            "error": "",
        }

        return result

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        traceback.print_exc()
        return {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_variable": row["variable"],
            "expected_temporal_resolution": (
                ""
                if pd.isna(row.get("temporal resolution", ""))
                else row.get("temporal resolution", "")
            ),
            "expected_start_year": (
                ""
                if pd.isna(row.get("start year", 0))
                else (
                    str(int(row.get("start year", 0)))
                    if isinstance(row.get("start year", 0), (int, float))
                    and not pd.isna(row.get("start year", 0))
                    else row.get("start year", 0)
                )
            ),
            "expected_end_year": (
                ""
                if pd.isna(row.get("end year", 0))
                else (
                    str(int(row.get("end year", 0)))
                    if isinstance(row.get("end year", 0), (int, float))
                    and not pd.isna(row.get("end year", 0))
                    else row.get("end year", 0)
                )
            ),
            "expected_mip": "" if pd.isna(row.get("MIP", "")) else row.get("MIP", ""),
            "expected_experiment": (
                "" if pd.isna(row.get("experiment", "")) else row.get("experiment", "")
            ),
            "predicted_variable": "",
            "predicted_temporal_resolution": "",
            "predicted_start_year": "",
            "predicted_end_year": "",
            "predicted_mip": "",
            "predicted_experiment": "",
            "variable_match": False,
            "temporal_resolution_match": False,
            "start_year_match": False,
            "end_year_match": False,
            "mip_match": False,
            "experiment_match": False,
            "success": False,
            "error": str(e),
        }


def run_evaluation(
    df,
    debug=False,
    max_retries=3,
    retry_delay=5,
    max_workers=None,
    output_dir="evaluation_results",
    samples=0,
    continue_from=None,
):
    """Run the evaluation on the DataFrame."""
    # Access the global interrupted flag
    global interrupted

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    output_file = os.path.join(
        output_dir, f"augment_evaluation_results_{timestamp}.csv"
    )

    # If debug is enabled, create debug directory
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug", f"evaluation_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug information will be saved to {debug_dir}")

    # Create a CSV file for real-time results
    results_file = os.path.join(output_dir, f"results_{timestamp}.csv")

    # Initialize results list
    results = []

    # Track indices to skip if continuing from previous run
    indices_to_skip = set()

    # If continuing from a previous run, load the results and determine which indices to skip
    if continue_from and os.path.exists(continue_from):
        try:
            previous_results = pd.read_csv(continue_from)
            # Get the query_id values that have already been processed
            processed_ids = previous_results["query_id"].tolist()
            indices_to_skip = set(processed_ids)
            print(f"Continuing from previous run: {continue_from}")
            print(f"Skipping {len(indices_to_skip)} already processed queries")
        except Exception as e:
            print(f"Error reading previous results file: {str(e)}")
            print("Will start evaluation from the beginning")

    # Limit the number of samples if requested
    if samples > 0:
        if samples >= len(df):
            print(
                f"Sample size {samples} is >= dataset size {len(df)}. Using entire dataset."
            )
        else:
            # Randomly select indices instead of taking first N samples
            random_indices = np.random.choice(len(df), samples, replace=False)
            df = df.iloc[random_indices].reset_index(drop=True)
            print(f"Randomly selected {samples} samples from the dataset.")

    # Set the maximum number of workers
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 4)  # Default to min(4, cpu_count)

    try:
        # Create a header for the results file
        columns = [
            "query_id",
            "query",
            "expected_variable",
            "expected_temporal_resolution",
            "expected_start_year",
            "expected_end_year",
            "expected_mip",
            "expected_experiment",
            "predicted_variable",
            "predicted_temporal_resolution",
            "predicted_start_year",
            "predicted_end_year",
            "predicted_mip",
            "predicted_experiment",
            "variable_match",
            "temporal_resolution_match",
            "start_year_match",
            "end_year_match",
            "mip_match",
            "experiment_match",
            "success",
            "error",
        ]
        # Create the results file with headers
        with open(results_file, "w", newline="", encoding="utf-8") as f:
            pd.DataFrame(columns=columns).to_csv(f, index=False)
            print(f"Created results file at {results_file}")

        # Process queries in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Initialize progress bar
            total_queries = len(df)
            progress_bar = tqdm(total=total_queries, desc="Processing queries")

            # Submit all tasks to the executor
            futures = []
            for i, (_, row) in enumerate(df.iterrows()):
                # Check if process is being interrupted
                if interrupted:
                    print("Skipping remaining queries due to interrupt...")
                    break

                # Skip if this index was already processed
                if i in indices_to_skip:
                    progress_bar.update(1)
                    continue

                # Submit the task
                future = executor.submit(
                    process_query, row, i, debug_dir, max_retries, retry_delay
                )
                futures.append((i, future))

            # Process completed tasks and update progress
            try:
                # Create a dictionary to map futures to their indices
                future_to_index = {future: idx for idx, future in futures}

                # Process futures as they complete
                for future in as_completed([f for _, f in futures]):
                    if interrupted:
                        # Mark remaining futures as cancelled if interrupted
                        for _, f in futures:
                            if not f.done():
                                f.cancel()
                        print("Cancelling pending tasks...")
                        break

                    # Get the index associated with this future
                    idx = future_to_index[future]

                    result = future.result()
                    results.append(result)

                    # Append result to CSV file
                    with open(results_file, "a", newline="", encoding="utf-8") as f:
                        pd.DataFrame([result]).to_csv(f, index=False, header=False)

                    # Update progress bar
                    progress_bar.update(1)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Cancelling remaining tasks...")
                for _, f in futures:
                    if not f.done():
                        f.cancel()
                interrupted = True
            finally:
                progress_bar.close()

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # If we need to merge with previous results
        if continue_from and os.path.exists(continue_from):
            try:
                previous_results = pd.read_csv(continue_from)
                # Combine previous and new results
                combined_df = pd.concat(
                    [previous_results, results_df], ignore_index=True
                )
                # Save combined results
                combined_df.to_csv(output_file, index=False)
                print(f"Combined results saved to {output_file}")

                # Update results_df to the combined one for metrics calculation
                results_df = combined_df
            except Exception as e:
                print(f"Error merging with previous results: {str(e)}")
                results_df.to_csv(output_file, index=False)
                print(f"New results only saved to {output_file}")
        else:
            # Save final results
            results_df.to_csv(output_file, index=False)
            print(f"Evaluation results saved to {output_file}")

        # Calculate and print metrics
        print("\nEvaluation Metrics:")
        print(f"Total Queries: {len(results_df)}")

        # Count successful queries
        successful_queries = results_df["success"].sum()
        print(
            f"Successful Queries: {successful_queries} ({successful_queries/len(results_df)*100:.2f}%)"
        )

        # Only calculate matches for successful queries
        if successful_queries > 0:
            success_mask = results_df["success"] == True

            # Calculate accuracy for each field
            metrics = {
                "variable": (
                    results_df.loc[success_mask, "variable_match"].sum()
                    / successful_queries
                )
                * 100,
                "temporal_resolution": (
                    results_df.loc[success_mask, "temporal_resolution_match"].sum()
                    / successful_queries
                )
                * 100,
                "start_year": (
                    results_df.loc[success_mask, "start_year_match"].sum()
                    / successful_queries
                )
                * 100,
                "end_year": (
                    results_df.loc[success_mask, "end_year_match"].sum()
                    / successful_queries
                )
                * 100,
                "mip": (
                    results_df.loc[success_mask, "mip_match"].sum() / successful_queries
                )
                * 100,
                "experiment": (
                    results_df.loc[success_mask, "experiment_match"].sum()
                    / successful_queries
                )
                * 100,
            }

            # Calculate overall accuracy (all fields match)
            all_fields_match = (
                results_df.loc[success_mask, "variable_match"]
                & results_df.loc[success_mask, "temporal_resolution_match"]
                & results_df.loc[success_mask, "start_year_match"]
                & results_df.loc[success_mask, "end_year_match"]
                & results_df.loc[success_mask, "mip_match"]
                & results_df.loc[success_mask, "experiment_match"]
            )
            overall_accuracy = (all_fields_match.sum() / successful_queries) * 100
        else:
            metrics = {
                "variable": 0.0,
                "temporal_resolution": 0.0,
                "start_year": 0.0,
                "end_year": 0.0,
                "mip": 0.0,
                "experiment": 0.0,
            }
            overall_accuracy = 0.0

        # Print metrics
        for field, accuracy in metrics.items():
            print(f"{field.replace('_', ' ').title()} Accuracy: {accuracy:.2f}%")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")

        return output_file, results_df

    except Exception as e:
        print(f"Error running evaluation: {str(e)}")
        traceback.print_exc()

        # If interrupted, save partial results
        if results:
            partial_results_df = pd.DataFrame(results)
            partial_output_file = os.path.join(
                output_dir, f"augment_evaluation_results_partial_{timestamp}.csv"
            )
            partial_results_df.to_csv(partial_output_file, index=False)
            print(f"Partial evaluation results saved to {partial_output_file}")

            return partial_output_file, partial_results_df

        return None, None


def main():
    """Main function for the evaluation script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Climate Pal API."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples to process (0 for all)",
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
        help="Maximum number of API retry attempts",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay between retry attempts in seconds",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/augment_aug21.xlsx",
        help="Path to the Excel data file containing evaluation queries",
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Path to a previous results file to continue from",
    )
    args = parser.parse_args()

    try:
        # Load evaluation queries from Excel file
        print(f"Loading data from {args.data_file}")
        if not os.path.exists(args.data_file):
            print(f"Error: Data file '{args.data_file}' not found.")
            sys.exit(1)

        try:
            df = pd.read_excel(args.data_file)
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            sys.exit(1)

        print(f"Loaded {len(df)} queries from {args.data_file}")

        # Run evaluation with the specified parameters
        results_file, results_df = run_evaluation(
            df,
            debug=args.debug,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            samples=args.samples,
            continue_from=args.continue_from,
        )

        print(f"Evaluation completed. Results saved to {results_file}")

        # Exit gracefully
        print("Done.")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
