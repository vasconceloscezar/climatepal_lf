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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import traceback
import signal
import uuid

# Load environment variables from .env file
load_dotenv()

# Generate a unique ID for this run
UNIQUE_RUN_ID = str(uuid.uuid4())

LANGFLOW_URL = os.getenv("LANGFLOW_URL")
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY")

if not LANGFLOW_URL or not LANGFLOW_API_KEY:
    raise ValueError(
        "LANGFLOW_URL or LANGFLOW_API_KEY environment variables are not set. Please set them in a .env file or in your environment."
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


def extract_url_from_text(text):
    """Extract URL from text using regex pattern."""
    url_pattern = r"https?://\\S+"
    urls = re.findall(url_pattern, text)
    return urls[0] if urls else ""


def extract_data_from_response(response_json):
    """Extract relevant data from the API response."""
    result = {"scenario": "", "variable": "", "year_range": "", "file_path": ""}

    if not isinstance(response_json, dict):
        return result

    outputs_top_level = response_json.get("outputs")
    if not isinstance(outputs_top_level, list) or not outputs_top_level:
        return result

    # Assuming the relevant data is always in the first item of the top-level "outputs" list
    first_output_group = outputs_top_level[0]
    if not isinstance(first_output_group, dict):
        return result

    detailed_outputs = first_output_group.get("outputs")
    if not isinstance(detailed_outputs, list):
        return result

    for output_item in detailed_outputs:
        if not isinstance(output_item, dict):
            continue

        results_field = output_item.get("results")
        if not isinstance(results_field, dict):
            continue

        message_data = results_field.get("message")
        if not isinstance(message_data, dict):
            continue

        sender_name = message_data.get("sender_name", "")
        text = message_data.get("text", "")

        if sender_name == "Experiment":
            result["scenario"] = text
        elif sender_name == "Variable":
            result["variable"] = text
        elif sender_name == "Date Range":
            result["year_range"] = text
        elif sender_name == "AI":
            # Try to parse the structured URL from the AI message
            url_match_ai = re.search(r"- URL: (https?://\\S+)", text)
            if url_match_ai:
                result["file_path"] = url_match_ai.group(1)
            elif "- URL: No Match" in text and not result["file_path"]:
                result["file_path"] = ""  # Explicitly no match from AI

        # General fallback: if a URL is in any message\\'s text and file_path is not yet set
        if not result["file_path"] and "http" in text:
            url_from_text = extract_url_from_text(text)
            if url_from_text:
                result["file_path"] = url_from_text

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
    global interrupted

    # Check if the process has been interrupted
    if interrupted:
        return {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_scenario": row["scenario"],
            "expected_variable": row["variable"],
            "expected_year_range": row["year_range"],
            "expected_file_path": row["file_path"],
            "predicted_scenario": "INTERRUPTED",
            "predicted_variable": "",
            "predicted_year_range": "",
            "predicted_file_path": "",
            "scenario_match": False,
            "variable_match": False,
            "year_range_match": False,
            "file_path_match": False,
            "success": False,
            "error": "Task interrupted",
        }

    # Prepare the API request payload
    payload = {
        "session_id": str(index) + "-" + UNIQUE_RUN_ID,
        "output_type": "chat",
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
                        "expected_scenario": row["scenario"],
                        "expected_variable": row["variable"],
                        "expected_year_range": row["year_range"],
                        "expected_file_path": row["file_path"],
                        "predicted_scenario": "INTERRUPTED",
                        "predicted_variable": "",
                        "predicted_year_range": "",
                        "predicted_file_path": "",
                        "scenario_match": False,
                        "variable_match": False,
                        "year_range_match": False,
                        "file_path_match": False,
                        "success": False,
                        "error": "Task interrupted",
                    }

                response = requests.post(
                    LANGFLOW_URL,
                    headers=headers,
                    json=payload,
                    params={"stream": "false"},
                    timeout=600,  # Add a timeout to prevent hanging requests
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

    except KeyboardInterrupt:
        # Handle keyboard interrupt within the thread
        return {
            "query_id": row.get("", index),
            "query": row["query"],
            "expected_scenario": row["scenario"],
            "expected_variable": row["variable"],
            "expected_year_range": row["year_range"],
            "expected_file_path": row["file_path"],
            "predicted_scenario": "INTERRUPTED",
            "predicted_variable": "",
            "predicted_year_range": "",
            "predicted_file_path": "",
            "scenario_match": False,
            "variable_match": False,
            "year_range_match": False,
            "file_path_match": False,
            "success": False,
            "error": "Task interrupted",
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


def run_evaluation(df, debug=False, max_retries=3, retry_delay=5, max_workers=None):
    """Run the evaluation on the given dataframe."""
    global interrupted

    # Create a debug directory if debug mode is enabled
    debug_dir = None
    if debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = os.path.join("debug", f"evaluation_{timestamp}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug information will be saved to: {debug_dir}")

    # Initialize results list
    results = []
    futures = []

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Submit tasks to the executor
            for index, row in df.iterrows():
                future = executor.submit(
                    process_query, row, index, debug_dir, max_retries, retry_delay
                )
                futures.append(future)

            # Create a progress bar
            with tqdm(total=len(futures), desc="Processing queries") as pbar:
                # Process completed tasks as they finish
                for future in as_completed(futures):
                    if interrupted:
                        break

                    result = future.result()
                    results.append(result)
                    pbar.update(1)

        except KeyboardInterrupt:
            interrupted = True
            print("\nKeyboard interrupt detected. Cancelling pending tasks...")

            # Mark all running threads as daemon to signal they should terminate
            for thread in threading.enumerate():
                if thread != threading.current_thread():
                    thread.daemon = True

            # Cancel all pending futures
            for future in futures:
                if not future.done():
                    future.cancel()

            # Collect results from completed tasks
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        result = future.result()
                        if result not in results:  # Avoid duplicates
                            results.append(result)
                    except Exception as e:
                        print(f"Error retrieving result: {str(e)}")

            print(
                f"Evaluation interrupted. Processed {len(results)} out of {len(futures)} queries."
            )

    # Convert results to a dataframe
    results_df = pd.DataFrame(results)

    # Calculate accuracy metrics if we have results
    if not results_df.empty:
        # Calculate accuracy metrics
        total_queries = len(results_df)
        successful_queries = results_df["success"].sum()
        failed_queries = total_queries - successful_queries

        # Only calculate matches for successful queries
        if successful_queries > 0:
            success_mask = results_df["success"] == True
            scenario_accuracy = (
                results_df.loc[success_mask, "scenario_match"].mean() * 100
            )
            variable_accuracy = (
                results_df.loc[success_mask, "variable_match"].mean() * 100
            )
            year_range_accuracy = (
                results_df.loc[success_mask, "year_range_match"].mean() * 100
            )
            file_path_accuracy = (
                results_df.loc[success_mask, "file_path_match"].mean() * 100
            )
            overall_accuracy = (
                results_df.loc[
                    success_mask,
                    [
                        "scenario_match",
                        "variable_match",
                        "year_range_match",
                        "file_path_match",
                    ],
                ]
                .all(axis=1)
                .mean()
                * 100
            )
        else:
            scenario_accuracy = variable_accuracy = year_range_accuracy = (
                file_path_accuracy
            ) = overall_accuracy = 0.0

        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total Queries: {total_queries}")
        print(f"Successful Queries: {successful_queries}")
        print(f"Failed Queries: {failed_queries}")

        if successful_queries > 0:
            print("\nAccuracy Metrics (for successful queries):")
            print(f"Scenario Accuracy: {scenario_accuracy:.2f}%")
            print(f"Variable Accuracy: {variable_accuracy:.2f}%")
            print(f"Year Range Accuracy: {year_range_accuracy:.2f}%")
            print(f"File Path Accuracy: {file_path_accuracy:.2f}%")
            print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return results_df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate queries against expected outcomes"
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
        "--debug", action="store_true", help="Enable debug mode to save API responses"
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
        "--max-workers", type=int, default=4, help="Maximum number of parallel workers"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print("Starting evaluation...")

        # Read the evaluation queries
        try:
            df = pd.read_csv("data/evaluation_queries.csv")
        except FileNotFoundError:
            print(
                "Error: data/evaluation_queries.csv not found. Please make sure the file exists."
            )
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(
                "Error: data/evaluation_queries.csv is empty. Please add queries to evaluate."
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error reading evaluation queries: {str(e)}")
            sys.exit(1)

        # Limit the number of samples if max_samples is set
        if args.samples > 0:
            df = df.head(args.samples)
            print(f"Processing {args.samples} samples out of {len(df)} total samples")
        else:
            print(f"Processing all {len(df)} samples")

        # Run evaluation with the specified parameters
        results_df = run_evaluation(
            df,
            debug=args.debug,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_workers=args.max_workers,
        )

        # Save results to file if we have any and haven't been interrupted
        if not results_df.empty and not interrupted:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                args.output_dir, f"evaluation_results_{timestamp}.csv"
            )
            results_df.to_csv(results_file, index=False)
            print(f"\nResults saved to {results_file}")
            print("Evaluation completed successfully.")
        elif not results_df.empty and interrupted:
            # Save partial results if interrupted
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                args.output_dir, f"evaluation_results_partial_{timestamp}.csv"
            )
            results_df.to_csv(results_file, index=False)
            print(f"\nPartial results saved to {results_file}")
            print("Evaluation was interrupted but partial results were saved.")

    except KeyboardInterrupt:
        # This should be caught by the signal handler, but just in case
        if not interrupted:
            interrupted = True
            print("\nEvaluation interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        traceback.print_exc()
    finally:
        print("Evaluation script finished.")
