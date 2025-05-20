import os
import pandas as pd
import argparse
import glob
from datetime import datetime
import sys

# Import the necessary function from evaluate.py
sys.path.append(".")
from evaluate import process_query, run_evaluation


def get_latest_evaluation_file():
    """Get the most recent evaluation results file."""
    evaluation_files = glob.glob("evaluation_results/evaluation_results_*.csv")
    if not evaluation_files:
        print("No evaluation result files found.")
        return None

    # Sort by modification time (most recent first)
    latest_file = max(evaluation_files, key=os.path.getmtime)
    return latest_file


def retry_failed_queries(
    input_file=None,
    output_dir="evaluation_results",
    debug=False,
    max_retries=3,
    retry_delay=5,
    max_workers=4,
):
    """Retry failed queries from an evaluation results file."""

    # If no input file is specified, use the most recent one
    if not input_file:
        input_file = get_latest_evaluation_file()
        if not input_file:
            return

    print(f"Loading failed queries from: {input_file}")

    # Read the evaluation results
    df = pd.read_csv(input_file)

    # Identify failed queries
    failed_queries = df[df["success"] == False].copy()

    if failed_queries.empty:
        print("No failed queries found to retry.")
        return

    print(f"Found {len(failed_queries)} failed queries to retry.")

    # Prepare the dataset for re-evaluation
    # Ensure we have the columns needed for process_query function
    retry_df = pd.DataFrame(
        {
            "query": failed_queries["query"],
            "scenario": failed_queries["expected_scenario"],
            "variable": failed_queries["expected_variable"],
            "year_range": failed_queries["expected_year_range"],
            "file_path": failed_queries["expected_file_path"],
        }
    )

    # Run evaluation on the failed queries
    print("Retrying failed queries...")
    results_df = run_evaluation(
        retry_df,
        debug=debug,
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_workers=max_workers,
    )

    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the results
    output_file = os.path.join(output_dir, f"retry_results_{timestamp}.csv")
    results_df.to_csv(output_file, index=False)

    print(f"Retry results saved to {output_file}")

    # Merge with original file to create a complete evaluation file
    original_df = df[df["success"] == True]
    combined_df = pd.concat([original_df, results_df])

    # Save the combined results
    combined_file = os.path.join(output_dir, f"combined_results_{timestamp}.csv")
    combined_df.to_csv(combined_file, index=False)

    print(f"Combined results saved to {combined_file}")

    # Print summary statistics
    if not results_df.empty:
        total_retried = len(results_df)
        successful_retries = results_df["success"].sum()
        failed_retries = total_retried - successful_retries

        print("\nRetry Summary:")
        print(f"Total Retried Queries: {total_retried}")
        print(f"Successful Retries: {successful_retries}")
        print(f"Failed Retries: {failed_retries}")

        if successful_retries > 0:
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

            print("\nAccuracy Metrics (for successful retries):")
            print(f"Scenario Accuracy: {scenario_accuracy:.2f}%")
            print(f"Variable Accuracy: {variable_accuracy:.2f}%")
            print(f"Year Range Accuracy: {year_range_accuracy:.2f}%")
            print(f"File Path Accuracy: {file_path_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry failed queries from an evaluation run"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input evaluation results file (default: most recent file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to store retry results",
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

    retry_failed_queries(
        input_file=args.input_file,
        output_dir=args.output_dir,
        debug=args.debug,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_workers=args.max_workers,
    )
