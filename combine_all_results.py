import os
import pandas as pd
import glob
from datetime import datetime


def combine_all_results(
    original_file, first_retry_file, second_retry_file, output_dir="evaluation_results"
):
    """
    Combine results from the original evaluation and two retry attempts into a single file.

    Args:
        original_file: Path to the original evaluation results file
        first_retry_file: Path to the first retry results file
        second_retry_file: Path to the second retry results file
        output_dir: Directory to save the final combined file
    """
    print(f"Creating final combined results file from:")
    print(f"1. Original file: {original_file}")
    print(f"2. First retry file: {first_retry_file}")
    print(f"3. Second retry file: {second_retry_file}")

    # Read the files
    original_df = pd.read_csv(original_file)
    first_retry_df = pd.read_csv(first_retry_file)
    second_retry_df = pd.read_csv(second_retry_file)

    # Get successful queries from each file
    original_success = original_df[original_df["success"] == True]
    first_retry_success = first_retry_df[first_retry_df["success"] == True]
    second_retry_success = second_retry_df[second_retry_df["success"] == True]

    # Find queries that still failed in all attempts
    all_failed = second_retry_df[second_retry_df["success"] == False]

    # Combine all successful queries and the remaining failed ones
    final_df = pd.concat(
        [original_success, first_retry_success, second_retry_success, all_failed]
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"final_results_{timestamp}.csv")

    # Save the combined results
    final_df.to_csv(output_file, index=False)

    # Print summary statistics
    total_queries = len(final_df)
    successful_queries = (final_df["success"] == True).sum()
    failed_queries = total_queries - successful_queries

    print(f"\nFinal Results Summary:")
    print(f"Total Queries: {total_queries}")
    print(f"Successful Queries: {successful_queries}")
    print(f"Failed Queries: {failed_queries}")

    if successful_queries > 0:
        success_mask = final_df["success"] == True
        scenario_accuracy = final_df.loc[success_mask, "scenario_match"].mean() * 100
        variable_accuracy = final_df.loc[success_mask, "variable_match"].mean() * 100
        year_range_accuracy = (
            final_df.loc[success_mask, "year_range_match"].mean() * 100
        )
        file_path_accuracy = final_df.loc[success_mask, "file_path_match"].mean() * 100

        print("\nAccuracy Metrics (for successful queries):")
        print(f"Scenario Accuracy: {scenario_accuracy:.2f}%")
        print(f"Variable Accuracy: {variable_accuracy:.2f}%")
        print(f"Year Range Accuracy: {year_range_accuracy:.2f}%")
        print(f"File Path Accuracy: {file_path_accuracy:.2f}%")

    print(f"\nFinal results saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine all evaluation results into a final file"
    )
    parser.add_argument(
        "--original-file",
        type=str,
        default="evaluation_results/evaluation_results_20250520_104535.csv",
        help="Original evaluation results file",
    )
    parser.add_argument(
        "--first-retry-file",
        type=str,
        default="evaluation_results/retry_results_20250520_131022.csv",
        help="First retry results file",
    )
    parser.add_argument(
        "--second-retry-file", type=str, help="Second retry results file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to store final results",
    )

    args = parser.parse_args()

    # If second retry file is not specified, find the most recent retry_results file
    if not args.second_retry_file:
        retry_files = glob.glob(os.path.join(args.output_dir, "retry_results_*.csv"))
        if len(retry_files) < 2:
            print("Error: Need at least two retry result files")
            exit(1)

        # Sort files by modification time (most recent first)
        retry_files.sort(key=os.path.getmtime, reverse=True)

        # Use the most recent file that's not the first retry file
        for file in retry_files:
            if file != args.first_retry_file:
                args.second_retry_file = file
                break

    combine_all_results(
        args.original_file,
        args.first_retry_file,
        args.second_retry_file,
        args.output_dir,
    )
