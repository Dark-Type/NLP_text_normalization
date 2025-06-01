"""
Quick submission creation script for Russian text normalization.
"""

import pandas as pd
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_submission_from_results(result_file='data/results/result.csv',
                                   output_file='data/results/submission.csv'):
    """
    Create a properly formatted submission file from results.

    Args:
        result_file: Path to the result file with normalization outputs
        output_file: Path where submission file should be saved
    """
    try:
        # Load results
        if not os.path.exists(result_file):
            logger.error(f"Result file not found: {result_file}")
            return False

        df = pd.read_csv(result_file, encoding='utf-8')
        logger.info(f"Loaded results with {len(df)} rows")

        # Validate columns
        required_cols = ['id', 'after']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        submission_df = df[['id', 'after']].copy()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        submission_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Submission file created: {output_file}")
        logger.info(f"Submission contains {len(submission_df)} entries")

        logger.info("Sample entries:")
        for i, row in submission_df.head(3).iterrows():
            logger.info(f"  {row['id']}: {row['after']}")

        return True

    except Exception as e:
        logger.error(f"Error creating submission: {e}")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create submission file from results")
    parser.add_argument("--input", default="data/results/result.csv",
                        help="Input result file (default: data/results/result.csv)")
    parser.add_argument("--output", default="data/results/submission.csv",
                        help="Output submission file (default: data/results/submission.csv)")

    args = parser.parse_args()

    success = create_submission_from_results(args.input, args.output)
    if success:
        print(f"Submission file ready: {args.output}")
    else:
        print("Failed to create submission file")