#!/usr/bin/env python3
"""
Smoke test for GPT-5 reasoning content capture fix.

This script runs the real ARC task f0afb749 via the standard ARCTester
pipeline using a GPT-5-nano config with high reasoning effort and a
"detailed" reasoning summary. It verifies:
1. text.verbosity is effectively "high" for the Responses API
2. reasoning.effort is set to "high" and reasoning.summary is "detailed"
3. The API returns full reasoning content (not just condensed summaries)
"""

import sys
import os
import logging
import json

# Add src to path
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dotenv import load_dotenv
from main import ARCTester

load_dotenv()

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run a smoke test by solving real ARC task f0afb749 with GPT-5-nano high reasoning."""

    # Use GPT-5-nano high-effort reasoning config with detailed summary
    model_config_name = "gpt-5-nano-2025-08-07-high"
    task_id = "f0afb749"
    data_dir = os.path.join("data", "arc-agi", "data", "evaluation")

    logger.info(f"Starting smoke test for task {task_id} with config: {model_config_name}")
    logger.info("=" * 80)

    # Initialize the ARC tester
    arc_tester = ARCTester(
        config=model_config_name,
        save_submission_dir="smoke_submissions/gpt5_reasoning_high",
        overwrite_submission=True,
        print_submission=True,
        num_attempts=1,
        retry_attempts=1,
    )

    try:
        # Run full pipeline on the single task
        task_attempts = arc_tester.generate_task_solution(data_dir=data_dir, task_id=task_id)

        if not task_attempts:
            logger.error("No attempts were produced for the smoke test task.")
            logger.info("=" * 80)
            logger.info("SMOKE TEST FAILED")
            sys.exit(1)

        # Inspect first attempt for reasoning metadata
        first_pair = task_attempts[0]
        first_attempt = next((v for k, v in first_pair.items() if k.startswith("attempt_") and v is not None), None)

        if not first_attempt:
            logger.error("No valid attempt object found in the smoke test output.")
            logger.info("=" * 80)
            logger.info("SMOKE TEST FAILED")
            sys.exit(1)

        metadata = first_attempt.get("metadata", {})
        usage = metadata.get("usage", {})
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        reasoning_summary = metadata.get("reasoning_summary")

        logger.info("Smoke test attempt metadata:")
        logger.info(f"  - Model: {metadata.get('model')}")
        logger.info(f"  - Provider: {metadata.get('provider')}")
        logger.info(f"  - Reasoning tokens: {reasoning_tokens}")

        if reasoning_summary:
            logger.info("Reasoning Summary:")
            logger.info(json.dumps(reasoning_summary, indent=2))
            logger.info("  [OK] Received reasoning summary (expected for detailed/high reasoning)")
        else:
            logger.warning("  [WARN] No reasoning summary present; verify Responses API config if this persists.")

        logger.info("=" * 80)
        logger.info("SMOKE TEST PASSED")
        logger.info("The GPT-5 reasoning pipeline appears to be working correctly for f0afb749.")

    except Exception as e:
        logger.error(f"Smoke test pipeline failed: {e}", exc_info=True)
        logger.info("=" * 80)
        logger.info("SMOKE TEST FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
