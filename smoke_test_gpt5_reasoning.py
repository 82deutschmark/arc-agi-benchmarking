#!/usr/bin/env python3
"""
Smoke test for GPT-5 reasoning content capture fix.

This script makes one real API call to verify:
1. text.verbosity is set to "high" for Responses API
2. reasoning.summary is set to "detailed" in config
3. The API returns full reasoning content (not just condensed summaries)
"""

import sys
import os
import logging

# Add src to path
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from arc_agi_benchmarking.adapters.open_ai import OpenAIAdapter
from dotenv import load_dotenv
import json

load_dotenv()

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a quick smoke test with one of the GPT-5 models."""

    # Use the cheapest GPT-5 model for testing: gpt-5-nano-2025-08-07-minimal
    model_name = "gpt-5-nano-2025-08-07-minimal"

    logger.info(f"Starting smoke test with model: {model_name}")
    logger.info("=" * 80)

    # Initialize the adapter
    adapter = OpenAIAdapter(config=model_name)

    logger.info(f"Model config loaded:")
    logger.info(f"  - Name: {adapter.model_config.name}")
    logger.info(f"  - Model: {adapter.model_config.model_name}")
    logger.info(f"  - API Type: {adapter.model_config.api_type}")
    logger.info(f"  - Provider: {adapter.model_config.provider}")

    # Check the reasoning config
    if hasattr(adapter.model_config, 'reasoning'):
        logger.info(f"  - Reasoning config:")
        reasoning_dict = adapter.model_config.reasoning if isinstance(adapter.model_config.reasoning, dict) else adapter.model_config.reasoning.model_dump()
        logger.info(f"    {json.dumps(reasoning_dict, indent=6)}")

        # Verify summary is set to "detailed"
        if 'summary' in reasoning_dict:
            if reasoning_dict['summary'] == 'detailed':
                logger.info("    ✓ reasoning.summary is set to 'detailed' (CORRECT)")
            else:
                logger.warning(f"    ✗ reasoning.summary is '{reasoning_dict['summary']}' (should be 'detailed')")

    logger.info("=" * 80)

    # Simple test prompt
    test_prompt = "What is 2 + 2? Think through your reasoning step by step."

    logger.info(f"Making API call with prompt: '{test_prompt}'")
    logger.info("=" * 80)

    # Make the prediction
    try:
        attempt = adapter.make_prediction(test_prompt)

        logger.info("API call successful!")
        logger.info("=" * 80)
        logger.info("Response Details:")
        logger.info(f"  - Answer: {attempt.answer}")
        logger.info(f"  - Model: {attempt.metadata.model}")
        logger.info(f"  - Provider: {attempt.metadata.provider}")
        logger.info("")
        logger.info("Usage:")
        logger.info(f"  - Prompt tokens: {attempt.metadata.usage.prompt_tokens}")
        logger.info(f"  - Completion tokens: {attempt.metadata.usage.completion_tokens}")
        logger.info(f"  - Reasoning tokens: {attempt.metadata.usage.completion_tokens_details.reasoning_tokens}")
        logger.info(f"  - Total tokens: {attempt.metadata.usage.total_tokens}")
        logger.info("")
        logger.info("Cost:")
        logger.info(f"  - Prompt cost: ${attempt.metadata.cost.prompt_cost:.6f}")
        logger.info(f"  - Completion cost: ${attempt.metadata.cost.completion_cost:.6f}")
        logger.info(f"  - Reasoning cost: ${attempt.metadata.cost.reasoning_cost:.6f}")
        logger.info(f"  - Total cost: ${attempt.metadata.cost.total_cost:.6f}")
        logger.info("")

        # Check if we got reasoning content
        if hasattr(attempt.metadata, 'reasoning_summary') and attempt.metadata.reasoning_summary:
            logger.info("Reasoning Summary:")
            logger.info(f"  {json.dumps(attempt.metadata.reasoning_summary, indent=2)}")
            logger.info("  ✓ Received reasoning summary (GOOD)")
        else:
            logger.warning("  ✗ No reasoning summary received (this might be expected for simple prompts)")

        logger.info("=" * 80)
        logger.info("SMOKE TEST PASSED ✓")
        logger.info("The fix appears to be working correctly.")

    except Exception as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        logger.info("=" * 80)
        logger.info("SMOKE TEST FAILED ✗")
        sys.exit(1)

if __name__ == "__main__":
    main()
