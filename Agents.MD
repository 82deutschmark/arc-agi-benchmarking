# AGENTS.md

This file provides guidance to AI coding agents (e.g., Claude Code, Cursor, Copilot, Replit Agent, etc.) when working with code in this repository.

## Overview

This repository tests AI model baselines on ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence). It provides infrastructure for running language models against visual pattern recognition tasks, managing concurrent requests, rate limiting, and collecting results.

## Development Commands

### Setup
```bash
# Clone and initialize
THE USUAL STUFF!!!  

# Install provider SDKs (for API error handling)
pip install openai anthropic google-api-python-client
```

### Environment Configuration
Copy `.env.example` to `.env` and add API keys:
```bash
OPENAI_API_KEY="your_key_here"
ANTHROPIC_API_KEY="your_key_here"
# Add other provider keys as needed  DUH!!!
```

### Running Tests

**Batch Testing (Recommended):**
```bash
# Run all tasks with concurrency, rate limiting, and retries
python cli/run_all.py \
    --task_list_file data/task_lists/public_evaluation_v1.txt \
    --model_configs "gpt-4o-2024-11-20,claude_opus" \
    --num_attempts 2 \
    --log-level INFO

# Enable metrics for performance analysis
python cli/run_all.py \
    --task_list_file data/task_lists/sample_tasks.txt \
    --model_configs "gpt-4o-2024-11-20" \
    --enable-metrics \
    --log-level DEBUG
```

**Single Task Testing (Debugging):**
```bash
# Standard single task run
python main.py --data_dir data/arc-agi/data/evaluation --config gpt-4o-2024-11-20 --task_id 0a1d4ef5

# Verbose debugging mode (shows API calls, token usage, full tracebacks)
python main.py --data_dir data/arc-agi/data/evaluation --config grok-4-0709 --task_id 0a1d4ef5 --verbose

# With metrics enabled
python main.py --data_dir data/arc-agi/data/evaluation --config gpt-4o-2024-11-20 --log-level DEBUG --enable-metrics
```

**Running Unit Tests:**
```bash
pytest
```

### Scoring Results
```bash
python src.scoring.scoring.py \
    --task_dir data/arc-agi/data/evaluation \
    --submission_dir submissions/gpt-4o-2024-11-20 \
    --results_dir results/gpt-4o-2024-11-20
```

### Uploading to Hugging Face
```bash
# Authenticate first
export HUGGING_FACE_HUB_TOKEN=your_token_here
# Or: huggingface-cli login

# Upload single model
python cli/submission_cli.py upload submissions/model_name --task-set public_eval_v1

# Bulk upload
python cli/submission_cli.py bulk-upload submissions/ --task-set public_eval_v1

# Public repository
python cli/submission_cli.py upload submissions/model_name --task-set public_eval_v1 --public
```

## Architecture

### Core Components

**ARCTester (main.py)**
- Main class orchestrating task prediction for a single model configuration
- Loads model config from `models.yml` and initializes appropriate provider adapter
- Manages prediction attempts, retries, and submission file creation
- Methods:
  - `predict_task_output()`: Converts task to prompt, calls provider
  - `get_task_prediction()`: Handles parsing and retry logic
  - `run_task_evaluation()`: Evaluates an entire task with multiple test pairs

**Provider Adapters (src/arc_agi_benchmarking/adapters/)**
- Abstract base class: `ProviderAdapter` defines interface for all providers
- Each provider (OpenAI, Anthropic, Gemini, DeepSeek, etc.) extends base class
- Key responsibilities:
  - Initialize API client with authentication
  - Convert prompts to provider-specific API calls
  - Parse responses and extract JSON grids
  - Track token usage and costs
  - Handle provider-specific features (reasoning tokens, streaming, etc.)
- Returns standardized `Attempt` objects containing:
  - `answer`: Parsed grid output
  - `metadata`: Usage stats, costs, model info, raw response

**Batch Orchestrator (cli/run_all.py)**
- Uses `asyncio` for concurrent task execution across multiple models
- Implements provider-level rate limiting via `AsyncRequestRateLimiter`
- Automatic retry logic using `tenacity` library for transient failures
- Centralizes logging and optional metrics collection
- Queues all (task_id, model_config) pairs and processes them concurrently

**Configuration System**
- `models.yml`: Defines all model configurations with:
  - `name`: Unique identifier for the config
  - `model_name`: Actual model name for API
  - `provider`: Which adapter to use
  - Parameters: max_tokens, temperature, reasoning_effort, etc.
  - `pricing`: Input/output costs per 1M tokens
- `provider_config.yml`: Rate limits per provider (requests/period)

**Data Flow**
1. Task loaded from JSON files in `data/arc-agi/data/`
2. `ARCTester` converts task pairs to text prompt via `convert_task_pairs_to_prompt()`
3. Provider adapter makes API call and returns `Attempt` object
4. Response parsed to extract grid answer (list of lists of ints)
5. Result saved as JSON in `submissions/{model_name}/{task_id}.json`
6. Scoring script compares submissions to ground truth answers

### Key Design Patterns

**Multi-Configuration Support**
- Same model can have multiple configs (e.g., `o1-2024-12-17-low`, `o1-2024-12-17-medium`, `o1-2024-12-17-high`)
- Configs differ by reasoning_effort, max_tokens, temperature, etc.
- Config name is used as directory name in submissions

**Unified Response Format**
- All providers return `Attempt` objects with standardized schema
- Metadata includes: model, provider, timestamps, usage, cost, raw_response
- Enables consistent metrics tracking across providers

**Rate Limiting Strategy**
- Provider-level limits from `provider_config.yml`
- Uses token bucket algorithm via `AsyncRequestRateLimiter`
- Prevents hitting API rate limits before errors occur
- Combines with tenacity retries for resilient execution

**Logging Architecture**
- Uses Python logging module with configurable levels
- `--log-level`: Controls global verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE)
- `--verbose`: Convenience flag for detailed debugging (sets DEBUG + reduces library noise)
- Library loggers (httpx, openai, anthropic) kept at WARNING to reduce noise
- Per-task logging includes: task_id, test_id, pair_index, costs, token usage

**Metrics Collection**
- Disabled by default to reduce overhead
- Enable with `--enable-metrics` flag
- Tracks timing, API calls, token usage per task
- Outputs to `metrics_output/` directory

## Adding New Providers

1. **Add model configurations to `src/arc_agi_benchmarking/models.yml`**
   ```yaml
   models:
     - name: "my-model-config"
       model_name: "actual-api-model-name"
       provider: "myprovider"
       max_tokens: 4024
       temperature: 0.0
       pricing:
         date: "2025-01-01"
         input: 1.00
         output: 3.00
   ```

2. **Add rate limits to `provider_config.yml`**
   ```yaml
   myprovider:
     rate: 400
     period: 60
   ```

3. **Create adapter in `src/arc_agi_benchmarking/adapters/myprovider.py`**
   - Extend `ProviderAdapter`
   - Implement `init_client()`: Set up API client, handle auth
   - Implement `make_prediction(prompt, task_id, test_id, pair_index)`: Make API call, return `Attempt`
   - Implement `extract_json_from_response(response)`: Parse response to grid format
   - Handle token counting and cost calculation
   - Store raw response in metadata for debugging

4. **Register adapter in `src/arc_agi_benchmarking/adapters/__init__.py`**
   ```python
   from .myprovider import MyProviderAdapter
   ```

5. **Update `ARCTester.init_provider()` in `main.py`**
   ```python
   elif provider_name == "myprovider":
       return MyProviderAdapter(self.config)
   ```

6. **Test the new provider**
   ```bash
   python main.py --data_dir data/arc-agi/data/evaluation --config my-model-config --task_id 0a1d4ef5 --verbose
   ```

## Important Notes

### API Type Differences
- OpenAI models use either `api_type: "responses"` (newer) or `api_type: "chat_completions"` (legacy)
- The Responses API supports reasoning models with extended thinking capabilities
- Different API types affect how responses are parsed and metadata is extracted

### Reasoning Models
- Models like o1, o3, Gemini with thinking, Claude with extended thinking support special reasoning parameters
- OpenAI: `reasoning_effort` (low/medium/high) or `reasoning` dict with `effort` and `summary`
- Gemini: `thinking_config` with `thinking_budget`
- Claude: `thinking` with `type` and `budget_tokens`
- Reasoning tokens counted separately in usage metrics and may have different pricing

### Background Jobs
- Some models (like o3-pro) use `background: true` for long-running requests
- These require polling for completion rather than streaming
- Implemented in OpenAI adapter's `_poll_background_response()` method

### Submission Format
- Submissions must be JSON with exactly 2 attempts per test case
- Each attempt is a list of lists of integers (grid format)
- Format: `{"task_id": {"attempt_1": [[...]], "attempt_2": [[...]]}}`
- Multiple test cases per task supported

### Common Pitfalls
- Provider name in `models.yml` must exactly match adapter name (case-sensitive)
- Rate limits in `provider_config.yml` must match provider name in `models.yml`
- API keys must be set in `.env` file with correct variable names (check adapter code)
- Metrics disabled by default - must explicitly enable with `--enable-metrics`
- When debugging, use `--verbose` flag to see API interactions without library noise
