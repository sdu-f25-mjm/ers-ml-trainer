# TODO List

## Database & Schema
- [ ] Ensure all tables are created on startup, including `best_models` and `cache_metrics`.
  - [x] Review the startup sequence to ensure `create_tables` or equivalent is called before any DB operations.
    - [x] **Check the main entry point (e.g., `main.py`, `app.py`, or server startup script) to verify `create_tables` is invoked before any database access.**
      - [x] **Locate the main entry file(s) responsible for starting the application.**
        - [x] `main.py` found at `C:\Github\caching service\ers-ml-trainer\main.py`
        - [x] `app.py` found at `C:\Github\caching service\ers-ml-trainer\api\app.py`
      - [x] Search for the `create_tables` function call and ensure it is executed before any database operations or API/server startup.
        - [x] Added call to `create_tables` in `main.py:create_app()` before app/server startup.
      - [x] Add logging to confirm table creation at startup.
      - [x] If missing, add a call to `create_tables` at the very beginning of the startup process.
    - [x] Ensure proper error handling/logging if table creation fails.
  - [x] Confirm that the SQL for both `best_models` and `cache_metrics` is present in the table creation logic.
  - [x] Add logging to verify table existence after startup.
- [ ] Add migrations or schema versioning if schema changes are frequent.

## Model Training & Evaluation
- [ ] Allow dynamic selection of feature columns based on `cache_metrics` columns.
- [ ] Validate feature columns before training starts.
- [ ] Add reasoning and cache membership columns to RL evaluation results.
- [ ] Improve error handling for observation shape mismatches.

## API
- [ ] Provide endpoint to fetch available feature columns as enum values.
- [ ] Document all API endpoints with example requests/responses.
- [ ] Add endpoint to retrieve and download the best model from `best_models`.

## Simulation & Metrics
- [ ] Expand cache metrics simulation to cover more realistic scenarios.
- [ ] Add more metrics (e.g., cache churn, eviction rate) to `cache_metrics`.

## Visualization
- [ ] Display reasoning and cache membership in evaluation visualizations.
- [ ] Add more interactive plots for cache performance.
- [ ] Display RL model metrics with both text and graphs, showing details of which items should be cached for further analysis.

## General
- [ ] Add unit and integration tests for all modules.
- [ ] Improve logging and error reporting.
- [ ] Add documentation for setup, usage, and contributing.

