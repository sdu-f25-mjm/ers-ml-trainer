# TODO List

## Database & Schema

- [x] Ensure all tables are created on startup, including `best_models` and `cache_metrics`.
    - [x] Review the startup sequence to ensure `create_tables` or equivalent is called before any DB operations.
        - [x] **Check the main entry point (e.g., `main.py`, `app.py`, or server startup script) to
          verify `create_tables` is invoked before any database access.**
            - [x] **Locate the main entry file(s) responsible for starting the application.**
                - [x] `main.py` found at `C:\Github\caching service\ers-ml-trainer\main.py`
                - [x] `app.py` found at `C:\Github\caching service\ers-ml-trainer\api\app.py`
            - [x] Search for the `create_tables` function call and ensure it is executed before any database operations
              or API/server startup.
                - [x] Added call to `create_tables` in `main.py:create_app()` before app/server startup.
            - [x] Add logging to confirm table creation at startup.
            - [x] If missing, add a call to `create_tables` at the very beginning of the startup process.
        - [x] Ensure proper error handling/logging if table creation fails.
    - [x] Confirm that the SQL for both `best_models` and `cache_metrics` is present in the table creation logic.
    - [x] Add logging to verify table existence after startup.
- [ ] Add migrations or schema versioning if schema changes are frequent.
    - [ ] **Evaluate tools like Alembic or Django migrations for schema management.**
    - [ ] Add migration scripts or instructions for updating the database schema.
    - [ ] Document the migration/versioning process for contributors.

## Model Training & Evaluation

- [ ] Allow dynamic selection of feature columns based on `cache_metrics` columns.
    - [x] Implement logic to fetch available columns from `cache_metrics` at runtime.
        - [x] Add a utility/database function to query column names from the `cache_metrics` table.
        - [x] Ensure this function is used wherever dynamic feature columns are needed (e.g., API, training).
        - [x] Add tests or logging to verify correct column fetching.
    - [x] Update training API to accept a dynamic list of feature columns.
    - [x] Ensure UI/API allows user to select from these columns.
- [x] Validate feature columns before training starts.
    - [x] Check that selected columns exist in `cache_metrics` before model training.
    - [x] Return a clear error if invalid columns are provided.
- [x] Add reasoning and cache membership columns to RL evaluation results.
    - [x] During evaluation, record the reasoning for each cache decision.
    - [x] Add a boolean column indicating if the item is in the cache after each step.
    - [x] Return these columns in the evaluation results for further analysis.
- [ ] Save generated best model as a base64‐encoded string in the `best_models` table.
- [ ] Improve error handling for observation shape mismatches.
    - [x] Add checks and clear error messages if the observation shape does not match model expectations.
    - [x] Log the expected vs. actual shapes for debugging.
    - [x] Auto‑adjust `cache_size` using model’s observation shape when mismatches occur.
- [ ] Improve auto-adjustment for observation shape mismatches.
    - [x] Provide clearer error messages and suggestions for resolving persistent mismatches.
    - [x] Log the values of feature_columns and cache_size used in both training and evaluation for debugging.
    - [x] Enhance logic to handle cases where auto-adjusted shape still mismatches expected shape.
    - [ ] Add user-facing documentation or API guidance: **If you see an observation shape mismatch, ensure you use the same `feature_columns` and `cache_size` for both training and evaluation.**  
          Example:  
          ```python
          with open(f"{model_path}.meta.json") as f:
              meta = json.load(f)
          feature_columns = meta["feature_columns"]
          cache_size = meta["cache_size"]
          # Pass these to evaluate_cache_model(..., feature_columns=feature_columns, cache_size=cache_size)
          ```
    - [ ] Expose an API endpoint to fetch `feature_columns` and `cache_size` from model metadata for user convenience.
- [ ] Standardize evaluation output format.
    - [x] Include a `success: true` flag on successful runs.
    - [x] Always return both `error` (nullable) and `success` fields.
- [ ] Add unit tests for `evaluate_cache_model`.
    - [ ] Test normal evaluation returns metrics plus `reasoning` and `in_cache_membership`.
    - [ ] Test shape‑mismatch handling and GPU→CPU retry path.

## API

- [ ] Provide endpoint to fetch available feature columns as enum values.
    - [ ] Implement GET `/features` endpoint in `api/app.py` or `app.py`.
    - [ ] Use `get_dynamic_feature_columns_enum` to generate the enum payload.
    - [ ] Write integration tests for `/features`.
    - [ ] Add example requests/responses in OpenAPI docs.
- [ ] Add endpoint to retrieve and download the best model from `best_models`.
    - [ ] Implement GET `/best-model` endpoint returning model artifacts.
    - [ ] Write tests for `/best-model`.
    - [ ] Add example in API documentation.
- [ ] Document all API endpoints with example requests/responses.

## Simulation & Metrics

- [ ] Expand cache metrics simulation to cover more realistic scenarios.
- [ ] Add more metrics (e.g., cache churn, eviction rate) to `cache_metrics`.

## Visualization

- [ ] Display reasoning and cache membership in evaluation visualizations.
- [ ] Add more interactive plots for cache performance.
- [ ] Display RL model metrics with both text and graphs, showing details of which items should be cached for further
  analysis.

## General

- [ ] Add unit and integration tests for all modules.
- [ ] Improve logging and error reporting.
- [ ] Add documentation for setup, usage, and contributing.






`