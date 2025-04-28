# TODO List

## Database & Schema
- [x] Ensure all tables are created on startup, including `best_models` and `cache_metrics`.
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
- [ ] Add reasoning and cache membership columns to RL evaluation results.
  - [ ] During evaluation, record the reasoning for each cache decision.
  - [ ] Add a boolean column indicating if the item is in the cache after each step.
  - [ ] Return these columns in the evaluation results for further analysis.
- [ ] Improve error handling for observation shape mismatches.
  - [ ] Add checks and clear error messages if the observation shape does not match model expectations.
  - [ ] Log the expected vs. actual shapes for debugging.
- [ ] Save generated best model as a base64‐encoded string in the `best_models` table.

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
- [ ] Display RL model metrics with both text and graphs, showing details of which items should be cached for further analysis.

## General
- [ ] Add unit and integration tests for all modules.
- [ ] Improve logging and error reporting.
- [ ] Add documentation for setup, usage, and contributing.


