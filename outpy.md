"C:\Github\caching service\ers-ml-trainer\.venv\Scripts\python.exe" "C:\Github\caching service\ers-ml-trainer\main.py" 
2025-05-02 09:45:05,882 - api.app_utils - INFO - Loading trained models into memory...
2025-05-02 09:45:05,883 - api.app_utils - INFO - Loaded 1 trained models
2025-05-02 09:45:07,205 - core.utils - INFO - Using database URL: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:45:07,205 - core.utils - INFO - Using database URL: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
Found 1 models:
- DQN (cache size: 10) trained on CPU at 20250430_132433
2025-05-02 09:45:07,279 - database.database_connection - INFO - Successfully connected to database.
2025-05-02 09:45:07,297 - __main__ - INFO - Logging initialized at INFO level
2025-05-02 09:45:07,309 - database.mysql_db - INFO - Connected to MySQL at 192.168.1.81:3306
2025-05-02 09:45:07,309 - database.mysql_db - INFO - Using database: cache_db
2025-05-02 09:45:07,310 - database.create_tables - INFO - Table/index for 'energy_data' created or already exists.
2025-05-02 09:45:07,310 - database.create_tables - INFO - Table/index for 'energy_data timestamp index' created or already exists.
2025-05-02 09:45:07,311 - database.create_tables - INFO - Table/index for 'energy_data price_area index' created or already exists.
2025-05-02 09:45:07,311 - database.create_tables - INFO - Table/index for 'production_data' created or already exists.
2025-05-02 09:45:07,311 - database.create_tables - INFO - Table/index for 'production_data timestamp index' created or already exists.
2025-05-02 09:45:07,311 - database.create_tables - INFO - Table/index for 'production_data price_area index' created or already exists.
2025-05-02 09:45:07,312 - database.create_tables - INFO - Table/index for 'consumption_data' created or already exists.
2025-05-02 09:45:07,312 - database.create_tables - INFO - Table/index for 'consumption_data timestamp index' created or already exists.
2025-05-02 09:45:07,312 - database.create_tables - INFO - Table/index for 'consumption_data price_area index' created or already exists.
2025-05-02 09:45:07,313 - database.create_tables - INFO - Table/index for 'exchange_data' created or already exists.
2025-05-02 09:45:07,313 - database.create_tables - INFO - Table/index for 'exchange_data timestamp index' created or already exists.
2025-05-02 09:45:07,314 - database.create_tables - INFO - Table/index for 'exchange_data price_area index' created or already exists.
2025-05-02 09:45:07,314 - database.create_tables - INFO - Table/index for 'carbon_intensity' created or already exists.
2025-05-02 09:45:07,314 - database.create_tables - INFO - Table/index for 'carbon_intensity timestamp index' created or already exists.
2025-05-02 09:45:07,315 - database.create_tables - INFO - Table/index for 'carbon_intensity price_area index' created or already exists.
2025-05-02 09:45:07,315 - database.create_tables - INFO - Table/index for 'aggregated_production' created or already exists.
2025-05-02 09:45:07,315 - database.create_tables - INFO - Table/index for 'aggregated_production period_start index' created or already exists.
2025-05-02 09:45:07,315 - database.create_tables - INFO - Table/index for 'aggregated_production price_area index' created or already exists.
2025-05-02 09:45:07,317 - database.create_tables - INFO - Table/index for 'comparison_analysis' created or already exists.
2025-05-02 09:45:07,317 - database.create_tables - INFO - Table/index for 'comparison_analysis price_area index' created or already exists.
2025-05-02 09:45:07,317 - database.create_tables - INFO - Table/index for 'consumption_forecast' created or already exists.
2025-05-02 09:45:07,318 - database.create_tables - INFO - Table/index for 'consumption_forecast request_date index' created or already exists.
2025-05-02 09:45:07,318 - database.create_tables - INFO - Table/index for 'consumption_forecast price_area index' created or already exists.
2025-05-02 09:45:07,419 - database.create_tables - INFO - Table/index for 'cache_metrics' created or already exists.
2025-05-02 09:45:07,420 - database.create_tables - INFO - Table/index for 'rl_models' created or already exists.
2025-05-02 09:45:07,420 - database.create_tables - INFO - All tables creation attempted (see errors above if any failed)
2025-05-02 09:45:07,420 - __main__ - INFO - Database tables ensured at startup.
2025-05-02 09:45:07,420 - __main__ - INFO - Table 'cache_metrics' exists in the database.
2025-05-02 09:45:07,421 - __main__ - INFO - Table 'rl_models' exists in the database.
2025-05-02 09:45:07,422 - __main__ - INFO - Logging initialized at INFO level
2025-05-02 09:45:07,422 - core.utils - INFO - CPU: 6 cores, 12 threads
2025-05-02 09:45:07,422 - core.utils - INFO - CPU Frequency: 3701.00MHz
2025-05-02 09:45:07,428 - core.utils - INFO - RAM: 31.92GB total, 13.28GB available
2025-05-02 09:45:07,522 - core.utils - INFO - GPU 0: NVIDIA GeForce RTX 3060 Ti
2025-05-02 09:45:07,522 - core.utils - INFO -   Memory: 1357MB used / 8192MB total
2025-05-02 09:45:07,522 - core.utils - INFO -   Temperature: 48.0°C
2025-05-02 09:45:07,522 - __main__ - INFO - Starting server on 0.0.0.0:8000
INFO:     Started server process [29296]
INFO:     Waiting for application startup.
2025-05-02 09:45:07,539 - api.app - INFO - Testing database connection on startup...
2025-05-02 09:45:07,540 - core.utils - INFO - Using database URL: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:45:07,540 - database.database_connection - INFO - Attempting to connect to database (attempt 1/5)...
2025-05-02 09:45:07,578 - database.database_connection - INFO - Successfully connected to database at 192.168.1.81:3306/cache_db
2025-05-02 09:45:07,578 - api.app - INFO - Database connection test successful
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
2025-05-02 09:45:14,006 - mock.mock_db - INFO - [<TableEnum.cache_metrics: 'cache_metrics'>]
2025-05-02 09:45:14,006 - mock.mock_db - INFO - Data types provided: ['cache_metrics']
2025-05-02 09:45:14,006 - mock.mock_db - INFO - Generating mock database for mysql with 100000 hours of data for: DK1, DK2
2025-05-02 09:45:14,018 - database.mysql_db - INFO - Connected to MySQL at 192.168.1.81:3306
2025-05-02 09:45:14,019 - database.mysql_db - INFO - Using database: cache_db
2025-05-02 09:45:14,019 - database.create_tables - INFO - Table/index for 'energy_data' created or already exists.
2025-05-02 09:45:14,019 - database.create_tables - INFO - Table/index for 'energy_data timestamp index' created or already exists.
2025-05-02 09:45:14,020 - database.create_tables - INFO - Table/index for 'energy_data price_area index' created or already exists.
2025-05-02 09:45:14,020 - database.create_tables - INFO - Table/index for 'production_data' created or already exists.
2025-05-02 09:45:14,020 - database.create_tables - INFO - Table/index for 'production_data timestamp index' created or already exists.
2025-05-02 09:45:14,021 - database.create_tables - INFO - Table/index for 'production_data price_area index' created or already exists.
2025-05-02 09:45:14,021 - database.create_tables - INFO - Table/index for 'consumption_data' created or already exists.
2025-05-02 09:45:14,021 - database.create_tables - INFO - Table/index for 'consumption_data timestamp index' created or already exists.
2025-05-02 09:45:14,021 - database.create_tables - INFO - Table/index for 'consumption_data price_area index' created or already exists.
2025-05-02 09:45:14,022 - database.create_tables - INFO - Table/index for 'exchange_data' created or already exists.
2025-05-02 09:45:14,022 - database.create_tables - INFO - Table/index for 'exchange_data timestamp index' created or already exists.
2025-05-02 09:45:14,022 - database.create_tables - INFO - Table/index for 'exchange_data price_area index' created or already exists.
2025-05-02 09:45:14,023 - database.create_tables - INFO - Table/index for 'carbon_intensity' created or already exists.
2025-05-02 09:45:14,023 - database.create_tables - INFO - Table/index for 'carbon_intensity timestamp index' created or already exists.
2025-05-02 09:45:14,023 - database.create_tables - INFO - Table/index for 'carbon_intensity price_area index' created or already exists.
2025-05-02 09:45:14,023 - database.create_tables - INFO - Table/index for 'aggregated_production' created or already exists.
2025-05-02 09:45:14,024 - database.create_tables - INFO - Table/index for 'aggregated_production period_start index' created or already exists.
2025-05-02 09:45:14,024 - database.create_tables - INFO - Table/index for 'aggregated_production price_area index' created or already exists.
2025-05-02 09:45:14,024 - database.create_tables - INFO - Table/index for 'comparison_analysis' created or already exists.
2025-05-02 09:45:14,025 - database.create_tables - INFO - Table/index for 'comparison_analysis price_area index' created or already exists.
2025-05-02 09:45:14,025 - database.create_tables - INFO - Table/index for 'consumption_forecast' created or already exists.
2025-05-02 09:45:14,025 - database.create_tables - INFO - Table/index for 'consumption_forecast request_date index' created or already exists.
2025-05-02 09:45:14,026 - database.create_tables - INFO - Table/index for 'consumption_forecast price_area index' created or already exists.
2025-05-02 09:45:14,026 - database.create_tables - INFO - Table/index for 'cache_metrics' created or already exists.
2025-05-02 09:45:14,026 - database.create_tables - INFO - Table/index for 'rl_models' created or already exists.
2025-05-02 09:45:14,026 - database.create_tables - INFO - All tables creation attempted (see errors above if any failed)
2025-05-02 09:45:14,026 - mock.mock_db - INFO - Generating derived data cache weights (normalized endpoint names)
2025-05-02 09:45:24,028 - mock.mock_db - INFO - Mock database generation completed successfully
INFO:     127.0.0.1:53215 - "POST /db/seed?db_type=mysql&host=192.168.1.81&port=3306&user=cacheuser&password=cachepass&database=cache_db&hours=100000&data_types=cache_metrics&use_simulate_live=false&n=10000 HTTP/1.1" 200 OK
2025-05-02 09:46:28,280 - api.app - INFO - Starting training model 7c9426ad-4c80-4b5d-b1b3-664924a8c8e7
2025-05-02 09:46:28,280 - api.app - INFO - Database URL: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:46:28,280 - api.app - INFO - Training model added to queue
INFO:     127.0.0.1:53231 - "POST /train?db_type=mysql%2Bmysqlconnector&host=192.168.1.81&port=3306&user=cacheuser&password=cachepass&database=cache_db&algorithm=dqn&cache_size=10&max_queries=500&timesteps=100000&table_name=cache_metrics&use_gpu=false HTTP/1.1" 200 OK
2025-05-02 09:46:28,306 - api.app_utils - INFO - Starting training for job 7c9426ad-4c80-4b5d-b1b3-664924a8c8e7
2025-05-02 09:46:28,307 - api.app_utils - INFO - Running training job 7c9426ad-4c80-4b5d-b1b3-664924a8c8e7 with algorithm AlgorithmEnum.dqn
2025-05-02 09:46:28,347 - api.app_utils - INFO - is_cuda_available: True, use_gpu: False
2025-05-02 09:46:28,394 - api.app_utils - INFO - Training job 7c9426ad-4c80-4b5d-b1b3-664924a8c8e7 with algorithm AlgorithmEnum.dqn using CPU
2025-05-02 09:46:28,394 - core.model_training - INFO - Using batch size: 64
2025-05-02 09:46:28,394 - core.cache_environment - INFO - Connecting to database: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:46:28,429 - database.database_connection - INFO - Successfully connected to database.
2025-05-02 09:46:28,431 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,431 - core.cache_environment - INFO - Available tables: aggregated_production, cache_metrics, carbon_intensity, comparison_analysis, consumption_data, consumption_forecast, energy_data, exchange_data, production_data, rl_models
2025-05-02 09:46:28,433 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,433 - core.cache_environment - INFO - Using specified table: cache_metrics
2025-05-02 09:46:28,434 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,436 - core.cache_environment - INFO - Loading data from table: cache_metrics
2025-05-02 09:46:28,465 - core.cache_environment - INFO - Loaded 1000 rows from cache_metrics
2025-05-02 09:46:28,467 - core.cache_environment - INFO - Connecting to database: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:46:28,492 - database.database_connection - INFO - Successfully connected to database.
2025-05-02 09:46:28,494 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,495 - core.cache_environment - INFO - Available tables: aggregated_production, cache_metrics, carbon_intensity, comparison_analysis, consumption_data, consumption_forecast, energy_data, exchange_data, production_data, rl_models
2025-05-02 09:46:28,496 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,496 - core.cache_environment - INFO - Using specified table: cache_metrics
2025-05-02 09:46:28,498 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:46:28,500 - core.cache_environment - INFO - Loading data from table: cache_metrics
2025-05-02 09:46:28,510 - core.cache_environment - INFO - Loaded 1000 rows from cache_metrics
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
2025-05-02 09:46:30,123 - core.model_training - INFO - Starting training with DQN for 100000 timesteps...
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.81     |
| time/               |          |
|    episodes         | 4        |
|    fps              | 2297     |
|    time_elapsed     | 0        |
|    total_timesteps  | 2000     |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.006    |
|    n_updates        | 249      |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.62     |
| time/               |          |
|    episodes         | 8        |
|    fps              | 1816     |
|    time_elapsed     | 2        |
|    total_timesteps  | 4000     |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.000453 |
|    n_updates        | 749      |
----------------------------------
Eval num_timesteps=5000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.525    |
| time/               |          |
|    total_timesteps  | 5000     |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.000468 |
|    n_updates        | 999      |
----------------------------------
New best mean reward!
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.43     |
| time/               |          |
|    episodes         | 12       |
|    fps              | 1298     |
|    time_elapsed     | 4        |
|    total_timesteps  | 6000     |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.000435 |
|    n_updates        | 1249     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.24     |
| time/               |          |
|    episodes         | 16       |
|    fps              | 1310     |
|    time_elapsed     | 6        |
|    total_timesteps  | 8000     |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.000762 |
|    n_updates        | 1749     |
----------------------------------
Eval num_timesteps=10000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.0501   |
| time/               |          |
|    total_timesteps  | 10000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00188  |
|    n_updates        | 2249     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 20       |
|    fps              | 1169     |
|    time_elapsed     | 8        |
|    total_timesteps  | 10000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 24       |
|    fps              | 1183     |
|    time_elapsed     | 10       |
|    total_timesteps  | 12000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0014   |
|    n_updates        | 2749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 28       |
|    fps              | 1203     |
|    time_elapsed     | 11       |
|    total_timesteps  | 14000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00371  |
|    n_updates        | 3249     |
----------------------------------
Eval num_timesteps=15000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 15000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00111  |
|    n_updates        | 3499     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 32       |
|    fps              | 1140     |
|    time_elapsed     | 14       |
|    total_timesteps  | 16000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00852  |
|    n_updates        | 3749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 36       |
|    fps              | 1158     |
|    time_elapsed     | 15       |
|    total_timesteps  | 18000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0063   |
|    n_updates        | 4249     |
----------------------------------
Eval num_timesteps=20000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 20000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00575  |
|    n_updates        | 4749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 40       |
|    fps              | 1124     |
|    time_elapsed     | 17       |
|    total_timesteps  | 20000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 44       |
|    fps              | 1139     |
|    time_elapsed     | 19       |
|    total_timesteps  | 22000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00599  |
|    n_updates        | 5249     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 48       |
|    fps              | 1152     |
|    time_elapsed     | 20       |
|    total_timesteps  | 24000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00119  |
|    n_updates        | 5749     |
----------------------------------
Eval num_timesteps=25000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 25000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00222  |
|    n_updates        | 5999     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 52       |
|    fps              | 1119     |
|    time_elapsed     | 23       |
|    total_timesteps  | 26000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00245  |
|    n_updates        | 6249     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 56       |
|    fps              | 1132     |
|    time_elapsed     | 24       |
|    total_timesteps  | 28000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00189  |
|    n_updates        | 6749     |
----------------------------------
Eval num_timesteps=30000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 30000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00151  |
|    n_updates        | 7249     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 60       |
|    fps              | 1103     |
|    time_elapsed     | 27       |
|    total_timesteps  | 30000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 64       |
|    fps              | 1116     |
|    time_elapsed     | 28       |
|    total_timesteps  | 32000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00183  |
|    n_updates        | 7749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 68       |
|    fps              | 1126     |
|    time_elapsed     | 30       |
|    total_timesteps  | 34000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00956  |
|    n_updates        | 8249     |
----------------------------------
Eval num_timesteps=35000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 35000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00467  |
|    n_updates        | 8499     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 72       |
|    fps              | 1108     |
|    time_elapsed     | 32       |
|    total_timesteps  | 36000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00509  |
|    n_updates        | 8749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 76       |
|    fps              | 1118     |
|    time_elapsed     | 33       |
|    total_timesteps  | 38000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00423  |
|    n_updates        | 9249     |
----------------------------------
Eval num_timesteps=40000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 40000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00563  |
|    n_updates        | 9749     |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 80       |
|    fps              | 1103     |
|    time_elapsed     | 36       |
|    total_timesteps  | 40000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 84       |
|    fps              | 1114     |
|    time_elapsed     | 37       |
|    total_timesteps  | 42000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00586  |
|    n_updates        | 10249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 88       |
|    fps              | 1124     |
|    time_elapsed     | 39       |
|    total_timesteps  | 44000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0615   |
|    n_updates        | 10749    |
----------------------------------
Eval num_timesteps=45000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 45000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00616  |
|    n_updates        | 10999    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 92       |
|    fps              | 1109     |
|    time_elapsed     | 41       |
|    total_timesteps  | 46000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0043   |
|    n_updates        | 11249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 96       |
|    fps              | 1117     |
|    time_elapsed     | 42       |
|    total_timesteps  | 48000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00599  |
|    n_updates        | 11749    |
----------------------------------
Eval num_timesteps=50000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 50000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0053   |
|    n_updates        | 12249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 100      |
|    fps              | 1104     |
|    time_elapsed     | 45       |
|    total_timesteps  | 50000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 104      |
|    fps              | 1112     |
|    time_elapsed     | 46       |
|    total_timesteps  | 52000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00238  |
|    n_updates        | 12749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 108      |
|    fps              | 1120     |
|    time_elapsed     | 48       |
|    total_timesteps  | 54000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00507  |
|    n_updates        | 13249    |
----------------------------------
Eval num_timesteps=55000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 55000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0117   |
|    n_updates        | 13499    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 112      |
|    fps              | 1109     |
|    time_elapsed     | 50       |
|    total_timesteps  | 56000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00514  |
|    n_updates        | 13749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 116      |
|    fps              | 1117     |
|    time_elapsed     | 51       |
|    total_timesteps  | 58000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0127   |
|    n_updates        | 14249    |
----------------------------------
Eval num_timesteps=60000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 60000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00441  |
|    n_updates        | 14749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 120      |
|    fps              | 1106     |
|    time_elapsed     | 54       |
|    total_timesteps  | 60000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 124      |
|    fps              | 1113     |
|    time_elapsed     | 55       |
|    total_timesteps  | 62000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00436  |
|    n_updates        | 15249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 128      |
|    fps              | 1120     |
|    time_elapsed     | 57       |
|    total_timesteps  | 64000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00608  |
|    n_updates        | 15749    |
----------------------------------
Eval num_timesteps=65000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 65000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0139   |
|    n_updates        | 15999    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 132      |
|    fps              | 1109     |
|    time_elapsed     | 59       |
|    total_timesteps  | 66000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00306  |
|    n_updates        | 16249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 136      |
|    fps              | 1115     |
|    time_elapsed     | 60       |
|    total_timesteps  | 68000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00535  |
|    n_updates        | 16749    |
----------------------------------
Eval num_timesteps=70000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 70000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00545  |
|    n_updates        | 17249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 140      |
|    fps              | 1107     |
|    time_elapsed     | 63       |
|    total_timesteps  | 70000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 144      |
|    fps              | 1112     |
|    time_elapsed     | 64       |
|    total_timesteps  | 72000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00851  |
|    n_updates        | 17749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 148      |
|    fps              | 1118     |
|    time_elapsed     | 66       |
|    total_timesteps  | 74000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00628  |
|    n_updates        | 18249    |
----------------------------------
Eval num_timesteps=75000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 75000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0822   |
|    n_updates        | 18499    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 152      |
|    fps              | 1109     |
|    time_elapsed     | 68       |
|    total_timesteps  | 76000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00449  |
|    n_updates        | 18749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 156      |
|    fps              | 1113     |
|    time_elapsed     | 70       |
|    total_timesteps  | 78000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00928  |
|    n_updates        | 19249    |
----------------------------------
Eval num_timesteps=80000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 80000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00604  |
|    n_updates        | 19749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 160      |
|    fps              | 1106     |
|    time_elapsed     | 72       |
|    total_timesteps  | 80000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 164      |
|    fps              | 1111     |
|    time_elapsed     | 73       |
|    total_timesteps  | 82000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0262   |
|    n_updates        | 20249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 168      |
|    fps              | 1116     |
|    time_elapsed     | 75       |
|    total_timesteps  | 84000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00634  |
|    n_updates        | 20749    |
----------------------------------
Eval num_timesteps=85000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 85000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00367  |
|    n_updates        | 20999    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 172      |
|    fps              | 1107     |
|    time_elapsed     | 77       |
|    total_timesteps  | 86000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0053   |
|    n_updates        | 21249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 176      |
|    fps              | 1110     |
|    time_elapsed     | 79       |
|    total_timesteps  | 88000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00555  |
|    n_updates        | 21749    |
----------------------------------
Eval num_timesteps=90000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 90000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00797  |
|    n_updates        | 22249    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 180      |
|    fps              | 1102     |
|    time_elapsed     | 81       |
|    total_timesteps  | 90000    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 184      |
|    fps              | 1105     |
|    time_elapsed     | 83       |
|    total_timesteps  | 92000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0276   |
|    n_updates        | 22749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 188      |
|    fps              | 1108     |
|    time_elapsed     | 84       |
|    total_timesteps  | 94000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00325  |
|    n_updates        | 23249    |
----------------------------------
Eval num_timesteps=95000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 95000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00715  |
|    n_updates        | 23499    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 192      |
|    fps              | 1100     |
|    time_elapsed     | 87       |
|    total_timesteps  | 96000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00476  |
|    n_updates        | 23749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 196      |
|    fps              | 1103     |
|    time_elapsed     | 88       |
|    total_timesteps  | 98000    |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.00482  |
|    n_updates        | 24249    |
----------------------------------
Eval num_timesteps=100000, episode_reward=-50.00 +/- 0.00
Episode length: 500.00 +/- 0.00
----------------------------------
| eval/               |          |
|    mean_ep_length   | 500      |
|    mean_reward      | -50      |
| rollout/            |          |
|    exploration_rate | 0.05     |
| time/               |          |
|    total_timesteps  | 100000   |
| train/              |          |
|    learning_rate    | 0.0003   |
|    loss             | 0.0929   |
|    n_updates        | 24749    |
----------------------------------
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 500      |
|    ep_rew_mean      | -50      |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 200      |
|    fps              | 1096     |
|    time_elapsed     | 91       |
|    total_timesteps  | 100000   |
----------------------------------
2025-05-02 09:48:01,321 - core.model_training - INFO - Training completed in 0:01:32.810488
2025-05-02 09:48:01,326 - core.model_training - INFO - Model saved to: models/database_cache_model_dqn_cpu_10_20250502_094801
2025-05-02 09:48:01,327 - api.app_utils - INFO - Evaluating model at models/database_cache_model_dqn_cpu_10_20250502_094801
2025-05-02 09:48:01,327 - core.model_training - INFO - Evaluating model on CPU
2025-05-02 09:48:01,328 - core.model_training - INFO - Loaded feature columns from metadata: ['id', 'cache_name', 'cache_key', 'hit_ratio', 'item_count', 'load_time_ms', 'policy_triggered', 'rl_action_taken', 'size_bytes', 'timestamp', 'traffic_intensity']
2025-05-02 09:48:01,341 - core.model_training - INFO - Model loaded successfully on cpu!
2025-05-02 09:48:01,341 - core.cache_environment - INFO - Connecting to database: mysql+mysqlconnector://cacheuser:cachepass@192.168.1.81:3306/cache_db
2025-05-02 09:48:01,372 - database.database_connection - INFO - Successfully connected to database.
2025-05-02 09:48:01,373 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:48:01,374 - core.cache_environment - INFO - Available tables: aggregated_production, cache_metrics, carbon_intensity, comparison_analysis, consumption_data, consumption_forecast, energy_data, exchange_data, production_data, rl_models
2025-05-02 09:48:01,375 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:48:01,376 - core.cache_environment - INFO - Using specified table: cache_metrics
2025-05-02 09:48:01,377 - database.database_connection - INFO - Found 10 tables in database
2025-05-02 09:48:01,379 - core.cache_environment - INFO - Loading data from table: cache_metrics
2025-05-02 09:48:01,389 - core.cache_environment - INFO - Loaded 1027 rows from cache_metrics
2025-05-02 09:48:01,424 - core.model_training - INFO - Step 100, hit rate: 0.0000, avg inference: 0.15ms
2025-05-02 09:48:01,463 - core.model_training - INFO - Step 200, hit rate: 0.0000, avg inference: 0.23ms
2025-05-02 09:48:01,500 - core.model_training - INFO - Step 300, hit rate: 0.0000, avg inference: 0.12ms
2025-05-02 09:48:01,538 - core.model_training - INFO - Step 400, hit rate: 0.0000, avg inference: 0.17ms
2025-05-02 09:48:01,571 - core.model_training - INFO - Step 500, hit rate: 0.0000, avg inference: 0.09ms
2025-05-02 09:48:01,605 - core.model_training - INFO - Step 600, hit rate: 0.0000, avg inference: 0.09ms
2025-05-02 09:48:01,640 - core.model_training - INFO - Step 700, hit rate: 0.0000, avg inference: 0.18ms
2025-05-02 09:48:01,673 - core.model_training - INFO - Step 800, hit rate: 0.0000, avg inference: 0.15ms
2025-05-02 09:48:01,706 - core.model_training - INFO - Step 900, hit rate: 0.0000, avg inference: 0.18ms
2025-05-02 09:48:01,741 - core.model_training - INFO - Step 1000, hit rate: 0.0000, avg inference: 0.12ms
2025-05-02 09:48:01,741 - core.model_training - INFO - Evaluation completed in 0.35 seconds
2025-05-02 09:48:01,747 - api.app_utils - INFO - Training job 7c9426ad-4c80-4b5d-b1b3-664924a8c8e7 completed successfully. Model saved at models/database_cache_model_dqn_cpu_10_20250502_094801
2025-05-02 09:48:01,748 - api.app_utils - INFO - Evaluation results: {'hit_rates': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'hit_history': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'rewards': [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1], 'moving_hit_rates': [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)], 'final_hit_rate': 0.0, 'total_reward': -99.9999999999986, 'avg_inference_time_ms': 0.1452839999999999, 'evaluation_time_seconds': 0.352402, 'device_used': 'cpu', 'step_reasoning': ['Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).', 'Cache MISS: item not in cache, not added (action=0).'], 'in_cache': [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], 'urls_hit': ['/exchange?exchangeCountry=NORWAY', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/comparison?priceArea=DK1&productionType=CENTRAL_POWER&comparisonType=CONSUMPTION&exchangeCountry=NETHERLANDS', '/production?priceArea=DK2&productionType=WIND', '/carbon-intensity?priceArea=DK2&productionType=HYDRO&aggregationType=YEARLY', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=WIND', '/comparison?priceArea=DK2&productionType=COMMERCIAL_POWER&comparisonType=EXCHANGE&exchangeCountry=SWEDEN', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=YEARLY', '/data?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/production?priceArea=DK2&productionType=WIND', '/data?priceArea=DK1', '/exchange?exchangeCountry=GREATBRITAIN', '/production?priceArea=DK1&productionType=WIND', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=SWEDEN', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=HYDRO', '/forecast?priceArea=DK2&horizon=427', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=NETHERLANDS', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=GREATBRITAIN', '/production?priceArea=DK1&productionType=SOLAR', '/data?priceArea=DK1', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=SOLAR', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=WEEKLY', '/comparison?priceArea=DK2&productionType=WIND&comparisonType=CONSUMPTION&exchangeCountry=GREATBRITAIN', '/exchange?exchangeCountry=NETHERLANDS', '/carbon-intensity?priceArea=DK1&productionType=CENTRAL_POWER&aggregationType=WEEKLY', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=NETHERLANDS', '/consumption?priceArea=DK2', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=DAILY', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=CENTRAL_POWER&aggregationType=HOURLY', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/forecast?priceArea=DK2&horizon=440', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=MONTHLY', '/production?priceArea=DK2&productionType=WIND', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/comparison?priceArea=DK1&productionType=SOLAR&comparisonType=PRODUCTION&exchangeCountry=GREATBRITAIN', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=HYDRO', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=SOLAR', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=YEARLY', '/production?priceArea=DK1&productionType=SOLAR', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=YEARLY', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/carbon-intensity?priceArea=DK2&productionType=HYDRO&aggregationType=YEARLY', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=CONSUMPTION&exchangeCountry=GERMANY', '/exchange?exchangeCountry=SWEDEN', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/comparison?priceArea=DK1&productionType=SOLAR&comparisonType=PRODUCTION&exchangeCountry=NETHERLANDS', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=WIND', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=SOLAR', '/forecast?priceArea=DK2&horizon=427', '/data?priceArea=DK2', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=DAILY', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=HYDRO', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/exchange?exchangeCountry=NETHERLANDS', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/comparison?priceArea=DK1&productionType=SOLAR&comparisonType=PRODUCTION&exchangeCountry=NORWAY', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/carbon-intensity?priceArea=DK1&productionType=COMMERCIAL_POWER&aggregationType=HOURLY', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK2&productionType=HYDRO', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=YEARLY', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=WIND', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=SOLAR', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=HYDRO', '/exchange?exchangeCountry=NORWAY', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=GREATBRITAIN', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/forecast?priceArea=DK2&horizon=703', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=SOLAR', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=PRODUCTION&exchangeCountry=NORWAY', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=WIND', '/consumption?priceArea=DK2', '/comparison?priceArea=DK2&productionType=CENTRAL_POWER&comparisonType=CONSUMPTION&exchangeCountry=NORWAY', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/carbon-intensity?priceArea=DK1&productionType=CENTRAL_POWER&aggregationType=HOURLY', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=SOLAR', '/carbon-intensity?priceArea=DK1&productionType=CENTRAL_POWER&aggregationType=YEARLY', '/exchange?exchangeCountry=NETHERLANDS', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=PRODUCTION&exchangeCountry=SWEDEN', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=WIND', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=SOLAR', '/consumption?priceArea=DK2', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=YEARLY', '/forecast?priceArea=DK1&horizon=382', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=WEEKLY', '/exchange?exchangeCountry=NETHERLANDS', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=SWEDEN', '/production?priceArea=DK2&productionType=WIND', '/carbon-intensity?priceArea=DK2&productionType=CENTRAL_POWER&aggregationType=HOURLY', '/forecast?priceArea=DK1&horizon=282', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=EXCHANGE&exchangeCountry=GERMANY', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=EXCHANGE&exchangeCountry=SWEDEN', '/production?priceArea=DK1&productionType=SOLAR', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=DAILY', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=DAILY', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=SOLAR', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/forecast?priceArea=DK1&horizon=527', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=PRODUCTION&exchangeCountry=SWEDEN', '/forecast?priceArea=DK1&horizon=267', '/carbon-intensity?priceArea=DK2&productionType=CENTRAL_POWER&aggregationType=HOURLY', '/data?priceArea=DK1', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/forecast?priceArea=DK2&horizon=364', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=WIND', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=HYDRO', '/forecast?priceArea=DK1&horizon=63', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=WIND', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=HYDRO', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=SOLAR', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=SOLAR', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=GERMANY', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=EXCHANGE&exchangeCountry=GERMANY', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=NETHERLANDS', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=PRODUCTION&exchangeCountry=NORWAY', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/comparison?priceArea=DK2&productionType=WIND&comparisonType=CONSUMPTION&exchangeCountry=SWEDEN', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/comparison?priceArea=DK2&productionType=WIND&comparisonType=EXCHANGE&exchangeCountry=GERMANY', '/data?priceArea=DK1', '/exchange?exchangeCountry=NORWAY', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=DAILY', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/exchange?exchangeCountry=NORWAY', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=CONSUMPTION&exchangeCountry=GREATBRITAIN', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=HOURLY', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=WIND', '/comparison?priceArea=DK1&productionType=COMMERCIAL_POWER&comparisonType=PRODUCTION&exchangeCountry=GREATBRITAIN', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=COMMERCIAL_POWER', '/forecast?priceArea=DK2&horizon=266', '/production?priceArea=DK1&productionType=SOLAR', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=WIND', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=WEEKLY', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=WIND', '/consumption?priceArea=DK1', '/forecast?priceArea=DK1&horizon=663', '/consumption?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/carbon-intensity?priceArea=DK2&productionType=HYDRO&aggregationType=WEEKLY', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=DAILY', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=HYDRO', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=SOLAR', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=WEEKLY', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=WIND', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK2', '/data?priceArea=DK2', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK1', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=SOLAR', '/production?priceArea=DK1&productionType=SOLAR', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=WEEKLY', '/data?priceArea=DK2', '/comparison?priceArea=DK2&productionType=COMMERCIAL_POWER&comparisonType=CONSUMPTION&exchangeCountry=NETHERLANDS', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/forecast?priceArea=DK1&horizon=541', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=WIND', '/consumption?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/carbon-intensity?priceArea=DK2&productionType=WIND&aggregationType=MONTHLY', '/forecast?priceArea=DK2&horizon=628', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/forecast?priceArea=DK2&horizon=632', '/data?priceArea=DK2', '/forecast?priceArea=DK2&horizon=505', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/comparison?priceArea=DK2&productionType=COMMERCIAL_POWER&comparisonType=CONSUMPTION&exchangeCountry=GERMANY', '/exchange?exchangeCountry=GERMANY', '/production?priceArea=DK2&productionType=WIND', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=HYDRO', '/data?priceArea=DK2', '/data?priceArea=DK1', '/exchange?exchangeCountry=NETHERLANDS', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=NETHERLANDS', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=WIND', '/data?priceArea=DK2', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/exchange?exchangeCountry=GERMANY', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/forecast?priceArea=DK1&horizon=549', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=WEEKLY', '/exchange?exchangeCountry=NETHERLANDS', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=SOLAR', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=GERMANY', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK1', '/forecast?priceArea=DK1&horizon=65', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=WIND', '/exchange?exchangeCountry=NORWAY', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=GERMANY', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=WIND', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=HOURLY', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=PRODUCTION&exchangeCountry=NORWAY', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/exchange?exchangeCountry=NETHERLANDS', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=WIND', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/comparison?priceArea=DK1&productionType=COMMERCIAL_POWER&comparisonType=EXCHANGE&exchangeCountry=GREATBRITAIN', '/production?priceArea=DK1&productionType=HYDRO', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=SWEDEN', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK1&productionType=SOLAR', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=MONTHLY', '/production?priceArea=DK2&productionType=SOLAR', '/comparison?priceArea=DK1&productionType=COMMERCIAL_POWER&comparisonType=PRODUCTION&exchangeCountry=SWEDEN', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=DAILY', '/comparison?priceArea=DK1&productionType=COMMERCIAL_POWER&comparisonType=EXCHANGE&exchangeCountry=NETHERLANDS', '/data?priceArea=DK2', '/exchange?exchangeCountry=NETHERLANDS', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=HYDRO', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=YEARLY', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=HYDRO', '/forecast?priceArea=DK2&horizon=61', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=WIND', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=WIND', '/consumption?priceArea=DK1', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=PRODUCTION&exchangeCountry=GREATBRITAIN', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=SOLAR', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=COMMERCIAL_POWER', '/comparison?priceArea=DK2&productionType=COMMERCIAL_POWER&comparisonType=PRODUCTION&exchangeCountry=NORWAY', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=WIND', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=PRODUCTION&exchangeCountry=GREATBRITAIN', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/comparison?priceArea=DK1&productionType=CENTRAL_POWER&comparisonType=EXCHANGE&exchangeCountry=SWEDEN', '/forecast?priceArea=DK1&horizon=675', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=YEARLY', '/consumption?priceArea=DK2', '/forecast?priceArea=DK1&horizon=333', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=SOLAR', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=HYDRO', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/exchange?exchangeCountry=NETHERLANDS', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=WIND', '/consumption?priceArea=DK2', '/forecast?priceArea=DK2&horizon=586', '/data?priceArea=DK2', '/forecast?priceArea=DK1&horizon=363', '/production?priceArea=DK1&productionType=SOLAR', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=SWEDEN', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK1', '/exchange?exchangeCountry=SWEDEN', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=NETHERLANDS', '/exchange?exchangeCountry=NORWAY', '/exchange?exchangeCountry=NORWAY', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK1&productionType=HYDRO', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=YEARLY', '/comparison?priceArea=DK2&productionType=CENTRAL_POWER&comparisonType=CONSUMPTION&exchangeCountry=NETHERLANDS', '/exchange?exchangeCountry=NORWAY', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=HYDRO', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=WIND', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/data?priceArea=DK2', '/exchange?exchangeCountry=NORWAY', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=SOLAR', '/production?priceArea=DK1&productionType=HYDRO', '/exchange?exchangeCountry=NETHERLANDS', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=WIND', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/forecast?priceArea=DK1&horizon=439', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=HYDRO', '/exchange?exchangeCountry=NETHERLANDS', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=SOLAR', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=WIND', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=COMMERCIAL_POWER', '/forecast?priceArea=DK2&horizon=554', '/production?priceArea=DK2&productionType=WIND', '/exchange?exchangeCountry=GERMANY', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/exchange?exchangeCountry=NORWAY', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=DAILY', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=WIND', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=GERMANY', '/data?priceArea=DK2', '/exchange?exchangeCountry=NORWAY', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=SWEDEN', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/forecast?priceArea=DK1&horizon=666', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=HYDRO', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=WIND', '/production?priceArea=DK1&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=SOLAR', '/consumption?priceArea=DK2', '/forecast?priceArea=DK2&horizon=27', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=SOLAR', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/forecast?priceArea=DK2&horizon=367', '/data?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/carbon-intensity?priceArea=DK1&productionType=COMMERCIAL_POWER&aggregationType=HOURLY', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/comparison?priceArea=DK2&productionType=WIND&comparisonType=EXCHANGE&exchangeCountry=NETHERLANDS', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=MONTHLY', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=HOURLY', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=DAILY', '/exchange?exchangeCountry=GERMANY', '/forecast?priceArea=DK1&horizon=227', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=HYDRO', '/exchange?exchangeCountry=SWEDEN', '/production?priceArea=DK1&productionType=SOLAR', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=HYDRO', '/comparison?priceArea=DK1&productionType=SOLAR&comparisonType=EXCHANGE&exchangeCountry=GERMANY', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=PRODUCTION&exchangeCountry=SWEDEN', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=WIND', '/production?priceArea=DK1&productionType=HYDRO', '/exchange?exchangeCountry=GERMANY', '/exchange?exchangeCountry=GERMANY', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=COMMERCIAL_POWER', '/carbon-intensity?priceArea=DK2&productionType=CENTRAL_POWER&aggregationType=MONTHLY', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=HYDRO', '/production?priceArea=DK2&productionType=HYDRO', '/production?priceArea=DK2&productionType=SOLAR', '/comparison?priceArea=DK2&productionType=CENTRAL_POWER&comparisonType=PRODUCTION&exchangeCountry=GERMANY', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/forecast?priceArea=DK2&horizon=641', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=SOLAR', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=WEEKLY', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=HYDRO', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=HYDRO', '/carbon-intensity?priceArea=DK1&productionType=COMMERCIAL_POWER&aggregationType=MONTHLY', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=HYDRO', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/exchange?exchangeCountry=NETHERLANDS', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/exchange?exchangeCountry=SWEDEN', '/data?priceArea=DK1', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/comparison?priceArea=DK1&productionType=WIND&comparisonType=EXCHANGE&exchangeCountry=GERMANY', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=NORWAY', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/carbon-intensity?priceArea=DK1&productionType=CENTRAL_POWER&aggregationType=DAILY', '/carbon-intensity?priceArea=DK2&productionType=SOLAR&aggregationType=DAILY', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=WIND', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK2&productionType=CENTRAL_POWER&aggregationType=YEARLY', '/consumption?priceArea=DK2', '/forecast?priceArea=DK2&horizon=97', '/aggregated-production?priceArea=DK2&aggregationType=HOURLY&productionType=WIND', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/forecast?priceArea=DK2&horizon=104', '/production?priceArea=DK2&productionType=HYDRO', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=SOLAR', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/production?priceArea=DK2&productionType=SOLAR', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/exchange?exchangeCountry=NETHERLANDS', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=CENTRAL_POWER', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=WEEKLY', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/comparison?priceArea=DK2&productionType=WIND&comparisonType=CONSUMPTION&exchangeCountry=NORWAY', '/forecast?priceArea=DK1&horizon=217', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/comparison?priceArea=DK1&productionType=CENTRAL_POWER&comparisonType=PRODUCTION&exchangeCountry=GERMANY', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=SOLAR', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=SOLAR', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=SOLAR', '/carbon-intensity?priceArea=DK1&productionType=COMMERCIAL_POWER&aggregationType=MONTHLY', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=COMMERCIAL_POWER', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=GERMANY', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=DAILY', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=HYDRO', '/data?priceArea=DK1', '/exchange?exchangeCountry=GREATBRITAIN', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=WIND', '/production?priceArea=DK2&productionType=HYDRO', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/production?priceArea=DK1&productionType=WIND', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/forecast?priceArea=DK1&horizon=283', '/comparison?priceArea=DK1&productionType=HYDRO&comparisonType=CONSUMPTION&exchangeCountry=NORWAY', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=WIND', '/production?priceArea=DK1&productionType=HYDRO', '/data?priceArea=DK2', '/data?priceArea=DK1', '/carbon-intensity?priceArea=DK2&productionType=HYDRO&aggregationType=WEEKLY', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=CONSUMPTION&exchangeCountry=GERMANY', '/forecast?priceArea=DK1&horizon=20', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=HYDRO', '/production?priceArea=DK2&productionType=HYDRO', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=SOLAR', '/exchange?exchangeCountry=GERMANY', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=WIND', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=HYDRO', '/aggregated-production?priceArea=DK1&aggregationType=DAILY&productionType=HYDRO', '/exchange?exchangeCountry=SWEDEN', '/data?priceArea=DK1', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=SOLAR', '/data?priceArea=DK2', '/production?priceArea=DK1&productionType=WIND', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK1', '/comparison?priceArea=DK1&productionType=COMMERCIAL_POWER&comparisonType=PRODUCTION&exchangeCountry=GERMANY', '/production?priceArea=DK1&productionType=WIND', '/carbon-intensity?priceArea=DK1&productionType=SOLAR&aggregationType=MONTHLY', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=HYDRO', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=WIND', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=HYDRO', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=WIND', '/consumption?priceArea=DK2', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=COMMERCIAL_POWER', '/data?priceArea=DK2', '/exchange?exchangeCountry=GERMANY', '/forecast?priceArea=DK2&horizon=243', '/forecast?priceArea=DK1&horizon=408', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=CONSUMPTION&exchangeCountry=NETHERLANDS', '/data?priceArea=DK2', '/forecast?priceArea=DK1&horizon=264', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=WIND', '/exchange?exchangeCountry=GREATBRITAIN', '/consumption?priceArea=DK1', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/forecast?priceArea=DK1&horizon=518', '/consumption?priceArea=DK1', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=SOLAR', '/forecast?priceArea=DK1&horizon=102', '/production?priceArea=DK1&productionType=WIND', '/consumption?priceArea=DK2', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK2', '/forecast?priceArea=DK2&horizon=453', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/exchange?exchangeCountry=GERMANY', '/data?priceArea=DK1', '/forecast?priceArea=DK2&horizon=700', '/data?priceArea=DK1', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/exchange?exchangeCountry=SWEDEN', '/consumption?priceArea=DK2', '/comparison?priceArea=DK1&productionType=CENTRAL_POWER&comparisonType=PRODUCTION&exchangeCountry=GERMANY', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=WEEKLY&productionType=COMMERCIAL_POWER', '/production?priceArea=DK2&productionType=SOLAR', '/data?priceArea=DK2', '/data?priceArea=DK2', '/data?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK1&aggregationType=HOURLY&productionType=HYDRO', '/aggregated-production?priceArea=DK1&aggregationType=MONTHLY&productionType=HYDRO', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=HYDRO', '/carbon-intensity?priceArea=DK2&productionType=COMMERCIAL_POWER&aggregationType=WEEKLY', '/exchange?exchangeCountry=NETHERLANDS', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=YEARLY', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=SOLAR', '/data?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/carbon-intensity?priceArea=DK1&productionType=HYDRO&aggregationType=YEARLY', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=SOLAR', '/exchange?exchangeCountry=NETHERLANDS', '/aggregated-production?priceArea=DK1&aggregationType=YEARLY&productionType=HYDRO', '/consumption?priceArea=DK1', '/consumption?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK1', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=COMMERCIAL_POWER', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=PRODUCTION&exchangeCountry=GERMANY', '/exchange?exchangeCountry=NORWAY', '/data?priceArea=DK2', '/data?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=SOLAR', '/consumption?priceArea=DK1', '/carbon-intensity?priceArea=DK1&productionType=WIND&aggregationType=DAILY', '/forecast?priceArea=DK2&horizon=297', '/aggregated-production?priceArea=DK2&aggregationType=YEARLY&productionType=CENTRAL_POWER', '/production?priceArea=DK1&productionType=WIND', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=GREATBRITAIN', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/consumption?priceArea=DK1', '/forecast?priceArea=DK2&horizon=510', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=SOLAR', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=SOLAR', '/consumption?priceArea=DK2', '/aggregated-production?priceArea=DK2&aggregationType=WEEKLY&productionType=WIND', '/forecast?priceArea=DK1&horizon=161', '/data?priceArea=DK2', '/data?priceArea=DK2', '/consumption?priceArea=DK2', '/data?priceArea=DK1', '/exchange?exchangeCountry=NETHERLANDS', '/data?priceArea=DK1', '/data?priceArea=DK2', '/data?priceArea=DK2', '/comparison?priceArea=DK2&productionType=HYDRO&comparisonType=EXCHANGE&exchangeCountry=SWEDEN', '/exchange?exchangeCountry=GREATBRITAIN', '/data?priceArea=DK1', '/data?priceArea=DK1', '/data?priceArea=DK2', '/comparison?priceArea=DK1&productionType=HYDRO&comparisonType=CONSUMPTION&exchangeCountry=GREATBRITAIN', '/production?priceArea=DK2&productionType=SOLAR', '/production?priceArea=DK1&productionType=SOLAR', '/exchange?exchangeCountry=GERMANY', '/consumption?priceArea=DK1', '/exchange?exchangeCountry=NETHERLANDS', '/production?priceArea=DK2&productionType=WIND', '/exchange?exchangeCountry=GREATBRITAIN', '/aggregated-production?priceArea=DK2&aggregationType=MONTHLY&productionType=CENTRAL_POWER', '/consumption?priceArea=DK2', '/consumption?priceArea=DK1', '/forecast?priceArea=DK2&horizon=511', '/carbon-intensity?priceArea=DK2&productionType=HYDRO&aggregationType=WEEKLY', '/data?priceArea=DK1', '/forecast?priceArea=DK2&horizon=76', '/consumption?priceArea=DK1', '/production?priceArea=DK1&productionType=HYDRO', '/consumption?priceArea=DK2', '/production?priceArea=DK2&productionType=HYDRO', '/consumption?priceArea=DK1', '/production?priceArea=DK2&productionType=COMMERCIAL_POWER', '/production?priceArea=DK1&productionType=HYDRO', '/aggregated-production?priceArea=DK2&aggregationType=DAILY&productionType=SOLAR', '/comparison?priceArea=DK2&productionType=SOLAR&comparisonType=CONSUMPTION&exchangeCountry=GERMANY', '/data?priceArea=DK2', '/carbon-intensity?priceArea=DK2&productionType=WIND&aggregationType=HOURLY', '/production?priceArea=DK1&productionType=CENTRAL_POWER', '/production?priceArea=DK2&productionType=CENTRAL_POWER', '/carbon-intensity?priceArea=DK2&productionType=WIND&aggregationType=MONTHLY', '/data?priceArea=DK1', '/data?priceArea=DK1', '/forecast?priceArea=DK2&horizon=20', '/exchange?exchangeCountry=SWEDEN']}
2025-05-02 09:48:02,195 - core.visualization - INFO - Cache visualization saved to cache_eval_results\cache_performance_20250502_094801.png
