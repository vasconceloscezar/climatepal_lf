-- Create the search_configs table
CREATE TABLE IF NOT EXISTS search_configs (
    id UUID PRIMARY KEY NOT NULL,
    run_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    url TEXT,
    temporal_resolution TEXT,
    date_range TEXT,
    mip TEXT,
    experiment TEXT,
    variable TEXT,
    user_query TEXT
);

-- Create the sentiment_feedback table
CREATE TABLE IF NOT EXISTS sentiment_feedback (
    id UUID PRIMARY KEY NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    search_config_run_id UUID NOT NULL,
    search_result_sentiment TEXT,
    variable_sentiment TEXT,
    experiment_sentiment TEXT,
    mip_sentiment TEXT,
    date_range_sentiment TEXT,
    temporal_resolution_sentiment TEXT,
    url_sentiment TEXT,
    extra_feedback TEXT,
        FOREIGN KEY (search_config_run_id) REFERENCES search_configs(id)
);