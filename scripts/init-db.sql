-- RAGFlow Production Database Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE processing_status AS ENUM ('pending', 'processing', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE chunk_type AS ENUM ('text', 'table', 'image', 'header', 'footer');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Create indexes for better performance
-- These will be created by Alembic migrations, but we can prepare the database

-- Create monitoring user for health checks
DO $$ BEGIN
    CREATE USER ragflow_monitor WITH PASSWORD 'monitor_password';
    GRANT CONNECT ON DATABASE ragflow_prod TO ragflow_monitor;
    GRANT USAGE ON SCHEMA public TO ragflow_monitor;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO ragflow_monitor;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ragflow_monitor;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create backup user
DO $$ BEGIN
    CREATE USER ragflow_backup WITH PASSWORD 'backup_password';
    GRANT CONNECT ON DATABASE ragflow_prod TO ragflow_backup;
    GRANT USAGE ON SCHEMA public TO ragflow_backup;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO ragflow_backup;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ragflow_backup;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create function for generating correlation IDs
CREATE OR REPLACE FUNCTION generate_correlation_id()
RETURNS TEXT AS $$
BEGIN
    RETURN 'corr_' || replace(uuid_generate_v4()::text, '-', '');
END;
$$ language 'plpgsql';

-- Create function for text search ranking
CREATE OR REPLACE FUNCTION calculate_search_rank(
    title_vector tsvector,
    content_vector tsvector,
    query tsquery,
    title_weight float DEFAULT 1.0,
    content_weight float DEFAULT 0.4
)
RETURNS float AS $$
BEGIN
    RETURN (
        COALESCE(ts_rank(title_vector, query), 0) * title_weight +
        COALESCE(ts_rank(content_vector, query), 0) * content_weight
    );
END;
$$ language 'plpgsql';

-- Create materialized view for document statistics (will be created by migrations)
-- This is just a placeholder for the structure

-- Set up logging for performance monitoring
CREATE TABLE IF NOT EXISTS query_performance_log (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    execution_time INTERVAL,
    rows_returned INTEGER,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on query performance log
CREATE INDEX IF NOT EXISTS idx_query_performance_log_executed_at 
ON query_performance_log(executed_at);

-- Create function to log slow queries
CREATE OR REPLACE FUNCTION log_slow_query()
RETURNS event_trigger AS $$
DECLARE
    r RECORD;
BEGIN
    -- This would be implemented to log slow queries
    -- For now, it's a placeholder
    NULL;
END;
$$ language 'plpgsql';

-- Set up connection limits and timeouts
ALTER DATABASE ragflow_prod SET statement_timeout = '300s';
ALTER DATABASE ragflow_prod SET idle_in_transaction_session_timeout = '600s';
ALTER DATABASE ragflow_prod SET lock_timeout = '30s';

-- Create schema for audit logging
CREATE SCHEMA IF NOT EXISTS audit;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.activity_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    correlation_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on audit log
CREATE INDEX IF NOT EXISTS idx_audit_activity_log_user_id 
ON audit.activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_activity_log_created_at 
ON audit.activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_activity_log_correlation_id 
ON audit.activity_log(correlation_id);

-- Create function for audit logging
CREATE OR REPLACE FUNCTION audit.log_activity(
    p_user_id VARCHAR(255),
    p_action VARCHAR(100),
    p_table_name VARCHAR(100),
    p_record_id VARCHAR(255),
    p_old_values JSONB DEFAULT NULL,
    p_new_values JSONB DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_correlation_id VARCHAR(255) DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO audit.activity_log (
        user_id, action, table_name, record_id, 
        old_values, new_values, ip_address, user_agent, correlation_id
    ) VALUES (
        p_user_id, p_action, p_table_name, p_record_id,
        p_old_values, p_new_values, p_ip_address, p_user_agent, p_correlation_id
    );
END;
$$ language 'plpgsql';

-- Grant permissions for audit schema
GRANT USAGE ON SCHEMA audit TO ragflow_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO ragflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT, INSERT ON TABLES TO ragflow_user;

-- Create maintenance procedures
CREATE OR REPLACE FUNCTION maintenance.cleanup_old_logs(days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit.activity_log 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    DELETE FROM query_performance_log 
    WHERE executed_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Create schema for maintenance functions
CREATE SCHEMA IF NOT EXISTS maintenance;

-- Grant permissions for maintenance schema
GRANT USAGE ON SCHEMA maintenance TO ragflow_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA maintenance TO ragflow_user;

-- Create backup and restore functions
CREATE OR REPLACE FUNCTION maintenance.create_backup_info()
RETURNS TABLE(
    backup_timestamp TIMESTAMP,
    database_size TEXT,
    table_count INTEGER,
    total_rows BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CURRENT_TIMESTAMP as backup_timestamp,
        pg_size_pretty(pg_database_size(current_database())) as database_size,
        COUNT(*)::INTEGER as table_count,
        SUM(n_tup_ins + n_tup_upd)::BIGINT as total_rows
    FROM pg_stat_user_tables;
END;
$$ language 'plpgsql';

-- Set up automatic vacuum and analyze
ALTER TABLE audit.activity_log SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- Create health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE(
    component TEXT,
    status TEXT,
    details JSONB
) AS $$
BEGIN
    -- Database connection check
    RETURN QUERY
    SELECT 
        'database'::TEXT as component,
        'healthy'::TEXT as status,
        jsonb_build_object(
            'connections', (SELECT count(*) FROM pg_stat_activity),
            'max_connections', (SELECT setting FROM pg_settings WHERE name = 'max_connections'),
            'database_size', pg_size_pretty(pg_database_size(current_database()))
        ) as details;
    
    -- Table statistics
    RETURN QUERY
    SELECT 
        'tables'::TEXT as component,
        'healthy'::TEXT as status,
        jsonb_build_object(
            'table_count', (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'),
            'total_size', pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))::bigint)
        ) as details
    FROM pg_tables WHERE schemaname = 'public';
END;
$$ language 'plpgsql';

-- Final message
DO $$ BEGIN
    RAISE NOTICE 'RAGFlow production database initialization completed successfully';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'Version: %', version();
    RAISE NOTICE 'Extensions created: uuid-ossp, pg_trgm, btree_gin';
    RAISE NOTICE 'Schemas created: audit, maintenance';
    RAISE NOTICE 'Users created: ragflow_monitor, ragflow_backup';
END $$;