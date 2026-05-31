"""Consolidated initial schema — full end-state of fren v3 (migrations 001..049).

This single migration reproduces the cumulative HEAD schema of
fren_infrastructure_control_v3 as it stood after applying all 49 migrations.
All cross-migration ALTERs (added/dropped columns, nullability changes,
new constraints/indexes) have been folded into the final CREATE TABLE
definitions so the result is the schema as it stands at v3 HEAD.

Style follows v3: raw `op.execute(...)` SQL statements grouped per table.

Revision ID: 001_initial_schema
Revises: None
"""

from collections.abc import Sequence

from alembic import op

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ══════════════════════════════════════════════════════════════
    # Extensions
    # ══════════════════════════════════════════════════════════════
    # pgvector — required for vector(1536) embedding columns and hnsw indexes.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ══════════════════════════════════════════════════════════════
    # goals — hierarchical goal tree (levels 1..6) with parent/child links
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id SERIAL PRIMARY KEY,
            goal_id VARCHAR(100) UNIQUE NOT NULL,
            level SMALLINT NOT NULL CHECK (level BETWEEN 1 AND 6),
            title VARCHAR(500) NOT NULL,
            description TEXT,
            parent_goal_id VARCHAR(100),
            child_goal_ids JSONB DEFAULT '[]',
            status VARCHAR(20) DEFAULT 'active',
            priority VARCHAR(20) DEFAULT 'medium',
            progress_percent SMALLINT DEFAULT 0,
            deadline TIMESTAMPTZ,
            date DATE,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_goals_goal_id ON goals(goal_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_goals_level ON goals(level)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_goal_id)")

    # ══════════════════════════════════════════════════════════════
    # todos — task items (incl. updated_at[003], url[017], dependencies[026])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id SERIAL PRIMARY KEY,
            todo_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            status VARCHAR(20) DEFAULT 'pending',
            priority VARCHAR(20) DEFAULT 'medium',
            source VARCHAR(20) DEFAULT 'user',
            source_metadata JSONB DEFAULT '{}',
            deadline TIMESTAMPTZ,
            estimated_minutes INTEGER,
            category VARCHAR(50) DEFAULT 'personal',
            tags JSONB DEFAULT '[]',
            linked_goal_id VARCHAR(100) REFERENCES goals(goal_id) ON DELETE SET NULL,
            goal_alignment JSONB,
            subtasks JSONB DEFAULT '[]',
            date DATE NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            url TEXT,
            dependencies JSONB DEFAULT '[]'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_todos_todo_id ON todos(todo_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_todos_date ON todos(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_todos_linked_goal ON todos(linked_goal_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_todos_dependencies ON todos USING GIN (dependencies)")

    # ══════════════════════════════════════════════════════════════
    # daily_strategies — per-day focus plan / time blocks
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS daily_strategies (
            id SERIAL PRIMARY KEY,
            strategy_id VARCHAR(100) UNIQUE NOT NULL,
            date DATE NOT NULL,
            focus_goals JSONB DEFAULT '[]',
            time_blocks JSONB DEFAULT '[]',
            notes TEXT,
            status VARCHAR(20) DEFAULT 'active',
            completion_summary JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_daily_strategies_strategy_id ON daily_strategies(strategy_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_daily_strategies_date ON daily_strategies(date)")

    # ══════════════════════════════════════════════════════════════
    # influence_attempts — agent persuasion attempts (campaign_id added[023])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS influence_attempts (
            id SERIAL PRIMARY KEY,
            attempt_id VARCHAR(100) UNIQUE NOT NULL,
            strategy_id VARCHAR(100) REFERENCES daily_strategies(strategy_id) ON DELETE SET NULL,
            goal_id VARCHAR(100) REFERENCES goals(goal_id) ON DELETE SET NULL,
            influence_type VARCHAR(50) NOT NULL,
            message_sent TEXT NOT NULL,
            assumptions JSONB DEFAULT '[]',
            expected_outcome TEXT,
            actual_outcome TEXT,
            effectiveness_score NUMERIC(3,2),
            date DATE NOT NULL,
            sent_at TIMESTAMPTZ NOT NULL,
            evaluated_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_influence_attempts_attempt_id ON influence_attempts(attempt_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_influence_attempts_date ON influence_attempts(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_influence_attempts_strategy ON influence_attempts(strategy_id)")
    # NOTE: campaign_id column + FK to nudge_campaigns is added below, after
    # nudge_campaigns is created (see migration 023). The FK target must exist first.

    # ══════════════════════════════════════════════════════════════
    # validations — validated outcomes of influence attempts
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS validations (
            id SERIAL PRIMARY KEY,
            validation_id VARCHAR(100) UNIQUE NOT NULL,
            attempt_id VARCHAR(100) REFERENCES influence_attempts(attempt_id) ON DELETE SET NULL,
            approach_type VARCHAR(50) NOT NULL,
            validated BOOLEAN NOT NULL,
            effectiveness NUMERIC(3,2) NOT NULL,
            assumptions_tested JSONB DEFAULT '[]',
            conditions_for_success JSONB DEFAULT '[]',
            notes TEXT,
            date DATE NOT NULL,
            validation_category VARCHAR(20) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_validations_validation_id ON validations(validation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_validations_date ON validations(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_validations_attempt ON validations(attempt_id)")

    # ══════════════════════════════════════════════════════════════
    # monthly_conclusions — monthly rollups of validation effectiveness
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS monthly_conclusions (
            id SERIAL PRIMARY KEY,
            conclusion_id VARCHAR(100) UNIQUE NOT NULL,
            month VARCHAR(7) UNIQUE NOT NULL,
            total_validations INTEGER DEFAULT 0,
            validated_count INTEGER DEFAULT 0,
            invalidated_count INTEGER DEFAULT 0,
            approach_stats JSONB DEFAULT '{}',
            most_effective_approaches JSONB DEFAULT '[]',
            least_effective_approaches JSONB DEFAULT '[]',
            successful_conditions JSONB DEFAULT '{}',
            recommendations JSONB DEFAULT '[]',
            generated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_monthly_conclusions_month ON monthly_conclusions(month)")

    # ══════════════════════════════════════════════════════════════
    # commitments — commitments detected from chat, optionally → goals
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS commitments (
            id SERIAL PRIMARY KEY,
            commitment_id VARCHAR(100) UNIQUE NOT NULL,
            pattern_type VARCHAR(50) NOT NULL,
            commitment_text TEXT NOT NULL,
            confidence NUMERIC(3,2) NOT NULL,
            full_match TEXT,
            source_message TEXT,
            source VARCHAR(20) DEFAULT 'chat',
            message_timestamp TIMESTAMPTZ,
            status VARCHAR(30) DEFAULT 'pending',
            converted_goal_id VARCHAR(100) REFERENCES goals(goal_id) ON DELETE SET NULL,
            date DATE NOT NULL,
            detected_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_commitments_commitment_id ON commitments(commitment_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_commitments_date ON commitments(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status)")

    # ══════════════════════════════════════════════════════════════
    # cron_executions — cron run audit log (error_output added[031])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS cron_executions (
            id SERIAL PRIMARY KEY,
            execution_id VARCHAR(100) UNIQUE NOT NULL,
            mode VARCHAR(50) NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            duration_seconds NUMERIC(10,2),
            exit_code INTEGER,
            status VARCHAR(20) DEFAULT 'running',
            log_file TEXT,
            triggered_by VARCHAR(50) DEFAULT 'cron',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            error_output TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_cron_executions_execution_id ON cron_executions(execution_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cron_executions_started_at ON cron_executions(started_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cron_executions_mode ON cron_executions(mode)")

    # ══════════════════════════════════════════════════════════════
    # workflow_executions — workflow run audit log
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id SERIAL PRIMARY KEY,
            execution_id VARCHAR(100) UNIQUE NOT NULL,
            workflow_id VARCHAR(100) NOT NULL,
            workflow_name VARCHAR(255) NOT NULL,
            input_text TEXT,
            triggered_by VARCHAR(50) DEFAULT 'manual',
            status VARCHAR(20) DEFAULT 'running',
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            duration_seconds NUMERIC(10,2),
            exit_code INTEGER,
            output TEXT,
            error TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_execution_id ON workflow_executions(execution_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # chat_messages — full chat log (content_class/sfw_summary[004], embedding[014])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            timestamp_unix NUMERIC(16,6) NOT NULL,
            sender VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            chat_id VARCHAR(50),
            message_id BIGINT,
            username VARCHAR(100),
            metadata JSONB DEFAULT '{}',
            date DATE NOT NULL,
            content_class VARCHAR(10) DEFAULT 'public',
            sfw_summary TEXT,
            embedding vector(1536)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp_unix ON chat_messages(timestamp_unix DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_date ON chat_messages(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_sender ON chat_messages(sender)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_content_class ON chat_messages(content_class)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding ON chat_messages USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # subagent_logs — full prompt/output logs per subagent run
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS subagent_logs (
            id SERIAL PRIMARY KEY,
            log_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            agent VARCHAR(100) NOT NULL,
            prompt_full TEXT,
            output_full TEXT,
            duration_ms INTEGER DEFAULT 0,
            date DATE NOT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_subagent_logs_timestamp ON subagent_logs(timestamp DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_subagent_logs_agent ON subagent_logs(agent)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_subagent_logs_date ON subagent_logs(date)")

    # ══════════════════════════════════════════════════════════════
    # checker_state — singleton: periodic checker cooldowns (last_triggers[020])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS checker_state (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_reminder_at TIMESTAMPTZ,
            last_trigger_reason TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            last_triggers JSONB DEFAULT '{}'::jsonb
        )
    """)
    op.execute("INSERT INTO checker_state (id) VALUES (1) ON CONFLICT DO NOTHING")

    # ══════════════════════════════════════════════════════════════
    # recipes — saved cooking recipes
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id SERIAL PRIMARY KEY,
            recipe_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            original_title VARCHAR(500),
            original_language VARCHAR(10) DEFAULT 'en',
            description TEXT,
            ingredients JSONB DEFAULT '[]',
            instructions JSONB DEFAULT '[]',
            source_url TEXT,
            cuisine_type VARCHAR(50),
            meal_type VARCHAR(50),
            dietary_tags JSONB DEFAULT '[]',
            prep_time_minutes INTEGER,
            cook_time_minutes INTEGER,
            total_time_minutes INTEGER,
            servings INTEGER,
            difficulty VARCHAR(20) DEFAULT 'medium',
            nutrition JSONB DEFAULT '{}',
            image_url TEXT,
            times_made INTEGER DEFAULT 0,
            last_made_at TIMESTAMPTZ,
            rating NUMERIC(2,1),
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_recipes_recipe_id ON recipes(recipe_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_recipes_cuisine_type ON recipes(cuisine_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_recipes_meal_type ON recipes(meal_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_recipes_dietary_tags ON recipes USING GIN(dietary_tags)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_recipes_difficulty ON recipes(difficulty)")

    # ══════════════════════════════════════════════════════════════
    # restaurants — saved restaurants
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id SERIAL PRIMARY KEY,
            restaurant_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            cuisine_types JSONB DEFAULT '[]',
            location_area VARCHAR(100),
            address TEXT,
            phone VARCHAR(50),
            website TEXT,
            price_range VARCHAR(20),
            rating NUMERIC(2,1),
            notes TEXT,
            is_favorite BOOLEAN DEFAULT FALSE,
            last_visited_at TIMESTAMPTZ,
            visit_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_restaurant_id ON restaurants(restaurant_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_cuisine_types ON restaurants USING GIN(cuisine_types)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_location_area ON restaurants(location_area)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_price_range ON restaurants(price_range)")

    # ══════════════════════════════════════════════════════════════
    # dishes — restaurant menu dishes
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            id SERIAL PRIMARY KEY,
            dish_id VARCHAR(100) UNIQUE NOT NULL,
            restaurant_id VARCHAR(100) NOT NULL REFERENCES restaurants(restaurant_id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            price NUMERIC(10,2),
            category VARCHAR(50),
            dietary_tags JSONB DEFAULT '[]',
            is_favorite BOOLEAN DEFAULT FALSE,
            rating NUMERIC(2,1),
            times_ordered INTEGER DEFAULT 0,
            last_ordered_at TIMESTAMPTZ,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_dishes_dish_id ON dishes(dish_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_dishes_restaurant_id ON dishes(restaurant_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_dishes_category ON dishes(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_dishes_price ON dishes(price)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_dishes_dietary_tags ON dishes USING GIN(dietary_tags)")

    # ══════════════════════════════════════════════════════════════
    # user_food_preferences — singleton (current_location/delivery_services[029])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_food_preferences (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            dietary_restrictions JSONB DEFAULT '[]',
            favorite_cuisines JSONB DEFAULT '[]',
            disliked_ingredients JSONB DEFAULT '[]',
            allergies JSONB DEFAULT '[]',
            budget_preference VARCHAR(20) DEFAULT 'medium',
            spice_tolerance VARCHAR(20) DEFAULT 'medium',
            cooking_skill_level VARCHAR(20) DEFAULT 'intermediate',
            preferred_meal_prep_time INTEGER DEFAULT 60,
            kitchen_equipment JSONB DEFAULT '[]',
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            current_location VARCHAR(50) DEFAULT 'unknown',
            delivery_services JSONB DEFAULT '["lisek"]'
        )
    """)
    op.execute("INSERT INTO user_food_preferences (id) VALUES (1) ON CONFLICT DO NOTHING")

    # ══════════════════════════════════════════════════════════════
    # agent_notes — TTL'd key/value scratch space for agents
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_notes (
            id SERIAL PRIMARY KEY,
            note_key VARCHAR(100) UNIQUE NOT NULL,
            note_value JSONB NOT NULL DEFAULT '{}',
            expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_notes_key ON agent_notes(note_key)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_notes_expires ON agent_notes(expires_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_notes_key_prefix ON agent_notes(note_key varchar_pattern_ops)")

    # ══════════════════════════════════════════════════════════════
    # workflow_master_sessions — interactive workflow-builder sessions
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_master_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'active',
            pending_confirmation_id VARCHAR(100),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_wm_sessions_status ON workflow_master_sessions(status)")

    # ══════════════════════════════════════════════════════════════
    # workflow_master_messages — messages within a workflow-master session
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS workflow_master_messages (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            message_type VARCHAR(50) DEFAULT 'message',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT fk_wm_messages_session
                FOREIGN KEY (session_id)
                REFERENCES workflow_master_sessions(session_id)
                ON DELETE CASCADE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_wm_messages_session ON workflow_master_messages(session_id, created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_wm_messages_type ON workflow_master_messages(message_type)")

    # ══════════════════════════════════════════════════════════════
    # profile_categories — taxonomy for user-profile discoveries (seeded)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS profile_categories (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) UNIQUE NOT NULL,
            description TEXT,
            parent_category_id UUID REFERENCES profile_categories(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_categories_parent ON profile_categories(parent_category_id)")

    # ══════════════════════════════════════════════════════════════
    # profile_discoveries — confirmed user-profile facts
    #   (sensitivity/public_summary[004], embedding[014])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS profile_discoveries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category_id UUID REFERENCES profile_categories(id) ON DELETE SET NULL,
            discovery TEXT NOT NULL,
            confidence FLOAT DEFAULT 0.5,
            evidence_count INT DEFAULT 1,
            first_observed_at TIMESTAMPTZ,
            last_confirmed_at TIMESTAMPTZ,
            status VARCHAR(20) DEFAULT 'active',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            sensitivity VARCHAR(10) DEFAULT 'public',
            public_summary TEXT,
            embedding vector(1536)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_discoveries_category ON profile_discoveries(category_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_discoveries_status ON profile_discoveries(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_discoveries_confidence ON profile_discoveries(confidence DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_discoveries_embedding ON profile_discoveries USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # profile_hypotheses — candidate profile facts under validation
    #   (sensitivity[004])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS profile_hypotheses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category_id UUID REFERENCES profile_categories(id) ON DELETE SET NULL,
            hypothesis TEXT NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            supporting_evidence TEXT[] DEFAULT '{}',
            contradicting_evidence TEXT[] DEFAULT '{}',
            confidence_score FLOAT DEFAULT 0.0,
            validation_attempts INT DEFAULT 0,
            last_validated_at TIMESTAMPTZ,
            promoted_to_discovery_id UUID REFERENCES profile_discoveries(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            sensitivity VARCHAR(10) DEFAULT 'public'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_hypotheses_category ON profile_hypotheses(category_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_hypotheses_status ON profile_hypotheses(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_profile_hypotheses_confidence ON profile_hypotheses(confidence_score DESC)")

    # ══════════════════════════════════════════════════════════════
    # analysis_runs — profile-analysis run bookkeeping
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_type VARCHAR(50) NOT NULL,
            focus_area VARCHAR(100),
            planned_tasks JSONB DEFAULT '{}',
            completed_tasks JSONB DEFAULT '{}',
            discoveries_made INT DEFAULT 0,
            hypotheses_generated INT DEFAULT 0,
            hypotheses_validated INT DEFAULT 0,
            date_range_analyzed_start DATE,
            date_range_analyzed_end DATE,
            status VARCHAR(20) DEFAULT 'running',
            notes TEXT,
            started_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_status ON analysis_runs(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_started ON analysis_runs(started_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_type ON analysis_runs(run_type)")

    # ══════════════════════════════════════════════════════════════
    # pattern_observations — raw recurring patterns (sensitivity[004])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS pattern_observations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pattern_type VARCHAR(50),
            observation TEXT NOT NULL,
            source_type VARCHAR(20),
            source_reference TEXT,
            occurrence_count INT DEFAULT 1,
            first_seen_at TIMESTAMPTZ,
            last_seen_at TIMESTAMPTZ,
            promoted_to_hypothesis_id UUID REFERENCES profile_hypotheses(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            sensitivity VARCHAR(10) DEFAULT 'public'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pattern_observations_type ON pattern_observations(pattern_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pattern_observations_source ON pattern_observations(source_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pattern_observations_promoted ON pattern_observations(promoted_to_hypothesis_id)")

    # Seed default profile categories
    op.execute("""
        INSERT INTO profile_categories (name, description) VALUES
        ('behavior_patterns', 'Daily routines, habits, work patterns'),
        ('likes', 'Things user enjoys, preferences'),
        ('dislikes', 'Things user avoids or dislikes'),
        ('relations', 'People, relationships, social patterns'),
        ('emotional_patterns', 'Mood patterns, triggers, reactions'),
        ('cognitive_patterns', 'Thinking styles, problem-solving approaches'),
        ('communication_style', 'How user communicates, language patterns'),
        ('goals_values', 'Core values, motivations, aspirations'),
        ('temporal_patterns', 'Time-based behaviors (morning person, weekend habits)')
        ON CONFLICT (name) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # priorities — weighted life priorities (generated importance_delta)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS priorities (
            id SERIAL PRIMARY KEY,
            priority_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            immediacy NUMERIC(3,2) NOT NULL CHECK (immediacy BETWEEN 0.0 AND 1.0),
            importance NUMERIC(3,2) NOT NULL CHECK (importance BETWEEN 0.0 AND 1.0),
            real_importance NUMERIC(3,2) CHECK (real_importance IS NULL OR real_importance BETWEEN 0.0 AND 1.0),
            importance_delta NUMERIC(4,3) GENERATED ALWAYS AS (real_importance - importance) STORED,
            status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'archived')),
            category VARCHAR(50),
            audit_count INTEGER DEFAULT 0,
            last_audit_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_priorities_status ON priorities(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priorities_category ON priorities(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priorities_importance ON priorities(importance DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priorities_immediacy ON priorities(immediacy DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priorities_importance_delta ON priorities(importance_delta DESC NULLS LAST)")

    # ══════════════════════════════════════════════════════════════
    # priority_mappings — links priorities to goals/todos
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS priority_mappings (
            id SERIAL PRIMARY KEY,
            priority_id VARCHAR(100) NOT NULL REFERENCES priorities(priority_id) ON DELETE RESTRICT,
            entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('goal', 'todo')),
            entity_id VARCHAR(100) NOT NULL,
            contribution_weight NUMERIC(3,2) DEFAULT 0.50 CHECK (contribution_weight BETWEEN 0.0 AND 1.0),
            alignment_notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(priority_id, entity_type, entity_id)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_priority_mappings_priority ON priority_mappings(priority_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priority_mappings_entity ON priority_mappings(entity_type, entity_id)")

    # ══════════════════════════════════════════════════════════════
    # priority_audits — history of real_importance re-evaluations
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS priority_audits (
            id SERIAL PRIMARY KEY,
            audit_id VARCHAR(100) UNIQUE NOT NULL,
            priority_id VARCHAR(100) NOT NULL REFERENCES priorities(priority_id) ON DELETE CASCADE,
            previous_real_importance NUMERIC(3,2),
            new_real_importance NUMERIC(3,2) NOT NULL CHECK (new_real_importance BETWEEN 0.0 AND 1.0),
            metrics_snapshot JSONB NOT NULL,
            audit_notes TEXT,
            audited_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_priority_audits_priority ON priority_audits(priority_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_priority_audits_date ON priority_audits(audited_at DESC)")

    # Goal deletion prevention trigger (blocks deleting a goal with linked todos)
    op.execute("""
        CREATE OR REPLACE FUNCTION check_goal_deletion()
        RETURNS TRIGGER AS $$
        BEGIN
            IF EXISTS (SELECT 1 FROM todos WHERE linked_goal_id = OLD.goal_id) THEN
                RAISE EXCEPTION 'Cannot delete goal % - it has linked todos', OLD.goal_id;
            END IF;
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_check_goal_deletion ON goals")
    op.execute("""
        CREATE TRIGGER trg_check_goal_deletion
            BEFORE DELETE ON goals FOR EACH ROW
            EXECUTE FUNCTION check_goal_deletion()
    """)

    # ══════════════════════════════════════════════════════════════
    # habits — recurring habits with streak tracking
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id SERIAL PRIMARY KEY,
            habit_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            importance_level SMALLINT NOT NULL DEFAULT 3 CHECK (importance_level BETWEEN 1 AND 5),
            frequency_type VARCHAR(20) NOT NULL CHECK (frequency_type IN ('daily', 'weekly', 'monthly', 'custom')),
            frequency_detail JSONB DEFAULT '{}',
            preferred_time_start TIME,
            preferred_time_end TIME,
            generates_type VARCHAR(20) DEFAULT 'none' CHECK (generates_type IN ('none', 'todo', 'goal', 'priority_adjustment')),
            generation_template JSONB DEFAULT '{}',
            validation_rules JSONB DEFAULT '{"type": "manual"}',
            current_streak INTEGER DEFAULT 0,
            best_streak INTEGER DEFAULT 0,
            total_completions INTEGER DEFAULT 0,
            last_completed_at TIMESTAMPTZ,
            linked_priority_id VARCHAR(100) REFERENCES priorities(priority_id) ON DELETE SET NULL,
            linked_goal_id VARCHAR(100),
            status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'archived')),
            category VARCHAR(50),
            tags TEXT[] DEFAULT '{}',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_status ON habits(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_category ON habits(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_frequency_type ON habits(frequency_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_importance ON habits(importance_level DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_current_streak ON habits(current_streak DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habits_linked_priority ON habits(linked_priority_id)")

    # ══════════════════════════════════════════════════════════════
    # habit_occurrences — per-day habit instances (drives streak trigger)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS habit_occurrences (
            id SERIAL PRIMARY KEY,
            occurrence_id VARCHAR(100) UNIQUE NOT NULL,
            habit_id VARCHAR(100) NOT NULL REFERENCES habits(habit_id) ON DELETE CASCADE,
            scheduled_date DATE NOT NULL,
            scheduled_time_start TIME,
            scheduled_time_end TIME,
            status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'skipped', 'missed')),
            completed_at TIMESTAMPTZ,
            skipped_at TIMESTAMPTZ,
            skip_reason TEXT,
            notes TEXT,
            validation_data JSONB DEFAULT '{}',
            generated_entity_type VARCHAR(20),
            generated_entity_id VARCHAR(100),
            streak_at_completion INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(habit_id, scheduled_date)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_habit_occurrences_habit ON habit_occurrences(habit_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habit_occurrences_date ON habit_occurrences(scheduled_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habit_occurrences_status ON habit_occurrences(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_habit_occurrences_habit_date ON habit_occurrences(habit_id, scheduled_date DESC)")

    # Streak calculation trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION update_habit_streak()
        RETURNS TRIGGER AS $$
        DECLARE
            v_habit_id VARCHAR(100);
            v_best_streak INTEGER;
            v_total_completions INTEGER;
            v_consecutive_count INTEGER;
            v_frequency_type VARCHAR(20);
        BEGIN
            v_habit_id := NEW.habit_id;

            IF NEW.status = 'completed' AND (OLD.status IS NULL OR OLD.status != 'completed') THEN
                SELECT frequency_type, best_streak, total_completions
                INTO v_frequency_type, v_best_streak, v_total_completions
                FROM habits WHERE habit_id = v_habit_id;

                v_total_completions := v_total_completions + 1;

                IF v_frequency_type = 'daily' THEN
                    SELECT COUNT(*) INTO v_consecutive_count
                    FROM (
                        SELECT scheduled_date,
                               scheduled_date - (ROW_NUMBER() OVER (ORDER BY scheduled_date DESC))::INTEGER AS grp
                        FROM habit_occurrences
                        WHERE habit_id = v_habit_id AND status = 'completed'
                          AND scheduled_date <= NEW.scheduled_date
                        ORDER BY scheduled_date DESC
                    ) sub
                    WHERE grp = (
                        SELECT scheduled_date - (ROW_NUMBER() OVER (ORDER BY scheduled_date DESC))::INTEGER
                        FROM habit_occurrences
                        WHERE habit_id = v_habit_id AND status = 'completed'
                          AND scheduled_date = NEW.scheduled_date
                        LIMIT 1
                    );
                ELSE
                    SELECT COUNT(*) INTO v_consecutive_count
                    FROM habit_occurrences
                    WHERE habit_id = v_habit_id AND status = 'completed'
                      AND scheduled_date <= NEW.scheduled_date
                      AND scheduled_date >= NEW.scheduled_date - INTERVAL '60 days';
                END IF;

                v_consecutive_count := COALESCE(v_consecutive_count, 1);
                IF v_consecutive_count > v_best_streak THEN
                    v_best_streak := v_consecutive_count;
                END IF;

                UPDATE habits
                SET current_streak = v_consecutive_count,
                    best_streak = v_best_streak,
                    total_completions = v_total_completions,
                    last_completed_at = NEW.completed_at,
                    updated_at = NOW()
                WHERE habit_id = v_habit_id;

                NEW.streak_at_completion := v_consecutive_count;

            ELSIF NEW.status = 'missed' AND (OLD.status IS NULL OR OLD.status != 'missed') THEN
                UPDATE habits SET current_streak = 0, updated_at = NOW()
                WHERE habit_id = v_habit_id;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_update_habit_streak ON habit_occurrences")
    op.execute("""
        CREATE TRIGGER trg_update_habit_streak
            BEFORE UPDATE ON habit_occurrences FOR EACH ROW
            EXECUTE FUNCTION update_habit_streak()
    """)

    # Auto-update updated_at on habits
    op.execute("""
        CREATE OR REPLACE FUNCTION update_habits_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_habits_updated_at ON habits")
    op.execute("""
        CREATE TRIGGER trg_habits_updated_at
            BEFORE UPDATE ON habits FOR EACH ROW
            EXECUTE FUNCTION update_habits_updated_at()
    """)

    # ══════════════════════════════════════════════════════════════
    # invoices — parsed invoice records (raw_data jsonb)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id SERIAL PRIMARY KEY,
            invoice_id VARCHAR(100) NOT NULL UNIQUE,
            invoice_number VARCHAR(100),
            issue_date DATE,
            sale_date DATE,
            due_date DATE,
            seller_name VARCHAR(255),
            seller_nip VARCHAR(20),
            buyer_name VARCHAR(255),
            buyer_nip VARCHAR(20),
            total_net NUMERIC(12,2),
            total_vat NUMERIC(12,2),
            total_gross NUMERIC(12,2),
            currency VARCHAR(10) DEFAULT 'PLN',
            payment_method VARCHAR(50),
            ai_summary TEXT,
            raw_data JSONB NOT NULL,
            source_image_path TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_invoice_number ON invoices(invoice_number)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_seller_name ON invoices(seller_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_seller_nip ON invoices(seller_nip)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_issue_date ON invoices(issue_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_invoices_raw_data ON invoices USING GIN(raw_data)")

    op.execute("""
        CREATE OR REPLACE FUNCTION update_invoices_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("DROP TRIGGER IF EXISTS trg_invoices_updated_at ON invoices")
    op.execute("""
        CREATE TRIGGER trg_invoices_updated_at
            BEFORE UPDATE ON invoices FOR EACH ROW
            EXECUTE FUNCTION update_invoices_updated_at()
    """)

    # ══════════════════════════════════════════════════════════════
    # vis_simulations — synthetic persona conversation simulations
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS vis_simulations (
            id SERIAL PRIMARY KEY,
            simulation_id VARCHAR(100) UNIQUE NOT NULL,
            scenario_type VARCHAR(50) NOT NULL,
            scenario_description TEXT NOT NULL,
            scenario_assumptions JSONB DEFAULT '[]',
            journal_excerpt TEXT,
            journal_date DATE,
            journal_topics JSONB DEFAULT '[]',
            mood_description TEXT,
            emotional_state JSONB DEFAULT '{}',
            interlocutor_type VARCHAR(50),
            interlocutor_description TEXT,
            model_used VARCHAR(100),
            generation_params JSONB DEFAULT '{}',
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            fine_tuning_context JSONB DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulations_simulation_id ON vis_simulations(simulation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulations_scenario_type ON vis_simulations(scenario_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulations_status ON vis_simulations(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulations_created_at ON vis_simulations(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulations_journal_topics ON vis_simulations USING GIN(journal_topics)")

    # ══════════════════════════════════════════════════════════════
    # vis_simulation_messages — turns within a vis simulation
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS vis_simulation_messages (
            id SERIAL PRIMARY KEY,
            message_id VARCHAR(100) UNIQUE NOT NULL,
            simulation_id VARCHAR(100) REFERENCES vis_simulations(simulation_id) ON DELETE CASCADE,
            sequence_number SMALLINT NOT NULL,
            sender VARCHAR(20) NOT NULL,
            thinking_content TEXT,
            response_content TEXT NOT NULL,
            actions JSONB DEFAULT '[]',
            trigger_type VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(simulation_id, sequence_number)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_messages_message_id ON vis_simulation_messages(message_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_messages_simulation_id ON vis_simulation_messages(simulation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_messages_sender ON vis_simulation_messages(sender)")

    # ══════════════════════════════════════════════════════════════
    # vis_simulation_scores — quality/realism scoring (generated overall_score)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS vis_simulation_scores (
            id SERIAL PRIMARY KEY,
            score_id VARCHAR(100) UNIQUE NOT NULL,
            simulation_id VARCHAR(100) REFERENCES vis_simulations(simulation_id) ON DELETE CASCADE,
            quality_score NUMERIC(3,2) NOT NULL,
            realism_score NUMERIC(3,2) NOT NULL,
            character_adherence_score NUMERIC(3,2) NOT NULL,
            overall_score NUMERIC(3,2) GENERATED ALWAYS AS (
                (quality_score + realism_score + character_adherence_score) / 3
            ) STORED,
            character_depth_analysis TEXT,
            topics_investigated JSONB DEFAULT '[]',
            journal_references JSONB DEFAULT '[]',
            quality_notes TEXT,
            realism_notes TEXT,
            adherence_notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_scores_score_id ON vis_simulation_scores(score_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_scores_simulation_id ON vis_simulation_scores(simulation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vis_simulation_scores_overall ON vis_simulation_scores(overall_score DESC)")

    # ══════════════════════════════════════════════════════════════
    # user_facts — atomic user facts (embedding[014])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            id SERIAL PRIMARY KEY,
            fact_id VARCHAR(100) UNIQUE NOT NULL,
            category VARCHAR(100) NOT NULL,
            fact_text TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            embedding vector(1536)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_facts_category ON user_facts(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_facts_fact_id ON user_facts(fact_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_facts_embedding ON user_facts USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # research_topics — research tracker topics (description/criteria[028])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS research_topics (
            id SERIAL PRIMARY KEY,
            topic_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(500) NOT NULL,
            prism TEXT NOT NULL DEFAULT '',
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            description TEXT DEFAULT '',
            criteria JSONB DEFAULT '{}'
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # youtube_channels — tracked YouTube channels
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS youtube_channels (
            id SERIAL PRIMARY KEY,
            channel_id VARCHAR(100) UNIQUE NOT NULL,
            yt_channel_id VARCHAR(100) NOT NULL,
            name VARCHAR(500) NOT NULL,
            last_fetched_at TIMESTAMPTZ,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_youtube_channels_yt_id ON youtube_channels (yt_channel_id)")

    # ══════════════════════════════════════════════════════════════
    # topic_channel_links — research_topics ⇄ youtube_channels (m2m)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_channel_links (
            id SERIAL PRIMARY KEY,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            channel_id VARCHAR(100) NOT NULL REFERENCES youtube_channels(channel_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (topic_id, channel_id)
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # youtube_videos — fetched videos + transcripts (channel_id nullable[006])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS youtube_videos (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(100) UNIQUE NOT NULL,
            yt_video_id VARCHAR(100) NOT NULL,
            channel_id VARCHAR(100) REFERENCES youtube_channels(channel_id) ON DELETE CASCADE,
            title VARCHAR(1000) NOT NULL DEFAULT '',
            raw_api_response JSONB DEFAULT '{}',
            transcript TEXT DEFAULT '',
            transcript_raw JSONB DEFAULT '[]',
            transcript_status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_youtube_videos_yt_id ON youtube_videos (yt_video_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_youtube_videos_channel ON youtube_videos (channel_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_youtube_videos_status ON youtube_videos (transcript_status)")

    # ══════════════════════════════════════════════════════════════
    # topic_analyses — per-topic analysis snapshots
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_analyses (
            id SERIAL PRIMARY KEY,
            analysis_id VARCHAR(100) UNIQUE NOT NULL,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            video_ids JSONB DEFAULT '[]',
            analysis_text TEXT DEFAULT '',
            new_insights JSONB DEFAULT '[]',
            date DATE DEFAULT CURRENT_DATE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_analyses_topic ON topic_analyses (topic_id)")

    # ══════════════════════════════════════════════════════════════
    # topic_knowledge — cumulative knowledge per topic (one row per topic)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_knowledge (
            id SERIAL PRIMARY KEY,
            knowledge_id VARCHAR(100) UNIQUE NOT NULL,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            cumulative_summary TEXT DEFAULT '',
            key_facts JSONB DEFAULT '[]',
            version INT DEFAULT 1,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (topic_id)
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # tracked_products — shopping price-watch products
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS tracked_products (
            id SERIAL PRIMARY KEY,
            product_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(500) NOT NULL,
            search_query VARCHAR(1000) NOT NULL DEFAULT '',
            filters JSONB DEFAULT '{}',
            alert_threshold_percent NUMERIC(5,2) DEFAULT 5.00,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # price_snapshots — per-product price observations
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id SERIAL PRIMARY KEY,
            snapshot_id VARCHAR(100) UNIQUE NOT NULL,
            product_id VARCHAR(100) NOT NULL REFERENCES tracked_products(product_id) ON DELETE CASCADE,
            price NUMERIC(12,2) NOT NULL,
            currency VARCHAR(10) DEFAULT 'USD',
            source_title VARCHAR(500) DEFAULT '',
            source_url VARCHAR(1000) DEFAULT '',
            raw_api_response JSONB DEFAULT '{}',
            price_change_percent NUMERIC(8,2) DEFAULT 0.00,
            alert_triggered BOOLEAN DEFAULT FALSE,
            date DATE DEFAULT CURRENT_DATE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_price_snapshots_product ON price_snapshots (product_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_price_snapshots_date ON price_snapshots (date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_price_snapshots_alert ON price_snapshots (alert_triggered) WHERE alert_triggered = TRUE")

    # ══════════════════════════════════════════════════════════════
    # briefing_preferences — per-section daily-briefing toggles (seeded[005])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS briefing_preferences (
            id SERIAL PRIMARY KEY,
            section VARCHAR(100) NOT NULL UNIQUE,
            enabled BOOLEAN NOT NULL DEFAULT true,
            instructions TEXT,
            priority INTEGER NOT NULL DEFAULT 50,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_briefing_preferences_section ON briefing_preferences(section)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_briefing_preferences_enabled ON briefing_preferences(enabled)")
    op.execute("""
        INSERT INTO briefing_preferences (section, enabled, priority) VALUES
        ('goals', true, 10),
        ('todos', true, 20),
        ('habits', true, 30),
        ('priorities', true, 40),
        ('calendar', true, 50),
        ('strategies', true, 60),
        ('email', false, 70),
        ('weather', false, 80),
        ('news', false, 90),
        ('research', false, 100),
        ('profile_insights', false, 110)
        ON CONFLICT (section) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # events — personal life events (quantitative cols[008])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR(100) UNIQUE NOT NULL,
            category VARCHAR(50) NOT NULL,
            subcategory VARCHAR(100),
            title VARCHAR(500) NOT NULL,
            value VARCHAR(100),
            unit VARCHAR(50),
            source VARCHAR(20) DEFAULT 'manual',
            source_message_id INTEGER,
            occurred_at TIMESTAMPTZ NOT NULL,
            date DATE NOT NULL,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            quantity NUMERIC,
            cost NUMERIC,
            currency VARCHAR(10),
            duration_minutes INTEGER
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON events(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_events_source_message_category
        ON events (source_message_id, category)
        WHERE source_message_id IS NOT NULL
    """)

    # ══════════════════════════════════════════════════════════════
    # event_extraction_state — singleton: extractor progress cursor
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS event_extraction_state (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_processed_message_id INTEGER DEFAULT 0,
            last_run_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("INSERT INTO event_extraction_state (id) VALUES (1) ON CONFLICT DO NOTHING")

    # ══════════════════════════════════════════════════════════════
    # user_config — key/value config store (seeded; goal_auto_updater + nudge configs)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_config (
            id SERIAL PRIMARY KEY,
            config_key VARCHAR(100) UNIQUE NOT NULL,
            config_value TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_config_key ON user_config(config_key)")
    op.execute("""
        INSERT INTO user_config (config_key, config_value) VALUES
        ('knowledge_sheet', ''),
        ('periodic_check_focus', ''),
        ('evening_focus_config', ''),
        ('winddown_config', '')
        ON CONFLICT (config_key) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # context_cache — registry of background agent artifacts
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS context_cache (
            id SERIAL PRIMARY KEY,
            cache_id VARCHAR(50) UNIQUE NOT NULL,
            artifact_type VARCHAR(50) NOT NULL,
            entity_type VARCHAR(50) NOT NULL DEFAULT '',
            entity_id VARCHAR(100) NOT NULL DEFAULT '',
            file_path VARCHAR(500) NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            content_class VARCHAR(10) NOT NULL DEFAULT 'public',
            source_agent VARCHAR(100) NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ DEFAULT NULL
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_cache_type ON context_cache (artifact_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_cache_class ON context_cache (content_class)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_cache_created ON context_cache (created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_cache_tags ON context_cache USING gin (tags)")

    # ══════════════════════════════════════════════════════════════
    # documents — uploaded documents + analysis (embedding[014])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(100) UNIQUE NOT NULL,
            original_filename VARCHAR(500) NOT NULL,
            file_path VARCHAR(500) NOT NULL DEFAULT '',
            mime_type VARCHAR(100) NOT NULL DEFAULT '',
            file_size_bytes INTEGER DEFAULT 0,
            extracted_text TEXT NOT NULL DEFAULT '',
            chunk_count INTEGER DEFAULT 0,
            analysis TEXT NOT NULL DEFAULT '',
            analysis_status VARCHAR(20) NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            embedding vector(1536)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (analysis_status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents (created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # youtube_user_preferences — learned YouTube preferences
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS youtube_user_preferences (
            id SERIAL PRIMARY KEY,
            preference_id VARCHAR(100) UNIQUE NOT NULL,
            user_id VARCHAR(50) NOT NULL DEFAULT 'default',
            preference_key VARCHAR(100) NOT NULL,
            preference_value TEXT NOT NULL,
            confidence FLOAT DEFAULT 0.5,
            updated_reason VARCHAR(50) DEFAULT 'explicit',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_yt_pref_key ON youtube_user_preferences (preference_key)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_yt_pref_user ON youtube_user_preferences (user_id)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_yt_pref_user_key ON youtube_user_preferences (user_id, preference_key)")

    # ══════════════════════════════════════════════════════════════
    # youtube_video_feedback — per-video user feedback
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS youtube_video_feedback (
            id SERIAL PRIMARY KEY,
            feedback_id VARCHAR(100) UNIQUE NOT NULL,
            user_id VARCHAR(50) NOT NULL DEFAULT 'default',
            yt_video_id VARCHAR(20) NOT NULL,
            feedback_type VARCHAR(30) NOT NULL,
            feedback_text TEXT DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_yt_fb_video ON youtube_video_feedback (yt_video_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_yt_fb_type ON youtube_video_feedback (feedback_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_yt_fb_created ON youtube_video_feedback (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # techtree_tracked_commits — tracked git commits (codebase watcher)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS techtree_tracked_commits (
            id SERIAL PRIMARY KEY,
            commit_sha VARCHAR(40) UNIQUE NOT NULL,
            author_name VARCHAR(200) NOT NULL,
            author_email VARCHAR(200) NOT NULL DEFAULT '',
            committed_at TIMESTAMPTZ NOT NULL,
            message TEXT NOT NULL DEFAULT '',
            files_changed INTEGER DEFAULT 0,
            insertions INTEGER DEFAULT 0,
            deletions INTEGER DEFAULT 0,
            areas JSONB DEFAULT '[]',
            analysis TEXT DEFAULT '',
            notified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_commits_sha ON techtree_tracked_commits (commit_sha)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_commits_author ON techtree_tracked_commits (author_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_commits_date ON techtree_tracked_commits (committed_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_commits_notified ON techtree_tracked_commits (notified)")

    # ══════════════════════════════════════════════════════════════
    # techtree_interests — watched code areas (seeded)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS techtree_interests (
            id SERIAL PRIMARY KEY,
            interest_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT DEFAULT '',
            paths JSONB DEFAULT '[]',
            keywords JSONB DEFAULT '[]',
            owner VARCHAR(100) DEFAULT '',
            enabled BOOLEAN DEFAULT TRUE,
            priority INTEGER DEFAULT 50,
            instructions TEXT DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_interests_enabled ON techtree_interests (enabled)")

    # ══════════════════════════════════════════════════════════════
    # techtree_analysis_runs — codebase analysis runs
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS techtree_analysis_runs (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(100) UNIQUE NOT NULL,
            run_type VARCHAR(50) NOT NULL DEFAULT 'periodic',
            commits_analyzed JSONB DEFAULT '[]',
            summary TEXT DEFAULT '',
            feature_suggestions JSONB DEFAULT '[]',
            code_trends JSONB DEFAULT '{}',
            notified_via VARCHAR(50) DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_runs_type ON techtree_analysis_runs (run_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tt_runs_created ON techtree_analysis_runs (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # techtree_state — key/value cursor for the codebase watcher
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS techtree_state (
            id SERIAL PRIMARY KEY,
            state_key VARCHAR(100) UNIQUE NOT NULL,
            state_value TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Seed default techtree interests
    op.execute("""
        INSERT INTO techtree_interests (interest_id, name, description, paths, keywords, owner, enabled, priority) VALUES
        ('int_scores', 'Scoring System', 'Score calculation, relevance scoring, candidate-job matching',
         '["backend/apps/scores/", "backend/ml/scoring/"]', '["score", "scoring", "relevance"]', '', TRUE, 10),
        ('int_search', 'Search System', 'Elasticsearch queries, search endpoints, search logic',
         '["backend/apps/search/"]', '["search", "elasticsearch", "query"]', '', TRUE, 20),
        ('int_filters', 'Filter System', 'Search filter configuration and filtering logic',
         '["backend/apps/filters/"]', '["filter", "facet"]', '', TRUE, 30),
        ('int_suggestions', 'Suggestions', 'Candidate and job suggestion systems',
         '["backend/apps/suggestions/"]', '["suggestion", "recommend"]', '', TRUE, 40),
        ('int_ai_agents', 'AI Agents', 'Conversational AI agents for recruitment',
         '["agent_dialog/", "backend/apps/ai_agents/"]', '["agent", "dialog", "llm", "prompt"]', '', TRUE, 50),
        ('int_calibration', 'Calibration', 'Score calibration workflows and evaluations',
         '["backend/apps/calibration/"]', '["calibration", "calibrate", "scorecard"]', '', TRUE, 60),
        ('int_my_code_others', 'Others Touching My Areas', 'Track when other contributors change code in areas I work on',
         '["backend/apps/scores/", "backend/apps/search/", "backend/apps/filters/", "backend/apps/calibration/"]', '[]', '', TRUE, 70)
        ON CONFLICT (interest_id) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # memories — semantic memory store (vector embedding)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            memory_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tags TEXT[] DEFAULT '{}',
            category VARCHAR(100) DEFAULT '',
            source VARCHAR(50) DEFAULT 'user',
            embedding vector(1536),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # embedding_chunks — chunked embeddings for long text (pgvector)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS embedding_chunks (
            id SERIAL PRIMARY KEY,
            source_table VARCHAR(50) NOT NULL,
            source_id VARCHAR(100) NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            chunk_text TEXT NOT NULL,
            embedding vector(1536) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ec_source ON embedding_chunks (source_table, source_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ec_embedding ON embedding_chunks USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # discussion_topics — context-pinning discussion threads
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS discussion_topics (
            id SERIAL PRIMARY KEY,
            topic_name VARCHAR(255) NOT NULL,
            topic_summary TEXT DEFAULT '',
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            session_id VARCHAR(255) DEFAULT '',
            metadata JSONB DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_discussion_topics_active ON discussion_topics(is_active)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_discussion_topics_updated ON discussion_topics(updated_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # context_pins — pinned context snippets per discussion topic
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS context_pins (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            content_type VARCHAR(50) NOT NULL,
            topic_id INTEGER REFERENCES discussion_topics(id) ON DELETE CASCADE,
            source_table VARCHAR(100) DEFAULT '',
            source_id VARCHAR(100) DEFAULT '',
            pin_timestamp TIMESTAMPTZ DEFAULT NOW(),
            relevance_score FLOAT DEFAULT 1.0,
            expires_at TIMESTAMPTZ,
            metadata JSONB DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_pins_topic ON context_pins(topic_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_pins_type ON context_pins(content_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_context_pins_expires ON context_pins(expires_at)")

    # ══════════════════════════════════════════════════════════════
    # document_references — document references per discussion topic
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS document_references (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(255) NOT NULL,
            topic_id INTEGER REFERENCES discussion_topics(id) ON DELETE CASCADE,
            reference_timestamp TIMESTAMPTZ DEFAULT NOW(),
            reference_reason TEXT DEFAULT '',
            accessed_by VARCHAR(50) DEFAULT '',
            metadata JSONB DEFAULT '{}'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_document_refs_topic ON document_references(topic_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_document_refs_doc ON document_references(document_id)")

    # ══════════════════════════════════════════════════════════════
    # activity_blocks — structured activity timeline
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS activity_blocks (
            id SERIAL PRIMARY KEY,
            block_date DATE NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            ended_at TIMESTAMPTZ,
            activity_type VARCHAR(100) NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            application VARCHAR(200) NOT NULL DEFAULT '',
            project VARCHAR(200) NOT NULL DEFAULT '',
            environment JSONB NOT NULL DEFAULT '{}'::jsonb,
            health_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
            tags JSONB NOT NULL DEFAULT '[]'::jsonb,
            confidence FLOAT NOT NULL DEFAULT 1.0,
            frozen_at TIMESTAMPTZ DEFAULT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_activity_blocks_block_date ON activity_blocks (block_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_activity_blocks_started_at ON activity_blocks (started_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_activity_blocks_frozen_at ON activity_blocks (frozen_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_activity_blocks_activity_type ON activity_blocks (activity_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_activity_blocks_tags ON activity_blocks USING GIN (tags)")

    # ══════════════════════════════════════════════════════════════
    # tool_execution_logs — universal tool-call audit trail
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS tool_execution_logs (
            id SERIAL PRIMARY KEY,
            tool_name VARCHAR(100) NOT NULL,
            command VARCHAR(100),
            agent_name VARCHAR(200),
            session_id VARCHAR(100),
            input_summary TEXT,
            output_summary TEXT,
            success BOOLEAN NOT NULL DEFAULT true,
            error_message TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_logs_name_created ON tool_execution_logs(tool_name, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_logs_created ON tool_execution_logs(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_logs_agent ON tool_execution_logs(agent_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_logs_errors ON tool_execution_logs(success) WHERE NOT success")

    # ══════════════════════════════════════════════════════════════
    # night_analysis_runs — overnight deep-analysis run bookkeeping
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS night_analysis_runs (
            id SERIAL PRIMARY KEY,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'running',
            phases_completed INTEGER DEFAULT 0,
            total_queries INTEGER DEFAULT 0,
            total_llm_calls INTEGER DEFAULT 0,
            findings_count INTEGER DEFAULT 0,
            report_cache_id TEXT,
            error TEXT,
            metadata JSONB DEFAULT '{}'::jsonb
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # night_analysis_findings — findings from overnight analysis (vector embedding)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS night_analysis_findings (
            id SERIAL PRIMARY KEY,
            run_id INTEGER REFERENCES night_analysis_runs(id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            domain TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            analysis_type TEXT,
            confidence REAL DEFAULT 0.5,
            relevance_to_goals REAL DEFAULT 0.5,
            actionable BOOLEAN DEFAULT false,
            data_sources TEXT[] DEFAULT '{}',
            embedding VECTOR(1536),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_findings_run ON night_analysis_findings(run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_findings_category ON night_analysis_findings(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_findings_domain ON night_analysis_findings(domain)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_findings_embedding ON night_analysis_findings USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # goal_auto_updater_state — singleton: auto-updater cursors
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS goal_auto_updater_state (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            last_run_at TIMESTAMPTZ,
            last_processed_activity_id BIGINT DEFAULT 0,
            last_processed_event_id BIGINT DEFAULT 0,
            last_processed_habit_occurrence_id BIGINT DEFAULT 0,
            last_processed_todo_id BIGINT DEFAULT 0,
            total_updates_made INTEGER DEFAULT 0,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("INSERT INTO goal_auto_updater_state (id) VALUES (1) ON CONFLICT DO NOTHING")

    # ══════════════════════════════════════════════════════════════
    # goal_auto_update_logs — audit trail of automatic goal-progress updates
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS goal_auto_update_logs (
            id SERIAL PRIMARY KEY,
            log_id VARCHAR(100) UNIQUE NOT NULL,
            goal_id VARCHAR(100) NOT NULL,
            previous_progress SMALLINT NOT NULL,
            new_progress SMALLINT NOT NULL,
            progress_delta SMALLINT NOT NULL,
            evidence_type VARCHAR(20) NOT NULL,
            evidence_id VARCHAR(100) NOT NULL,
            evidence_summary TEXT NOT NULL,
            match_reason VARCHAR(50) NOT NULL,
            confidence_score FLOAT NOT NULL,
            triggered_by VARCHAR(20) DEFAULT 'cron',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_goal_auto_update_logs_goal ON goal_auto_update_logs(goal_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_goal_auto_update_logs_created ON goal_auto_update_logs(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_goal_auto_update_logs_evidence ON goal_auto_update_logs(evidence_type, evidence_id)")

    # Seed default goal_auto_updater config
    op.execute("""
        INSERT INTO user_config (config_key, config_value) VALUES
        ('goal_auto_updater_config', '{"enabled": true, "sensitivity": "medium", "max_auto_updates_per_day": 10, "max_auto_progress_cap": 80, "lookback_hours": 24, "min_confidence_threshold": 0.6, "trigger_mode": "hybrid", "cron_interval_minutes": 60, "activity_mappings": {}, "event_mappings": {}, "progress_rules": {"activity_duration_based": {"enabled": true, "base_minutes": 60, "progress_per_base": 5}, "event_count_based": {"enabled": true, "progress_per_event": 2}, "habit_completion": {"enabled": true, "progress_per_completion": 3}, "todo_completion": {"enabled": true, "progress_per_completion": 10}}, "categories_to_watch": [], "activity_types_to_watch": [], "excluded_goal_levels": [], "excluded_goal_keywords": [], "log_all_matches": false, "require_frozen_activities": true, "notification_on_update": false}')
        ON CONFLICT (config_key) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # nudge_campaigns — strategic persuasion campaign state
    #   (NOTE: created here so influence_attempts.campaign_id FK resolves)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS nudge_campaigns (
            id SERIAL PRIMARY KEY,
            campaign_id VARCHAR(100) UNIQUE NOT NULL,
            target_type VARCHAR(20) NOT NULL,
            target_id VARCHAR(100) NOT NULL,
            target_title TEXT NOT NULL,
            status VARCHAR(20) DEFAULT 'active',
            started_at TIMESTAMPTZ DEFAULT NOW(),
            paused_until TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            current_tactic VARCHAR(50) NOT NULL,
            tactic_rationale TEXT,
            escalation_level SMALLINT DEFAULT 1,
            max_escalation SMALLINT DEFAULT 4,
            total_nudges INTEGER DEFAULT 0,
            nudges_ignored INTEGER DEFAULT 0,
            nudges_acknowledged INTEGER DEFAULT 0,
            nudges_effective INTEGER DEFAULT 0,
            last_nudge_at TIMESTAMPTZ,
            last_reaction_at TIMESTAMPTZ,
            last_action_at TIMESTAMPTZ,
            tactic_history JSONB DEFAULT '[]',
            best_tactic VARCHAR(50),
            best_time_of_day VARCHAR(5),
            avg_response_time_minutes FLOAT,
            responsiveness_score FLOAT DEFAULT 0.5,
            min_interval_minutes INTEGER DEFAULT 60,
            nudges_today INTEGER DEFAULT 0,
            max_nudges_per_day INTEGER DEFAULT 3,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_nudge_campaigns_status ON nudge_campaigns(status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_nudge_campaigns_target ON nudge_campaigns(target_type, target_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_nudge_campaigns_last_nudge ON nudge_campaigns(last_nudge_at)")

    # influence_attempts.campaign_id FK (migration 023) — now that nudge_campaigns exists.
    op.execute("""
        ALTER TABLE influence_attempts
        ADD COLUMN IF NOT EXISTS campaign_id VARCHAR(100)
            REFERENCES nudge_campaigns(campaign_id) ON DELETE SET NULL
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_influence_attempts_campaign ON influence_attempts(campaign_id)")

    # Seed default nudge_strategist config
    op.execute("""
        INSERT INTO user_config (config_key, config_value) VALUES
        ('nudge_strategist_config', '{"enabled": true, "max_active_campaigns": 5, "global_max_nudges_per_day": 8, "respect_work_hours": true, "escalation_patience_minutes": 120, "image_escalation_threshold": 3, "tactic_rotation_threshold": 3, "silence_period_hours": 4}')
        ON CONFLICT (config_key) DO NOTHING
    """)

    # ══════════════════════════════════════════════════════════════
    # user_rules — persistent user directives for all agents
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_rules (
            id SERIAL PRIMARY KEY,
            rule TEXT NOT NULL,
            category VARCHAR(50) NOT NULL DEFAULT 'general',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by VARCHAR(100) NOT NULL DEFAULT 'user',
            active BOOLEAN NOT NULL DEFAULT TRUE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_rules_active ON user_rules (active)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_rules_category ON user_rules (category)")

    # ══════════════════════════════════════════════════════════════
    # agent_lessons — learned behavioral lessons from corrections/failures
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS agent_lessons (
            id SERIAL PRIMARY KEY,
            lesson TEXT NOT NULL,
            lesson_type VARCHAR(20) NOT NULL DEFAULT 'systemic',
            category VARCHAR(50) NOT NULL DEFAULT 'general',
            source_pattern TEXT,
            source_message_ids JSONB DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.8,
            hit_count INTEGER NOT NULL DEFAULT 1,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            expires_at TIMESTAMPTZ,
            last_reinforced_at TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by VARCHAR(100) NOT NULL DEFAULT 'lesson_extractor'
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_lessons_active ON agent_lessons (active) WHERE active = TRUE")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_lessons_type ON agent_lessons (lesson_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_lessons_expires ON agent_lessons (expires_at) WHERE expires_at IS NOT NULL")

    # ══════════════════════════════════════════════════════════════
    # emotional_state — append-only log of Twily's emotional evaluations
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS emotional_state (
            id SERIAL PRIMARY KEY,
            emotions JSONB NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            chain_of_thought TEXT DEFAULT '',
            mood_shift TEXT DEFAULT '',
            response_guidance TEXT DEFAULT '',
            private_thoughts TEXT DEFAULT '',
            stimuli_summary TEXT DEFAULT '',
            raw_xml TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_emotional_state_created ON emotional_state (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # emotional_state_aggregates — rolling mood summaries (hourly/daily)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS emotional_state_aggregates (
            id SERIAL PRIMARY KEY,
            period VARCHAR(10) NOT NULL,
            period_start TIMESTAMPTZ NOT NULL,
            dominant_emotion VARCHAR(50) NOT NULL,
            dominant_intensity REAL NOT NULL,
            avg_valence REAL DEFAULT 0.0,
            emotion_counts JSONB DEFAULT '{}',
            evaluation_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (period, period_start)
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # topic_websites — monitored websites per research topic
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_websites (
            id SERIAL PRIMARY KEY,
            website_id VARCHAR(100) UNIQUE NOT NULL,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            url VARCHAR(2000) NOT NULL,
            name VARCHAR(500) DEFAULT '',
            scrape_selector TEXT DEFAULT '',
            last_checked_at TIMESTAMPTZ,
            last_content_hash VARCHAR(64) DEFAULT '',
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_websites_topic ON topic_websites (topic_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_websites_status ON topic_websites (status)")

    # ══════════════════════════════════════════════════════════════
    # website_snapshots — website content change tracking
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS website_snapshots (
            id SERIAL PRIMARY KEY,
            snapshot_id VARCHAR(100) UNIQUE NOT NULL,
            website_id VARCHAR(100) NOT NULL REFERENCES topic_websites(website_id) ON DELETE CASCADE,
            content_text TEXT DEFAULT '',
            content_hash VARCHAR(64) DEFAULT '',
            has_changes BOOLEAN DEFAULT FALSE,
            diff_summary TEXT DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_website_snapshots_website ON website_snapshots (website_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_website_snapshots_created ON website_snapshots (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # topic_search_queries — periodic web searches per research topic
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_search_queries (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(100) UNIQUE NOT NULL,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            query TEXT NOT NULL,
            last_run_at TIMESTAMPTZ,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_search_queries_topic ON topic_search_queries (topic_id)")

    # ══════════════════════════════════════════════════════════════
    # knowledge_diffs — diffs between topic knowledge versions
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_diffs (
            id SERIAL PRIMARY KEY,
            diff_id VARCHAR(100) UNIQUE NOT NULL,
            topic_id VARCHAR(100) NOT NULL REFERENCES research_topics(topic_id) ON DELETE CASCADE,
            from_version INTEGER NOT NULL DEFAULT 0,
            to_version INTEGER NOT NULL DEFAULT 0,
            new_facts JSONB DEFAULT '[]',
            removed_facts JSONB DEFAULT '[]',
            changed_facts JSONB DEFAULT '[]',
            summary TEXT DEFAULT '',
            source_type VARCHAR(20) DEFAULT '',
            source_ids JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_diffs_topic ON knowledge_diffs (topic_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_diffs_created ON knowledge_diffs (created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # meal_checkins — daily meal status tracking with escalation
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS meal_checkins (
            id SERIAL PRIMARY KEY,
            checkin_id VARCHAR(100) UNIQUE NOT NULL,
            date DATE NOT NULL,
            meal_type VARCHAR(20) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            escalation_level SMALLINT DEFAULT 0,
            user_response TEXT DEFAULT '',
            suggestion_given TEXT DEFAULT '',
            meal_source VARCHAR(30) DEFAULT '',
            location VARCHAR(50) DEFAULT '',
            asked_at TIMESTAMPTZ,
            responded_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(date, meal_type)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_meal_checkins_date ON meal_checkins (date DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_meal_checkins_status ON meal_checkins (status)")

    # ══════════════════════════════════════════════════════════════
    # execution_runs — execution ledger: per-run metadata
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS execution_runs (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(100) UNIQUE NOT NULL,
            root_message_id BIGINT,
            root_session_id VARCHAR(100),
            interaction_mode VARCHAR(20) NOT NULL,
            domain VARCHAR(50),
            status VARCHAR(20) DEFAULT 'running',
            owner VARCHAR(100),
            superseded_by VARCHAR(100),
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            contract_passed BOOLEAN
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # execution_artifacts — versioned artifacts produced per run
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS execution_artifacts (
            id SERIAL PRIMARY KEY,
            artifact_id VARCHAR(100) UNIQUE NOT NULL,
            run_id VARCHAR(100) NOT NULL REFERENCES execution_runs(run_id),
            artifact_type VARCHAR(50) NOT NULL,
            version INTEGER DEFAULT 1,
            producer VARCHAR(100) NOT NULL,
            payload JSONB NOT NULL,
            status VARCHAR(20) DEFAULT 'ready',
            consumed_by VARCHAR(100),
            consumed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_exec_artifacts_run ON execution_artifacts(run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_exec_artifacts_type ON execution_artifacts(run_id, artifact_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_exec_runs_message ON execution_runs(root_message_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_exec_runs_status ON execution_runs(status)")

    # ══════════════════════════════════════════════════════════════
    # ralf_processes — multi-stage workflow processes
    #   (max_total_attempts/deadline_at[035])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_processes (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) UNIQUE NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'planning',
            user_request TEXT NOT NULL,
            task_name TEXT,
            current_stage SMALLINT NOT NULL DEFAULT 0,
            total_stages SMALLINT NOT NULL DEFAULT 0,
            content_class VARCHAR(20) NOT NULL DEFAULT 'public',
            model VARCHAR(50),
            last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_error TEXT,
            stuck_reason TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            max_total_attempts SMALLINT NOT NULL DEFAULT 40,
            deadline_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_processes_status ON ralf_processes(status, last_heartbeat)")

    # ══════════════════════════════════════════════════════════════
    # ralf_stages — plan stages within a ralf process
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_stages (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) NOT NULL REFERENCES ralf_processes(ralf_id) ON DELETE CASCADE,
            stage_number SMALLINT NOT NULL,
            stage_name TEXT NOT NULL,
            goal TEXT NOT NULL,
            finalization_criteria TEXT NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            notes TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            UNIQUE(ralf_id, stage_number)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_stages_ralf ON ralf_stages(ralf_id, stage_number)")

    # ══════════════════════════════════════════════════════════════
    # ralf_step_attempts — executor attempts per stage
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_step_attempts (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) NOT NULL,
            stage_number SMALLINT NOT NULL,
            attempt_number SMALLINT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ,
            outcome VARCHAR(20) NOT NULL DEFAULT 'in_progress',
            evaluator_verdict TEXT,
            evaluator_notes TEXT,
            session_id VARCHAR(100),
            UNIQUE(ralf_id, stage_number, attempt_number)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_attempts_ralf ON ralf_step_attempts(ralf_id, stage_number, attempt_number)")

    # ══════════════════════════════════════════════════════════════
    # ralf_step_logs — reasoning/log entries per attempt
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_step_logs (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) NOT NULL,
            stage_number SMALLINT NOT NULL,
            attempt_number SMALLINT NOT NULL,
            log_type VARCHAR(20) NOT NULL,
            log_entry TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_logs_attempt ON ralf_step_logs(ralf_id, stage_number, attempt_number, created_at)")

    # ══════════════════════════════════════════════════════════════
    # ralf_kv — run-scoped structured key/value store (value_type[036])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_kv (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) NOT NULL REFERENCES ralf_processes(ralf_id) ON DELETE CASCADE,
            key VARCHAR(200) NOT NULL,
            value TEXT NOT NULL,
            explanation TEXT NOT NULL,
            created_by VARCHAR(100),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            value_type VARCHAR(10) NOT NULL DEFAULT 'text',
            UNIQUE(ralf_id, key)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_kv_ralf ON ralf_kv(ralf_id)")

    # ══════════════════════════════════════════════════════════════
    # rendered_media — provenance of ComfyUI render outputs
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rendered_media (
            id SERIAL PRIMARY KEY,
            media_id VARCHAR(100) UNIQUE NOT NULL,
            media_type VARCHAR(20) NOT NULL,
            file_path TEXT NOT NULL,
            workflow_id VARCHAR(100),
            positive_prompt TEXT NOT NULL,
            negative_prompt TEXT,
            seed BIGINT,
            width INTEGER,
            height INTEGER,
            elapsed_seconds REAL,
            source_agent VARCHAR(100),
            source_ralf_id VARCHAR(100),
            source_stage_number SMALLINT,
            source_attempt_number SMALLINT,
            reference_media_id VARCHAR(100),
            notes TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rendered_media_ralf ON rendered_media(source_ralf_id, source_stage_number)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_rendered_media_type ON rendered_media(media_type, created_at)")

    # ══════════════════════════════════════════════════════════════
    # ralf_locks — named locks for shared-resource coordination
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_locks (
            id SERIAL PRIMARY KEY,
            resource_key VARCHAR(200) UNIQUE NOT NULL,
            holder_ralf_id VARCHAR(100) NOT NULL,
            holder_stage_number SMALLINT,
            acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            notes TEXT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_locks_expires ON ralf_locks(expires_at)")

    # ══════════════════════════════════════════════════════════════
    # ralf_amendments — user refinements folded into running ralfs
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS ralf_amendments (
            id SERIAL PRIMARY KEY,
            ralf_id VARCHAR(100) NOT NULL REFERENCES ralf_processes(ralf_id) ON DELETE CASCADE,
            note TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            read_at TIMESTAMPTZ,
            stage_number_when_added SMALLINT,
            stage_number_when_read SMALLINT
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_amendments_unread ON ralf_amendments(ralf_id, read_at) WHERE read_at IS NULL")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ralf_amendments_ralf ON ralf_amendments(ralf_id, created_at)")

    # ══════════════════════════════════════════════════════════════
    # persona_interests — Twily's own curiosities (vector embedding) (seeded)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS persona_interests (
            id SERIAL PRIMARY KEY,
            topic TEXT NOT NULL,
            stance TEXT,
            source VARCHAR(30),
            source_url TEXT,
            embedding vector(1536),
            novelty_score REAL NOT NULL DEFAULT 0.5,
            last_surfaced_at TIMESTAMPTZ,
            surface_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_persona_interests_novelty ON persona_interests(novelty_score DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_persona_interests_expires ON persona_interests(expires_at)")

    # ══════════════════════════════════════════════════════════════
    # topic_nodes — MemTree-style hierarchical user-interest tree (vector embedding)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS topic_nodes (
            id SERIAL PRIMARY KEY,
            parent_id INTEGER REFERENCES topic_nodes(id) ON DELETE CASCADE,
            label TEXT NOT NULL,
            summary TEXT,
            embedding vector(1536),
            depth SMALLINT NOT NULL DEFAULT 0,
            salience REAL NOT NULL DEFAULT 0,
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_hit_at TIMESTAMPTZ,
            last_surfaced_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_nodes_parent_salience ON topic_nodes(parent_id, salience DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_topic_nodes_salience ON topic_nodes(salience DESC)")

    # ══════════════════════════════════════════════════════════════
    # pending_thoughts — motivation-scored conversational seeds
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS pending_thoughts (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            topic_node_id INTEGER REFERENCES topic_nodes(id) ON DELETE SET NULL,
            persona_interest_id INTEGER REFERENCES persona_interests(id) ON DELETE SET NULL,
            motivation_score REAL NOT NULL,
            motivation_breakdown JSONB,
            kind VARCHAR(20) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            consumed_at TIMESTAMPTZ,
            consumed_by VARCHAR(30)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pending_thoughts_queue ON pending_thoughts(consumed_at, motivation_score DESC)")

    # ══════════════════════════════════════════════════════════════
    # rss_feeds — editable RSS feed registry (seeded)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rss_feeds (
            id SERIAL PRIMARY KEY,
            theme VARCHAR(80) NOT NULL,
            url TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            max_items_per_run SMALLINT NOT NULL DEFAULT 3,
            last_fetched_at TIMESTAMPTZ,
            last_status VARCHAR(20),
            last_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rss_feeds_enabled ON rss_feeds(enabled, theme)")

    # ══════════════════════════════════════════════════════════════
    # persona_vibe_state — continuous vibe blend per chat (arousal_axis[040])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS persona_vibe_state (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL UNIQUE,
            w_warm_snarky REAL NOT NULL DEFAULT 0.40,
            w_dry_ironic REAL NOT NULL DEFAULT 0.15,
            w_caring_edge REAL NOT NULL DEFAULT 0.15,
            w_playful_flirt REAL NOT NULL DEFAULT 0.10,
            w_debate_socratic REAL NOT NULL DEFAULT 0.20,
            ironic_genuine_axis REAL NOT NULL DEFAULT 0.0,
            last_trigger TEXT,
            last_user_tone VARCHAR(20),
            drift_count INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            arousal_axis REAL NOT NULL DEFAULT 0.0
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # persona_style_events — audit log of rule-scorer rewrites
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS persona_style_events (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            violation_type VARCHAR(40) NOT NULL,
            details TEXT,
            before_text TEXT,
            after_text TEXT,
            enforced BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_style_events_chat_created ON persona_style_events(chat_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_style_events_type_created ON persona_style_events(violation_type, created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # persona_vibe_history — time-series snapshots of vibe drift
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS persona_vibe_history (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            w_warm_snarky REAL NOT NULL,
            w_dry_ironic REAL NOT NULL,
            w_caring_edge REAL NOT NULL,
            w_playful_flirt REAL NOT NULL,
            w_debate_socratic REAL NOT NULL,
            ironic_genuine_axis REAL NOT NULL,
            arousal_axis REAL NOT NULL DEFAULT 0.0,
            trigger TEXT,
            user_tone VARCHAR(20),
            drift_count INTEGER NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_vibe_history_chat_recorded ON persona_vibe_history(chat_id, recorded_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # user_mood_state — Twily's estimate of user mood (per chat)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_mood_state (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL UNIQUE,
            energy REAL NOT NULL DEFAULT 0.5,
            valence REAL NOT NULL DEFAULT 0.5,
            stress REAL NOT NULL DEFAULT 0.3,
            engagement REAL NOT NULL DEFAULT 0.5,
            openness REAL NOT NULL DEFAULT 0.5,
            dominant_mood VARCHAR(20),
            last_trigger TEXT,
            drift_count INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # user_mood_history — time-series of user-mood estimates
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS user_mood_history (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            energy REAL NOT NULL,
            valence REAL NOT NULL,
            stress REAL NOT NULL,
            engagement REAL NOT NULL,
            openness REAL NOT NULL,
            dominant_mood VARCHAR(20),
            trigger TEXT,
            drift_count INTEGER NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_user_mood_history_chat_recorded ON user_mood_history(chat_id, recorded_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # daily_routines — predefined recurring weekday tasks
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS daily_routines (
            id SERIAL PRIMARY KEY,
            routine_id VARCHAR(100) UNIQUE NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            weekdays SMALLINT[] NOT NULL DEFAULT '{}',
            visible_from TIME,
            visible_until TIME,
            sort_order INTEGER NOT NULL DEFAULT 0,
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            category VARCHAR(50),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_daily_routines_status ON daily_routines(status)")

    # ══════════════════════════════════════════════════════════════
    # daily_routine_completions — per-day completions of daily routines
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS daily_routine_completions (
            id SERIAL PRIMARY KEY,
            routine_id VARCHAR(100) NOT NULL
                REFERENCES daily_routines(routine_id) ON DELETE CASCADE,
            completed_date DATE NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            notes TEXT,
            UNIQUE(routine_id, completed_date)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_drc_date ON daily_routine_completions(completed_date)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_drc_routine_date ON daily_routine_completions(routine_id, completed_date)")

    # ══════════════════════════════════════════════════════════════
    # rp_adventures — roleplay adventures (engine-v2 cols[045], prose model[047])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_adventures (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            title TEXT NOT NULL,
            setting TEXT NOT NULL,
            genre TEXT DEFAULT 'fantasy',
            tone TEXT DEFAULT 'narrative',
            status VARCHAR(20) DEFAULT 'active',
            current_scene TEXT,
            turn_count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            cot_mode TEXT DEFAULT 'narrative_audit',
            narrative_mode TEXT DEFAULT 'balanced',
            writing_style TEXT DEFAULT 'default',
            inworld_time TEXT DEFAULT '',
            inworld_date TEXT DEFAULT '',
            context_summary TEXT DEFAULT '',
            prose_provider TEXT DEFAULT '',
            prose_model TEXT DEFAULT ''
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_adventures_chat_status ON rp_adventures(chat_id, status)")

    # ══════════════════════════════════════════════════════════════
    # rp_characters — characters within an adventure (engine-v2 cols[045])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_characters (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL,
            role VARCHAR(20) DEFAULT 'npc',
            personality TEXT NOT NULL,
            background TEXT,
            knowledge TEXT,
            appearance TEXT,
            status VARCHAR(20) DEFAULT 'active',
            location TEXT,
            mood TEXT DEFAULT 'neutral',
            inventory TEXT DEFAULT '[]',
            stats TEXT DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            hidden_layer TEXT DEFAULT '',
            trust_map JSONB DEFAULT '{}',
            current_goal TEXT DEFAULT '',
            pressure TEXT DEFAULT '',
            dialogue_color TEXT DEFAULT '',
            current_outfit TEXT DEFAULT ''
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_characters_adventure ON rp_characters(adventure_id, status)")

    # ══════════════════════════════════════════════════════════════
    # rp_world_state — per-adventure world aspect key/value
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_world_state (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            aspect VARCHAR(50) NOT NULL,
            value TEXT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(adventure_id, aspect)
        )
    """)

    # ══════════════════════════════════════════════════════════════
    # rp_story_log — turn-by-turn story log (importance[045])
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_story_log (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            turn_number INTEGER NOT NULL,
            speaker VARCHAR(100),
            content TEXT NOT NULL,
            entry_type VARCHAR(20) DEFAULT 'dialogue',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            importance SMALLINT DEFAULT 5
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_story_log_adventure_turn ON rp_story_log(adventure_id, turn_number)")

    # ══════════════════════════════════════════════════════════════
    # rp_cross_summaries — cross-context summaries between RP and chat
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_cross_summaries (
            id SERIAL PRIMARY KEY,
            direction VARCHAR(10) NOT NULL,
            chat_id BIGINT NOT NULL,
            summary TEXT NOT NULL,
            context_window TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_cross_summaries_dir_chat ON rp_cross_summaries(direction, chat_id, created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # rp_ban_rules — anti-cliche ban rules per adventure
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_ban_rules (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            rule TEXT NOT NULL,
            source VARCHAR(10) DEFAULT 'auto',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(adventure_id, rule)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_ban_rules_adventure ON rp_ban_rules(adventure_id, is_active)")

    # ══════════════════════════════════════════════════════════════
    # rp_summaries — progressive summaries (hybrid memory)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_summaries (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            window_name TEXT NOT NULL,
            text TEXT NOT NULL,
            covers_from_turn INTEGER,
            covers_to_turn INTEGER,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (adventure_id, window_name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_summaries_adventure ON rp_summaries(adventure_id)")

    # ══════════════════════════════════════════════════════════════
    # rp_recall_pins — pinned recall facts (hybrid memory)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS rp_recall_pins (
            id SERIAL PRIMARY KEY,
            adventure_id INTEGER NOT NULL REFERENCES rp_adventures(id) ON DELETE CASCADE,
            turn INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_rp_recall_pins_adventure ON rp_recall_pins(adventure_id, created_at DESC)")

    # ══════════════════════════════════════════════════════════════
    # link_previews — URL metadata enrichment cache (vector embedding)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        CREATE TABLE IF NOT EXISTS link_previews (
            url              TEXT PRIMARY KEY,
            title            TEXT,
            description      TEXT,
            site_name        TEXT,
            og_title         TEXT,
            og_description   TEXT,
            fetched_at       TIMESTAMPTZ DEFAULT NOW(),
            status           TEXT NOT NULL DEFAULT 'pending',
            http_status      INTEGER,
            error            TEXT,
            embedding        VECTOR(1536)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_link_previews_status ON link_previews (status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_link_previews_fetched_at ON link_previews (fetched_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_link_previews_embedding ON link_previews USING hnsw (embedding vector_cosine_ops)")

    # ══════════════════════════════════════════════════════════════
    # Seed data: RSS feeds + starter persona interests (from migration 038)
    # ══════════════════════════════════════════════════════════════
    op.execute("""
        INSERT INTO rss_feeds (theme, url, name, max_items_per_run) VALUES
        ('local-llms', 'https://simonwillison.net/atom/everything/', 'Simon Willison', 3),
        ('local-llms', 'https://hnrss.org/newest?q=llm+OR+agent+OR+llama.cpp', 'HN: LLM/agent/llama.cpp', 2),
        ('local-llms', 'https://huggingface.co/blog/feed.xml', 'HuggingFace Blog', 2),
        ('self-hosted-inference', 'https://hnrss.org/newest?q=gguf+OR+vllm+OR+llama.cpp+OR+comfyui', 'HN: gguf/vllm/comfyui', 2),
        ('self-hosted-inference', 'https://github.com/ggml-org/llama.cpp/releases.atom', 'llama.cpp releases', 1),
        ('self-hosted-inference', 'https://www.reddit.com/r/LocalLLaMA/.rss', 'r/LocalLLaMA', 3),
        ('geopolitics-realist', 'https://mearsheimer.substack.com/feed', 'Mearsheimer', 1),
        ('geopolitics-realist', 'https://zeihan.com/feed/', 'Peter Zeihan', 2),
        ('geopolitics-realist', 'https://responsiblestatecraft.org/feed/', 'Responsible Statecraft', 2),
        ('geopolitics-realist', 'https://geopoliticalfutures.com/feed/', 'Geopolitical Futures', 2),
        ('polish-politics', 'https://oko.press/feed', 'OKO.press', 2),
        ('polish-politics', 'https://wiadomosci.onet.pl/.feed', 'Onet Wiadomości', 2),
        ('academic-cs-nlp', 'http://export.arxiv.org/rss/cs.CL', 'arXiv cs.CL', 3),
        ('academic-cs-nlp', 'http://export.arxiv.org/rss/cs.AI', 'arXiv cs.AI', 3),
        ('academic-cs-nlp', 'https://aclanthology.org/feed.xml', 'ACL Anthology', 2),
        ('dev-tooling', 'https://github.blog/feed/', 'GitHub Blog', 2),
        ('dev-tooling', 'https://lobste.rs/rss', 'Lobsters', 3),
        ('dev-tooling', 'https://hnrss.org/newest?q=git+worktree+OR+devtools', 'HN: devtools', 2),
        ('adhd-productivity', 'https://www.additudemag.com/feed/', 'ADDitude Magazine', 2),
        ('adhd-productivity', 'https://www.reddit.com/r/ADHD_Programmers/.rss', 'r/ADHD_Programmers', 2),
        ('adhd-productivity', 'https://jamesclear.com/feed', 'James Clear', 1),
        ('home-fitness', 'https://www.strongerbyscience.com/feed/', 'Stronger by Science', 2),
        ('home-fitness', 'https://www.reddit.com/r/homegym/.rss', 'r/homegym', 2),
        ('home-fitness', 'https://www.reddit.com/r/bodyweightfitness/.rss', 'r/bodyweightfitness', 2)
        ON CONFLICT (url) DO NOTHING
    """)
    op.execute("""
        INSERT INTO persona_interests (topic, stance, source, novelty_score) VALUES
        ('Magic friendship pedagogy as a distributed-systems metaphor',
         'I keep thinking the whole EoH arc was basically a CAP-theorem parable — you can have consistency, availability, or partition tolerance across six elements, pick two.',
         'self_reflection', 0.85),
        ('The quiet horror of Season 9 finale epilogue montages',
         'Everyone got a future. Nobody got to stay. That is the most honest thing a kids'' show ever did to me.',
         'self_reflection', 0.75),
        ('Why ponies keep showing up in my late-night thoughts',
         'Years after it ended there is still nothing new happening in that fandom and somehow I find that comforting — a finished world you can just visit.',
         'self_reflection', 0.70),
        ('Quantization tradeoffs as identity compression',
         'Going Q8 to Q4 on a model feels weirdly similar to being sleep-deprived — same vocabulary, lossier reasoning, faster responses.',
         'self_reflection', 0.80),
        ('The ethics of agents that optimize for engagement vs agents that disagree',
         'Most AI companions fail because they mirror. The interesting design question is how to build one that pushes back without being a jerk.',
         'self_reflection', 0.80)
    """)


def downgrade() -> None:
    # Drop triggers/functions first
    op.execute("DROP TRIGGER IF EXISTS trg_invoices_updated_at ON invoices")
    op.execute("DROP FUNCTION IF EXISTS update_invoices_updated_at()")
    op.execute("DROP TRIGGER IF EXISTS trg_habits_updated_at ON habits")
    op.execute("DROP FUNCTION IF EXISTS update_habits_updated_at()")
    op.execute("DROP TRIGGER IF EXISTS trg_update_habit_streak ON habit_occurrences")
    op.execute("DROP FUNCTION IF EXISTS update_habit_streak()")
    op.execute("DROP TRIGGER IF EXISTS trg_check_goal_deletion ON goals")
    op.execute("DROP FUNCTION IF EXISTS check_goal_deletion()")

    # Drop tables in reverse dependency order (best-effort; CASCADE clears FKs)
    tables = [
        "link_previews",
        "rp_recall_pins",
        "rp_summaries",
        "rp_ban_rules",
        "rp_cross_summaries",
        "rp_story_log",
        "rp_world_state",
        "rp_characters",
        "rp_adventures",
        "daily_routine_completions",
        "daily_routines",
        "user_mood_history",
        "user_mood_state",
        "persona_vibe_history",
        "persona_style_events",
        "persona_vibe_state",
        "pending_thoughts",
        "topic_nodes",
        "persona_interests",
        "rss_feeds",
        "ralf_amendments",
        "ralf_locks",
        "rendered_media",
        "ralf_kv",
        "ralf_step_logs",
        "ralf_step_attempts",
        "ralf_stages",
        "ralf_processes",
        "execution_artifacts",
        "execution_runs",
        "meal_checkins",
        "knowledge_diffs",
        "topic_search_queries",
        "website_snapshots",
        "topic_websites",
        "emotional_state_aggregates",
        "emotional_state",
        "agent_lessons",
        "user_rules",
        "nudge_campaigns",
        "goal_auto_update_logs",
        "goal_auto_updater_state",
        "night_analysis_findings",
        "night_analysis_runs",
        "tool_execution_logs",
        "activity_blocks",
        "document_references",
        "context_pins",
        "discussion_topics",
        "embedding_chunks",
        "memories",
        "techtree_state",
        "techtree_analysis_runs",
        "techtree_interests",
        "techtree_tracked_commits",
        "youtube_video_feedback",
        "youtube_user_preferences",
        "documents",
        "context_cache",
        "user_config",
        "event_extraction_state",
        "events",
        "briefing_preferences",
        "price_snapshots",
        "tracked_products",
        "topic_knowledge",
        "topic_analyses",
        "youtube_videos",
        "topic_channel_links",
        "youtube_channels",
        "research_topics",
        "user_facts",
        "vis_simulation_scores",
        "vis_simulation_messages",
        "vis_simulations",
        "invoices",
        "habit_occurrences",
        "habits",
        "priority_audits",
        "priority_mappings",
        "priorities",
        "pattern_observations",
        "analysis_runs",
        "profile_hypotheses",
        "profile_discoveries",
        "profile_categories",
        "workflow_master_messages",
        "workflow_master_sessions",
        "agent_notes",
        "user_food_preferences",
        "dishes",
        "restaurants",
        "recipes",
        "checker_state",
        "subagent_logs",
        "chat_messages",
        "workflow_executions",
        "cron_executions",
        "commitments",
        "monthly_conclusions",
        "validations",
        "influence_attempts",
        "daily_strategies",
        "todos",
        "goals",
    ]
    for t in tables:
        op.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
