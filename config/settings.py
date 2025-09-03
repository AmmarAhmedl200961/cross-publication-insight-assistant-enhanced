"""
Configuration settings for the Cross-Publication Insight Assistant.
This module handles all configuration loading from environment variables
and provides defaults for the application.
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from pydantic import validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable & backward compatibility.

    We accept both the original long-form env variable names (e.g. CREWAI_VERBOSE)
    and simplified internal ones. Extra env vars are ignored to avoid validation errors.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # --- OpenAI / LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, alias="OPENAI_MAX_TOKENS")

    # --- Crew / Agent Orchestration ---
    crew_verbose: bool = Field(default=True, alias="CREWAI_VERBOSE")
    crew_memory: bool = Field(default=True, alias="CREWAI_MEMORY")
    crew_max_iterations: int = Field(default=10, alias="CREWAI_MAX_ITERATIONS")
    max_rpm: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst_size: int = Field(default=10, alias="RATE_LIMIT_BURST_SIZE")
    max_execution_time: int = Field(default=300, alias="MAX_EXECUTION_TIME")
    # Legacy compatibility (older code references expect these)
    rate_limit_calls: int = Field(default=60, alias="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, alias="RATE_LIMIT_PERIOD")

    # --- Scraping ---
    scraping_timeout: int = Field(default=30, alias="WEB_SCRAPING_TIMEOUT")
    scraping_delay: float = Field(default=1.0, alias="WEB_SCRAPING_RATE_LIMIT")
    max_retries: int = Field(default=3, alias="WEB_SCRAPING_MAX_RETRIES")
    user_agent: str = Field(default="CrossPublicationInsightAssistant/1.0", alias="WEB_SCRAPING_USER_AGENT")
    enable_selenium: bool = Field(default=False, alias="ENABLE_SELENIUM")

    # --- Security ---
    # Store raw domains string then expose processed list via property to prevent json parsing
    allowed_domains_raw: str = Field(
        default="arxiv.org,scholar.google.com,researchgate.net,ieee.org,acm.org,springer.com,nature.com,science.org",
        alias="SECURITY_ALLOWED_DOMAINS"
    )
    max_content_length: int = Field(default=10_485_760, alias="SECURITY_MAX_CONTENT_SIZE")  # 10MB
    validate_ssl: bool = Field(default=True, alias="SECURITY_VALIDATE_SSL")

    # --- UI ---
    ui_host: str = Field(default="localhost", alias="UI_HOST")
    ui_port: int = Field(default=8501, alias="UI_PORT")
    ui_debug: bool = Field(default=False, alias="UI_DEBUG")
    ui_max_file_size: str = Field(default="50MB", alias="UI_MAX_FILE_SIZE")

    # --- Caching ---
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    cache_max_size: int = Field(default=1000, alias="CACHE_MAX_SIZE")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", alias="LOG_FORMAT")
    log_file: str = Field(default="logs/app.log", alias="LOG_FILE")
    log_max_size: str = Field(default="10MB", alias="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")

    # --- Analysis Parameters ---
    analysis_min_keywords: int = Field(default=5, alias="ANALYSIS_MIN_KEYWORDS")
    analysis_max_keywords: int = Field(default=50, alias="ANALYSIS_MAX_KEYWORDS")
    analysis_confidence_threshold: float = Field(default=0.7, alias="ANALYSIS_CONFIDENCE_THRESHOLD")
    analysis_batch_size: int = Field(default=10, alias="ANALYSIS_BATCH_SIZE")

    # --- Visualization ---
    viz_default_theme: str = Field(default="plotly_white", alias="VIZ_DEFAULT_THEME")
    viz_color_palette: str = Field(default="viridis", alias="VIZ_COLOR_PALETTE")
    viz_figure_width: int = Field(default=800, alias="VIZ_FIGURE_WIDTH")
    viz_figure_height: int = Field(default=600, alias="VIZ_FIGURE_HEIGHT")

    # --- Testing Flags ---
    test_mode: bool = Field(default=False, alias="TEST_MODE")
    test_mock_llm: bool = Field(default=False, alias="TEST_MOCK_LLM")
    test_skip_slow: bool = Field(default=False, alias="TEST_SKIP_SLOW")

    # --- Legacy / Compatibility Fields (fallbacks) ---
    debug: bool = Field(default=False, alias="DEBUG")
    database_url: str = Field(default="sqlite:///insights.db", alias="DATABASE_URL")
    enable_persistence: bool = Field(default=False, alias="ENABLE_PERSISTENCE")
    enable_caching: bool = Field(default=True, alias="ENABLE_CACHING")

    @validator('openai_api_key')
    def validate_openai_key(cls, v: str):
        """Allow empty key for local/test; basic prefix check when provided."""
        if v and not v.startswith(('sk-', 'mock-')):
            # Don't fail hard—log warning instead of raising to improve UX.
            logging.getLogger(__name__).warning(
                "OPENAI_API_KEY does not use expected prefix; continuing (set DEBUG=true to silence)"
            )
        return v

    @property
    def allowed_domains(self) -> List[str]:
        return [d.strip() for d in self.allowed_domains_raw.split(',') if d.strip()]


# Global settings instance
settings = Settings()


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration for CrewAI agents."""
    return {
        "model": settings.openai_model,
        "temperature": settings.openai_temperature,
        "max_tokens": settings.openai_max_tokens,
        "api_key": settings.openai_api_key or "mock-key-for-testing"
    }


def get_crew_config() -> Dict[str, Any]:
    """Get CrewAI crew configuration."""
    return {
        "verbose": settings.crew_verbose,
        "memory": settings.crew_memory,
        "max_rpm": settings.max_rpm,
        "max_execution_time": settings.max_execution_time
    }


def get_scraping_config() -> Dict[str, Any]:
    """Get web scraping configuration."""
    return {
        "timeout": settings.scraping_timeout,
        "delay": settings.scraping_delay,
        "user_agent": settings.user_agent,
        "max_retries": settings.max_retries,
        "enable_selenium": settings.enable_selenium,
        "allowed_domains": settings.allowed_domains,
        "max_content_length": settings.max_content_length
    }


def setup_logging() -> None:
    """Setup application logging configuration."""
    # Create logs directory if it doesn't exist
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(settings.log_file),
            logging.StreamHandler()
        ]
    )


def validate_configuration() -> bool:
    """Validate the current configuration."""
    issues = []
    
    # Check OpenAI API key for non-mock usage
    if not settings.openai_api_key and not settings.debug:
        issues.append("OPENAI_API_KEY is required for production use")
    
    # Check rate limiting settings
    effective_calls = getattr(settings, 'rate_limit_calls', None) or settings.max_rpm
    if effective_calls <= 0:
        issues.append("Rate limit must be positive (RATE_LIMIT_CALLS or RATE_LIMIT_REQUESTS_PER_MINUTE)")
    
    # Check timeout settings
    if settings.scraping_timeout <= 0:
        issues.append("SCRAPING_TIMEOUT must be positive")
    
    if issues:
        for issue in issues:
            logging.error(f"Configuration issue: {issue}")
        return False
    
    return True


# Initialize logging when module is imported
setup_logging()
