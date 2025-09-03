"""
Configuration settings for the Cross-Publication Insight Assistant.
This module handles all configuration loading from environment variables
and provides defaults for the application.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings, validator
from pydantic_settings import BaseSettings


# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    
    # CrewAI Configuration
    crew_verbose: bool = os.getenv("CREW_VERBOSE", "true").lower() == "true"
    crew_memory: bool = os.getenv("CREW_MEMORY", "true").lower() == "true"
    max_rpm: int = int(os.getenv("MAX_RPM", "10"))
    max_execution_time: int = int(os.getenv("MAX_EXECUTION_TIME", "300"))  # 5 minutes
    
    # Web Scraping Configuration
    scraping_timeout: int = int(os.getenv("SCRAPING_TIMEOUT", "30"))
    scraping_delay: float = float(os.getenv("SCRAPING_DELAY", "1.0"))
    user_agent: str = os.getenv(
        "USER_AGENT", 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    enable_selenium: bool = os.getenv("ENABLE_SELENIUM", "false").lower() == "true"
    
    # Rate Limiting
    rate_limit_calls: int = int(os.getenv("RATE_LIMIT_CALLS", "10"))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds
    
    # Security Settings
    allowed_domains: List[str] = os.getenv(
        "ALLOWED_DOMAINS", 
        "arxiv.org,blog.langchain.dev,readytensor.ai,medium.com,towards*science.com"
    ).split(",")
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", "1000000"))  # 1MB
    
    # Application Settings
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    
    # UI Configuration
    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    streamlit_host: str = os.getenv("STREAMLIT_HOST", "localhost")
    
    # Database Configuration (optional)
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///insights.db")
    enable_persistence: bool = os.getenv("ENABLE_PERSISTENCE", "false").lower() == "true"
    
    # Caching
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    
    @validator('openai_api_key')
    def validate_openai_key(cls, v):
        """Validate OpenAI API key format."""
        if v and not v.startswith(('sk-', 'mock-')):
            raise ValueError("OpenAI API key must start with 'sk-' or 'mock-' for testing")
        return v
    
    @validator('allowed_domains')
    def validate_domains(cls, v):
        """Clean and validate allowed domains."""
        if isinstance(v, str):
            v = [domain.strip() for domain in v.split(",")]
        return [domain for domain in v if domain]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


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
    if settings.rate_limit_calls <= 0:
        issues.append("RATE_LIMIT_CALLS must be positive")
    
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
