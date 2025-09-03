"""
Enhanced web scraping tool with security, rate limiting, and error handling.
"""

import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from functools import wraps

import requests
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import validators
from ratelimit import limits, sleep_and_retry
from cachetools import TTLCache
from pydantic import PrivateAttr

from crewai.tools import BaseTool
from config.settings import settings, get_scraping_config


logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Result of web scraping operation."""
    url: str
    content: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    processing_time: Optional[float] = None


class SecurityValidator:
    """Security validation for web scraping."""
    
    def __init__(self, allowed_domains: List[str], max_content_length: int):
        self.allowed_domains = allowed_domains
        self.max_content_length = max_content_length
    
    def validate_url(self, url: str) -> bool:
        """Validate URL for security concerns."""
        if not validators.url(url):
            return False
        
        parsed = urlparse(url)
        
        # Check if domain is allowed
        domain = parsed.netloc.lower()
        
        # Support wildcard domains (e.g., *.example.com)
        for allowed in self.allowed_domains:
            if allowed.startswith('*'):
                if domain.endswith(allowed[1:]):
                    return True
            elif domain == allowed or domain.endswith('.' + allowed):
                return True
        
        return len(self.allowed_domains) == 0  # Allow all if no restrictions
    
    def sanitize_content(self, content: str) -> str:
        """Sanitize scraped content."""
        if len(content) > self.max_content_length:
            logger.warning(f"Content truncated from {len(content)} to {self.max_content_length} characters")
            content = content[:self.max_content_length] + "..."
        
        # Remove potentially harmful content
        content = content.replace('\x00', '')  # Remove null bytes
        return content.strip()


class WebScrapingTool(BaseTool):
    """Enhanced web scraping tool with CrewAI integration.

    Note: crewai.tools.BaseTool inherits from a Pydantic model that forbids
    setting undeclared attributes. Internal runtime state is stored using
    PrivateAttr to avoid validation errors while keeping the public schema clean.
    """

    name: str = "web_scraper"
    description: str = "Scrapes content from web pages with security and rate limiting"

    # Private internal attributes (not part of the public model schema)
    _config: dict = PrivateAttr()
    _validator: SecurityValidator = PrivateAttr()
    _cache: TTLCache = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._config = get_scraping_config()
        self._validator = SecurityValidator(
            self._config['allowed_domains'],
            self._config['max_content_length']
        )
        self._cache = TTLCache(maxsize=100, ttl=settings.cache_ttl)
        self._session = self._create_session()

    # Backward compatibility for tests expecting .validator
    @property
    def validator(self) -> SecurityValidator:  # type: ignore
        return self._validator

    def _create_session(self) -> requests.Session:
        """Create a configured requests session."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self._config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session
    
    # Rate limit using available config (fallback: max_rpm over 60s)
    @sleep_and_retry
    @limits(calls=getattr(settings, 'max_rpm', 60), period=60)
    def _rate_limited_request(self, url: str) -> requests.Response:
        """Make a rate-limited HTTP request."""
        return self._session.get(
            url,
            timeout=self._config['timeout'],
            allow_redirects=True
        )
    
    def _scrape_with_requests(self, url: str, selector: Optional[str] = None) -> ScrapingResult:
        """Scrape using requests and BeautifulSoup."""
        start_time = time.time()
        
        try:
            response = self._rate_limited_request(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove scripts, styles, and comments
            for script in soup(["script", "style"]):
                script.decompose()
            
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else None
            
            # Extract content
            if selector:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator=" ", strip=True)
                else:
                    return ScrapingResult(
                        url=url,
                        content="",
                        success=False,
                        error_message=f"Element with selector '{selector}' not found"
                    )
            else:
                # Try common content selectors
                content_selectors = [
                    'main', 'article', '.content', '#content',
                    '.post-content', '.entry-content', '.article-content'
                ]
                
                content = ""
                for sel in content_selectors:
                    element = soup.select_one(sel)
                    if element:
                        content = element.get_text(separator=" ", strip=True)
                        break
                
                if not content:
                    content = soup.get_text(separator=" ", strip=True)
            
            # Sanitize content
            content = self._validator.sanitize_content(content)
            
            processing_time = time.time() - start_time
            
            return ScrapingResult(
                url=url,
                content=content,
                title=title,
                success=True,
                status_code=response.status_code,
                processing_time=processing_time,
                metadata={
                    'content_length': len(content),
                    'response_headers': dict(response.headers),
                    'final_url': response.url
                }
            )
            
        except requests.exceptions.RequestException as e:
            processing_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"Request error: {str(e)}",
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"Parsing error: {str(e)}",
                processing_time=processing_time
            )
    
    def _scrape_with_selenium(self, url: str, selector: Optional[str] = None) -> ScrapingResult:
        """Scrape using Selenium for JavaScript-heavy sites."""
        start_time = time.time()
        driver = None
        
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--user-agent={self._config["user_agent"]}')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(self._config['timeout'])
            
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            # Extract title
            title = driver.title
            
            # Extract content
            if selector:
                try:
                    element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    content = element.text
                except TimeoutException:
                    return ScrapingResult(
                        url=url,
                        content="",
                        success=False,
                        error_message=f"Element with selector '{selector}' not found"
                    )
            else:
                content = driver.find_element(By.TAG_NAME, "body").text
            
            # Sanitize content
            content = self._validator.sanitize_content(content)
            
            processing_time = time.time() - start_time
            
            return ScrapingResult(
                url=url,
                content=content,
                title=title,
                success=True,
                processing_time=processing_time,
                metadata={
                    'content_length': len(content),
                    'method': 'selenium'
                }
            )
            
        except (TimeoutException, WebDriverException) as e:
            processing_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"Selenium error: {str(e)}",
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                processing_time=processing_time
            )
        finally:
            if driver:
                driver.quit()
    
    def _run(self, url: str, selector: Optional[str] = None, use_selenium: bool = False) -> str:
        """Execute the web scraping operation with retry, caching, and validation."""
        # Validate URL
        if not self._validator.validate_url(url):
            error_msg = f"URL validation failed: {url}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        cache_key = f"{url}:{selector or 'default'}"
        if settings.enable_caching and cache_key in self._cache:
            logger.info(f"Cache hit for {url}")
            return self._cache[cache_key].content

        logger.info(f"Scraping URL: {url}")
        time.sleep(self._config['delay'])  # polite delay

        for attempt in range(self._config['max_retries']):
            try:
                # Choose method
                if use_selenium or self._config['enable_selenium']:
                    result = self._scrape_with_selenium(url, selector)
                else:
                    result = self._scrape_with_requests(url, selector)

                if result.success:
                    logger.info(f"Successfully scraped {url} in {result.processing_time:.2f}s")
                    if settings.enable_caching:
                        self._cache[cache_key] = result
                    return result.content

                logger.warning(f"Scraping failed for {url}: {result.error_message}")

                # Attempt Selenium fallback on last retry if not already using it
                if attempt == self._config['max_retries'] - 1 and not use_selenium:
                    logger.info(f"Trying Selenium as fallback for {url}")
                    selenium_result = self._scrape_with_selenium(url, selector)
                    if selenium_result.success:
                        if settings.enable_caching:
                            self._cache[cache_key] = selenium_result
                        return selenium_result.content
                return f"Error: {result.error_message}"
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < self._config['max_retries'] - 1:
                    time.sleep(2 ** attempt)  # exponential backoff

        error_msg = f"All scraping attempts failed for {url}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


# Legacy function for backward compatibility
def scrape_url(url: str, selector: Optional[str] = None) -> str:
    """Legacy function for backward compatibility."""
    tool = WebScrapingTool()
    return tool._run(url, selector)
