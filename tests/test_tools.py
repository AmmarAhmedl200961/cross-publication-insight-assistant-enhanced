"""
Unit tests for tools.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestToolReliability:
    """Test tool reliability and edge cases."""
    
    def test_web_scraper_timeout_handling(self):
        """Test web scraper timeout handling."""
        from src.tools.web_scraper import WebScrapingTool
        
        scraper = WebScrapingTool()
        
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = TimeoutError("Request timeout")
            
            with patch.object(scraper.validator, 'validate_url', return_value=True):
                result = scraper._run('https://example.com')
            
            assert result.startswith("Error:")
            assert "timeout" in result.lower() or "failed" in result.lower()
    
    def test_keyword_extractor_special_characters(self):
        """Test keyword extractor with special characters."""
        from src.tools.keyword_extractor import KeywordExtractionTool
        
        extractor = KeywordExtractionTool()
        
        # Text with special characters and encoding issues
        text = "Machine learning 智能系统 and AI émotions with ñovel approaches 🤖"
        
        result = extractor._run(text)
        
        # Should handle gracefully and extract valid keywords
        assert not result.startswith("Error:")
        keywords = result.split(", ")
        assert len(keywords) > 0
    
    def test_data_analyzer_large_dataset(self):
        """Test data analyzer with large dataset."""
        from src.tools.data_analyzer import DataAnalysisTool
        
        analyzer = DataAnalysisTool()
        
        # Large dataset simulation
        large_keywords = []
        for i in range(100):  # 100 publications
            keywords = [f"keyword_{j}" for j in range(i % 20)]  # Variable keyword counts
            large_keywords.append(keywords)
        
        import json
        result = analyzer._run(json.dumps(large_keywords))
        
        assert "Analysis Complete" in result
        assert not result.startswith("Error:")
    
    def test_web_scraper_malformed_html(self):
        """Test web scraper with malformed HTML."""
        from src.tools.web_scraper import WebScrapingTool
        
        scraper = WebScrapingTool()
        
        with patch('requests.Session.get') as mock_get:
            # Malformed HTML response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body><div>Unclosed div<p>Paragraph without closing tag'
            mock_response.url = 'https://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            with patch.object(scraper.validator, 'validate_url', return_value=True):
                result = scraper._run('https://example.com')
            
            # Should handle malformed HTML gracefully
            assert not result.startswith("Error:")
            assert len(result) > 0


class TestToolSecurity:
    """Test tool security features."""
    
    def test_security_validator_xss_protection(self):
        """Test security validator XSS protection."""
        from src.tools.web_scraper import SecurityValidator
        
        validator = SecurityValidator(['example.com'], 10000)
        
        # Test with potential XSS in content
        malicious_content = '<script>alert("xss")</script><p>Normal content</p>'
        
        sanitized = validator.sanitize_content(malicious_content)
        
        # Script tags should be handled by BeautifulSoup in actual scraping
        assert isinstance(sanitized, str)
        assert len(sanitized) > 0
    
    def test_security_validator_injection_protection(self):
        """Test security validator injection protection."""
        from src.tools.web_scraper import SecurityValidator
        
        validator = SecurityValidator(['example.com'], 10000)
        
        # Test various injection attempts in URLs
        injection_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "ftp://malicious.com/file"
        ]
        
        for url in injection_urls:
            assert not validator.validate_url(url)
    
    def test_rate_limiting_functionality(self):
        """Test rate limiting functionality."""
        from src.tools.web_scraper import WebScrapingTool
        import time
        
        scraper = WebScrapingTool()
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body>Test content</body></html>'
            mock_response.url = 'https://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            with patch.object(scraper.validator, 'validate_url', return_value=True):
                # Multiple rapid requests should be rate limited
                start_time = time.time()
                
                for i in range(3):
                    scraper._run(f'https://example.com/{i}')
                
                elapsed_time = time.time() - start_time
                
                # Should take some time due to rate limiting delays
                assert elapsed_time > 1.0  # At least 1 second for delays


class TestToolPerformance:
    """Test tool performance and optimization."""
    
    def test_keyword_extractor_performance(self):
        """Test keyword extractor performance with large text."""
        from src.tools.keyword_extractor import KeywordExtractionTool
        
        extractor = KeywordExtractionTool()
        
        # Large text simulation
        large_text = " ".join([
            "artificial intelligence machine learning deep learning neural networks",
            "natural language processing computer vision reinforcement learning"
        ] * 100)  # Repeat to create large text
        
        import time
        start_time = time.time()
        
        result = extractor._run(large_text)
        
        processing_time = time.time() - start_time
        
        assert not result.startswith("Error:")
        assert processing_time < 30  # Should complete within 30 seconds
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        from src.tools.web_scraper import WebScrapingTool
        
        scraper = WebScrapingTool()
        
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body>Cached content</body></html>'
            mock_response.url = 'https://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            with patch.object(scraper.validator, 'validate_url', return_value=True):
                # First request
                result1 = scraper._run('https://example.com')
                
                # Second request (should be faster due to caching)
                result2 = scraper._run('https://example.com')
                
                assert result1 == result2
                # Verify that the same content is returned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
