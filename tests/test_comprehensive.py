"""
Comprehensive test suite for the Cross-Publication Insight Assistant.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.web_scraper import WebScrapingTool, SecurityValidator
from src.tools.keyword_extractor import KeywordExtractionTool
from src.tools.data_analyzer import DataAnalysisTool
from src.agents.publication_analyzer import PublicationAnalyzerAgent
from src.agents.trend_aggregator import TrendAggregatorAgent
from src.agents.insight_generator import InsightGeneratorAgent
from src.crews.publication_crew import PublicationInsightCrew
from src.flows.publication_flow import PublicationAnalysisFlow, create_publication_flow
from src.main import CrossPublicationInsightAssistant
from config.settings import settings


class TestWebScrapingTool:
    """Test suite for web scraping tool."""
    
    @pytest.fixture
    def web_scraper(self):
        return WebScrapingTool()
    
    @pytest.fixture
    def security_validator(self):
        return SecurityValidator(['example.com', '*.test.com'], 10000)
    
    def test_security_validator_valid_domains(self, security_validator):
        """Test security validator with valid domains."""
        assert security_validator.validate_url('https://example.com/page')
        assert security_validator.validate_url('https://sub.test.com/page')
        assert security_validator.validate_url('https://www.test.com/page')
    
    def test_security_validator_invalid_domains(self, security_validator):
        """Test security validator with invalid domains."""
        assert not security_validator.validate_url('https://malicious.com/page')
        assert not security_validator.validate_url('https://evil.org/page')
    
    def test_security_validator_invalid_urls(self, security_validator):
        """Test security validator with invalid URLs."""
        assert not security_validator.validate_url('not-a-url')
        assert not security_validator.validate_url('ftp://example.com')
    
    def test_content_sanitization(self, security_validator):
        """Test content sanitization."""
        long_content = "a" * 15000
        sanitized = security_validator.sanitize_content(long_content)
        
        assert len(sanitized) <= security_validator.max_content_length + 3  # +3 for "..."
        assert sanitized.endswith("...")
    
    @patch('requests.Session.get')
    def test_scraping_success(self, mock_get, web_scraper):
        """Test successful web scraping."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><body><h1>Test Title</h1><p>Test content</p></body></html>'
        mock_response.url = 'https://example.com'
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_get.return_value = mock_response
        
        # Mock validator to allow all URLs
        with patch.object(web_scraper.validator, 'validate_url', return_value=True):
            result = web_scraper._run('https://example.com')
        
        assert "Test Title" in result
        assert "Test content" in result
    
    @patch('requests.Session.get')
    def test_scraping_with_selector(self, mock_get, web_scraper):
        """Test web scraping with CSS selector."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><body><div class="content">Target content</div><div>Other content</div></body></html>'
        mock_response.url = 'https://example.com'
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        with patch.object(web_scraper.validator, 'validate_url', return_value=True):
            result = web_scraper._run('https://example.com', selector='.content')
        
        assert "Target content" in result
        assert "Other content" not in result
    
    @patch('requests.Session.get')
    def test_scraping_failure(self, mock_get, web_scraper):
        """Test web scraping failure handling."""
        mock_get.side_effect = Exception("Network error")
        
        with patch.object(web_scraper.validator, 'validate_url', return_value=True):
            result = web_scraper._run('https://example.com')
        
        assert result.startswith("Error:")
        assert "Network error" in result
    
    def test_url_validation_failure(self, web_scraper):
        """Test URL validation failure."""
        result = web_scraper._run('https://blocked-domain.com')
        assert result.startswith("Error:")
        assert "validation failed" in result


class TestKeywordExtractionTool:
    """Test suite for keyword extraction tool."""
    
    @pytest.fixture
    def keyword_extractor(self):
        return KeywordExtractionTool()
    
    def test_keyword_extraction_basic(self, keyword_extractor):
        """Test basic keyword extraction."""
        text = "Machine learning and artificial intelligence are transforming natural language processing."
        
        result = keyword_extractor._run(text)
        
        assert isinstance(result, str)
        keywords = result.split(", ")
        assert len(keywords) > 0
        assert any("machine" in kw.lower() or "learning" in kw.lower() for kw in keywords)
    
    def test_keyword_extraction_empty_text(self, keyword_extractor):
        """Test keyword extraction with empty text."""
        result = keyword_extractor._run("")
        assert result.startswith("Error:")
    
    def test_keyword_extraction_short_text(self, keyword_extractor):
        """Test keyword extraction with very short text."""
        result = keyword_extractor._run("AI")
        assert result.startswith("Error:")
    
    def test_domain_keyword_extraction(self, keyword_extractor):
        """Test domain-specific keyword extraction."""
        text = "The neural network uses transformer architecture for deep learning tasks."
        
        domain_keywords = keyword_extractor._extract_domain_keywords(text)
        
        assert "neural" in domain_keywords
        assert "transformer" in domain_keywords
        assert "deep" in domain_keywords
    
    @patch('spacy.load')
    def test_entity_extraction_failure(self, mock_spacy_load, keyword_extractor):
        """Test entity extraction when spaCy model is not available."""
        mock_spacy_load.side_effect = OSError("Model not found")
        
        # Reinitialize to trigger the exception
        keyword_extractor._setup_spacy()
        
        entities = keyword_extractor._extract_spacy_entities("Test text with OpenAI")
        assert entities == []
    
    def test_detailed_extraction(self, keyword_extractor):
        """Test detailed keyword extraction with full result object."""
        text = "Machine learning models use neural networks for pattern recognition."
        
        result = keyword_extractor.extract_detailed(text)
        
        assert hasattr(result, 'keywords')
        assert hasattr(result, 'keyword_scores')
        assert hasattr(result, 'entities')
        assert len(result.keywords) > 0
        assert isinstance(result.keyword_scores, dict)


class TestDataAnalysisTool:
    """Test suite for data analysis tool."""
    
    @pytest.fixture
    def data_analyzer(self):
        return DataAnalysisTool()
    
    def test_trend_analysis_basic(self, data_analyzer):
        """Test basic trend analysis."""
        keywords_data = [
            ["machine", "learning", "neural", "network"],
            ["deep", "learning", "neural", "transformer"],
            ["artificial", "intelligence", "machine", "learning"]
        ]
        
        result = data_analyzer._run(json.dumps(keywords_data))
        
        assert "Analysis Complete" in result
        assert not result.startswith("Error:")
    
    def test_trend_analysis_empty_data(self, data_analyzer):
        """Test trend analysis with empty data."""
        result = data_analyzer._run("[]")
        assert result.startswith("Error:")
    
    def test_diversity_index_calculation(self, data_analyzer):
        """Test diversity index calculation."""
        # High diversity case
        high_div_freq = {f"keyword_{i}": 1 for i in range(10)}
        high_diversity = data_analyzer._calculate_diversity_index(high_div_freq)
        
        # Low diversity case
        low_div_freq = {"dominant": 9, "minor": 1}
        low_diversity = data_analyzer._calculate_diversity_index(low_div_freq)
        
        assert high_diversity > low_diversity
        assert 0 <= high_diversity <= 1
        assert 0 <= low_diversity <= 1
    
    def test_trend_score_calculation(self, data_analyzer):
        """Test trend score calculation."""
        # High consistency case
        high_consistency = [
            ["keyword1", "keyword2", "keyword3"],
            ["keyword1", "keyword2", "keyword4"],
            ["keyword1", "keyword2", "keyword5"]
        ]
        
        # Low consistency case
        low_consistency = [
            ["keyword1", "keyword2", "keyword3"],
            ["keyword4", "keyword5", "keyword6"],
            ["keyword7", "keyword8", "keyword9"]
        ]
        
        high_score = data_analyzer._calculate_trend_score(high_consistency)
        low_score = data_analyzer._calculate_trend_score(low_consistency)
        
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
    
    def test_keyword_clustering(self, data_analyzer):
        """Test keyword clustering."""
        keywords = ["neural", "network", "machine", "learning", "agent", "collaboration"]
        frequencies = {kw: 1 for kw in keywords}
        
        clusters = data_analyzer._cluster_keywords(keywords, frequencies)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        assert all(isinstance(cluster, list) for cluster in clusters)
    
    def test_co_occurrence_identification(self, data_analyzer):
        """Test co-occurrence pattern identification."""
        keywords_lists = [
            ["ai", "machine", "learning"],
            ["machine", "learning", "neural"],
            ["ai", "neural", "network"]
        ]
        
        co_occurrence = data_analyzer._identify_co_occurrences(keywords_lists)
        
        assert isinstance(co_occurrence, dict)
        assert "machine" in co_occurrence
        assert "learning" in co_occurrence.get("machine", {})


class TestPublicationAnalyzerAgent:
    """Test suite for publication analyzer agent."""
    
    @pytest.fixture
    def publication_analyzer(self):
        with patch('langchain_openai.ChatOpenAI'):
            return PublicationAnalyzerAgent()
    
    @patch.object(WebScrapingTool, '_run')
    @patch.object(KeywordExtractionTool, 'extract_detailed')
    def test_analyze_publication_success(self, mock_extract, mock_scrape, publication_analyzer):
        """Test successful publication analysis."""
        # Mock scraping result
        mock_scrape.return_value = "Test content about machine learning and AI"
        
        # Mock keyword extraction result
        mock_keyword_result = Mock()
        mock_keyword_result.keywords = ["machine", "learning", "ai"]
        mock_keyword_result.keyword_scores = {"machine": 0.8, "learning": 0.7, "ai": 0.9}
        mock_keyword_result.entities = [{"text": "OpenAI", "label": "ORG"}]
        mock_keyword_result.topics = ["machine learning", "artificial intelligence"]
        mock_keyword_result.sentiment = None
        mock_extract.return_value = mock_keyword_result
        
        result = publication_analyzer.analyze_publication("https://example.com")
        
        assert result['success'] is True
        assert result['url'] == "https://example.com"
        assert 'keywords' in result
        assert 'processing_time' in result
    
    @patch.object(WebScrapingTool, '_run')
    def test_analyze_publication_scraping_failure(self, mock_scrape, publication_analyzer):
        """Test publication analysis with scraping failure."""
        mock_scrape.return_value = "Error: Failed to scrape"
        
        result = publication_analyzer.analyze_publication("https://example.com")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_batch_analyze(self, publication_analyzer):
        """Test batch publication analysis."""
        publications = [
            {"url": "https://example1.com"},
            {"url": "https://example2.com"}
        ]
        
        with patch.object(publication_analyzer, 'analyze_publication') as mock_analyze:
            mock_analyze.return_value = {"success": True, "url": "test"}
            
            results = publication_analyzer.batch_analyze(publications)
            
            assert len(results) == 2
            assert mock_analyze.call_count == 2


class TestTrendAggregatorAgent:
    """Test suite for trend aggregator agent."""
    
    @pytest.fixture
    def trend_aggregator(self):
        with patch('langchain_openai.ChatOpenAI'):
            return TrendAggregatorAgent()
    
    def test_aggregate_trends_success(self, trend_aggregator):
        """Test successful trend aggregation."""
        publication_results = [
            {
                "success": True,
                "url": "https://example1.com",
                "keywords": ["machine", "learning", "ai"],
                "entities": [{"text": "OpenAI", "label": "ORG"}],
                "topics": ["machine learning"],
                "processing_time": 1.5
            },
            {
                "success": True,
                "url": "https://example2.com",
                "keywords": ["deep", "learning", "neural"],
                "entities": [{"text": "Google", "label": "ORG"}],
                "topics": ["deep learning"],
                "processing_time": 2.0
            }
        ]
        
        with patch.object(trend_aggregator.data_analyzer, '_run') as mock_run, \\
             patch.object(trend_aggregator.data_analyzer, 'get_detailed_analysis') as mock_analysis:
            
            mock_run.return_value = "Analysis complete"
            mock_analysis.return_value = Mock(
                keyword_frequencies={"learning": 2, "machine": 1},
                diversity_index=0.8,
                trend_score=0.6
            )
            
            result = trend_aggregator.aggregate_trends(publication_results)
            
            assert result['success'] is True
            assert result['publications_analyzed'] == 2
            assert 'enhanced_insights' in result
    
    def test_aggregate_trends_no_valid_results(self, trend_aggregator):
        """Test trend aggregation with no valid results."""
        publication_results = [
            {"success": False, "error": "Failed to analyze"}
        ]
        
        result = trend_aggregator.aggregate_trends(publication_results)
        
        assert result['success'] is False
        assert 'No valid publication results' in result['error']
    
    def test_cross_pattern_identification(self, trend_aggregator):
        """Test cross-publication pattern identification."""
        publication_data = [
            {
                "keywords": ["ai", "machine", "learning"],
                "topics": ["artificial intelligence"],
                "entities": [{"text": "OpenAI"}]
            },
            {
                "keywords": ["machine", "learning", "neural"],
                "topics": ["machine learning"],
                "entities": [{"text": "Google"}]
            }
        ]
        
        patterns = trend_aggregator._identify_cross_patterns(publication_data)
        
        assert 'keyword_cooccurrence' in patterns
        assert 'topic_distribution' in patterns
        assert 'entity_mentions' in patterns


class TestInsightGeneratorAgent:
    """Test suite for insight generator agent."""
    
    @pytest.fixture
    def insight_generator(self):
        with patch('langchain_openai.ChatOpenAI'):
            return InsightGeneratorAgent()
    
    def test_generate_insights_success(self, insight_generator):
        """Test successful insight generation."""
        trend_analysis = {
            "success": True,
            "publications_analyzed": 3,
            "unique_keywords": 25,
            "statistical_analysis": {
                "top_keywords": [{"keyword": "machine", "frequency": 5}],
                "trend_score": 0.7,
                "diversity_index": 0.6
            },
            "enhanced_insights": {
                "dominant_themes": ["AI automation"],
                "emerging_patterns": ["Multi-agent systems"],
                "research_directions": ["Collaborative AI"]
            },
            "trend_predictions": {
                "rising_trends": [{"keyword": "agents", "score": 8}],
                "breakthrough_indicators": []
            }
        }
        
        with patch.object(insight_generator.agent, 'llm') as mock_llm:
            mock_llm.predict.return_value = "Detailed analysis of AI trends shows significant growth in multi-agent systems."
            
            result = insight_generator.generate_insights(trend_analysis)
            
            assert result['success'] is True
            assert 'executive_summary' in result
            assert 'strategic_insights' in result
            assert 'recommendations' in result
    
    def test_generate_insights_invalid_data(self, insight_generator):
        """Test insight generation with invalid data."""
        trend_analysis = {"success": False, "error": "No data"}
        
        result = insight_generator.generate_insights(trend_analysis)
        
        assert result['success'] is False
        assert 'Invalid trend analysis data' in result['error']
    
    def test_confidence_level_calculation(self, insight_generator):
        """Test confidence level calculation."""
        # High confidence case
        high_conf_analysis = {
            "publications_analyzed": 15,
            "unique_keywords": 75,
            "statistical_analysis": {"trend_score": 0.8}
        }
        
        # Low confidence case
        low_conf_analysis = {
            "publications_analyzed": 2,
            "unique_keywords": 5,
            "statistical_analysis": {"trend_score": 0.2}
        }
        
        high_confidence = insight_generator._calculate_confidence_level(high_conf_analysis)
        low_confidence = insight_generator._calculate_confidence_level(low_conf_analysis)
        
        assert high_confidence in ['High', 'Medium-High']
        assert low_confidence in ['Low-Medium', 'Medium']


class TestPublicationInsightCrew:
    """Test suite for publication insight crew."""
    
    @pytest.fixture
    def crew(self):
        with patch('crewai.Crew'), \\
             patch('langchain_openai.ChatOpenAI'):
            return PublicationInsightCrew()
    
    def test_crew_initialization(self, crew):
        """Test crew initialization."""
        assert crew.publication_analyzer is not None
        assert crew.trend_aggregator is not None
        assert crew.insight_generator is not None
        assert crew.crew is not None
    
    def test_get_crew_status(self, crew):
        """Test crew status retrieval."""
        status = crew.get_crew_status()
        
        assert 'crew_initialized' in status
        assert 'agents_count' in status
        assert 'config' in status
        assert 'agents' in status


class TestPublicationAnalysisFlow:
    """Test suite for publication analysis flow."""
    
    @pytest.fixture
    def flow(self):
        with patch('crewai.flow.flow.Flow.__init__'):
            return create_publication_flow()
    
    def test_flow_creation(self, flow):
        """Test flow creation."""
        assert isinstance(flow, PublicationAnalysisFlow)
        assert hasattr(flow, 'crew')
    
    def test_input_validation(self, flow):
        """Test publication input validation."""
        publications = [
            {"url": "https://valid-example.com"},
            {"url": "invalid-url"},
            {"url": "https://blocked-domain.com"}
        ]
        
        # Mock the flow state and validation
        from src.flows.publication_flow import FlowState, AnalysisConfig, PublicationInput
        
        state = FlowState(
            publications=[PublicationInput(url=pub["url"]) for pub in publications],
            config=AnalysisConfig()
        )
        
        # Mock settings for domain validation
        with patch('config.settings.settings') as mock_settings:
            mock_settings.allowed_domains = ['valid-example.com']
            
            validated_state = flow.validate_inputs(state)
            
            assert len(validated_state.errors) > 0  # Should have validation errors
            assert len(validated_state.publications) < len(publications)  # Some should be filtered


class TestCrossPublicationInsightAssistant:
    """Test suite for main application class."""
    
    @pytest.fixture
    def assistant(self):
        with patch('config.settings.validate_configuration', return_value=True), \\
             patch('src.crews.publication_crew.PublicationInsightCrew'), \\
             patch('src.flows.publication_flow.create_publication_flow'):
            return CrossPublicationInsightAssistant()
    
    def test_assistant_initialization(self, assistant):
        """Test assistant initialization."""
        assert assistant.crew is not None
        assert assistant.flow is not None
    
    def test_analyze_publications_crew_method(self, assistant):
        """Test crew-based analysis method."""
        publications = [{"url": "https://example.com"}]
        
        with patch.object(assistant.crew, 'analyze_publications') as mock_analyze:
            mock_analyze.return_value = {"success": True, "method": "crew"}
            
            result = assistant.analyze_publications(publications, method="crew")
            
            assert result["success"] is True
            assert result["method"] == "crew"
    
    def test_analyze_publications_flow_method(self, assistant):
        """Test flow-based analysis method."""
        publications = [{"url": "https://example.com"}]
        
        with patch.object(assistant.flow, 'kickoff') as mock_kickoff:
            mock_state = Mock()
            mock_state.current_stage = "completed"
            mock_state.publication_results = []
            mock_state.trend_analysis = None
            mock_state.insights = None
            mock_state.errors = []
            mock_kickoff.return_value = mock_state
            
            with patch.object(assistant.flow, 'get_flow_summary') as mock_summary:
                mock_summary.return_value = {"status": "completed"}
                
                result = assistant.analyze_publications(publications, method="flow")
                
                assert result["success"] is True
                assert result["method"] == "flow"
    
    def test_invalid_analysis_method(self, assistant):
        """Test invalid analysis method."""
        publications = [{"url": "https://example.com"}]
        
        with pytest.raises(ValueError):
            assistant.analyze_publications(publications, method="invalid")
    
    def test_save_results(self, assistant, tmp_path):
        """Test results saving."""
        results = {"test": "data"}
        output_file = tmp_path / "test_results.json"
        
        assistant.save_results(results, str(output_file))
        
        assert output_file.exists()
        
        with open(output_file) as f:
            saved_data = json.load(f)
        
        assert saved_data == results


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @pytest.fixture
    def mock_environment(self):
        """Setup mock environment for integration tests."""
        with patch('requests.Session.get') as mock_get, \\
             patch('langchain_openai.ChatOpenAI') as mock_llm, \\
             patch('config.settings.validate_configuration', return_value=True):
            
            # Mock HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body><h1>AI Research</h1><p>This paper discusses machine learning and neural networks.</p></body></html>'
            mock_response.url = 'https://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            # Mock LLM response
            mock_llm_instance = Mock()
            mock_llm_instance.predict.return_value = "This research focuses on machine learning applications in AI systems."
            mock_llm.return_value = mock_llm_instance
            
            yield {
                'mock_get': mock_get,
                'mock_llm': mock_llm,
                'mock_llm_instance': mock_llm_instance
            }
    
    def test_end_to_end_crew_workflow(self, mock_environment):
        """Test complete end-to-end workflow using crew method."""
        assistant = CrossPublicationInsightAssistant()
        
        publications = [
            {"url": "https://example.com/paper1"},
            {"url": "https://example.com/paper2"}
        ]
        
        # Mock domain validation to allow example.com
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew",
                include_entities=True,
                include_sentiment=False,
                include_topics=True
            )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'method' in result
        assert result['method'] == 'crew'
    
    def test_error_handling_workflow(self, mock_environment):
        """Test error handling in the workflow."""
        # Make HTTP requests fail
        mock_environment['mock_get'].side_effect = Exception("Network error")
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [{"url": "https://example.com/paper1"}]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew"
            )
        
        # Should handle errors gracefully
        assert isinstance(result, dict)
        assert 'success' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
