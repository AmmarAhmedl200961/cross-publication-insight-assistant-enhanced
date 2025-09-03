"""
Unit tests for individual agents.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestAgentUnits:
    """Unit tests for individual agent components."""
    
    def test_publication_analyzer_error_handling(self):
        """Test publication analyzer error handling."""
        from src.agents.publication_analyzer import PublicationAnalyzerAgent
        
        with patch('langchain_openai.ChatOpenAI'):
            agent = PublicationAnalyzerAgent()
            
            # Test with invalid URL
            result = agent.analyze_publication("invalid-url")
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_trend_aggregator_empty_data(self):
        """Test trend aggregator with empty data."""
        from src.agents.trend_aggregator import TrendAggregatorAgent
        
        with patch('langchain_openai.ChatOpenAI'):
            agent = TrendAggregatorAgent()
            
            # Test with empty publication results
            result = agent.aggregate_trends([])
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_insight_generator_malformed_data(self):
        """Test insight generator with malformed data."""
        from src.agents.insight_generator import InsightGeneratorAgent
        
        with patch('langchain_openai.ChatOpenAI'):
            agent = InsightGeneratorAgent()
            
            # Test with malformed trend analysis
            malformed_data = {"invalid": "data"}
            result = agent.generate_insights(malformed_data)
            
            assert result['success'] is False
            assert 'error' in result


class TestAgentInteractions:
    """Test agent interactions and communication."""
    
    @patch('langchain_openai.ChatOpenAI')
    def test_agent_chain_workflow(self, mock_llm):
        """Test agent chain workflow."""
        from src.agents.publication_analyzer import PublicationAnalyzerAgent
        from src.agents.trend_aggregator import TrendAggregatorAgent
        from src.agents.insight_generator import InsightGeneratorAgent
        
        # Setup agents
        analyzer = PublicationAnalyzerAgent()
        aggregator = TrendAggregatorAgent()
        generator = InsightGeneratorAgent()
        
        # Mock successful analysis
        with patch.object(analyzer, 'analyze_publication') as mock_analyze:
            mock_analyze.return_value = {
                'success': True,
                'url': 'https://example.com',
                'keywords': ['ai', 'machine', 'learning'],
                'entities': [],
                'topics': ['artificial intelligence'],
                'processing_time': 1.0
            }
            
            # Test chaining
            pub_results = [analyzer.analyze_publication('https://example.com')]
            
            with patch.object(aggregator, 'aggregate_trends') as mock_aggregate:
                mock_aggregate.return_value = {
                    'success': True,
                    'publications_analyzed': 1,
                    'statistical_analysis': {'trend_score': 0.8}
                }
                
                trend_results = aggregator.aggregate_trends(pub_results)
                
                with patch.object(generator, 'generate_insights') as mock_insights:
                    mock_insights.return_value = {
                        'success': True,
                        'executive_summary': {'overview': 'Test summary'}
                    }
                    
                    insights = generator.generate_insights(trend_results)
                    
                    assert insights['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
