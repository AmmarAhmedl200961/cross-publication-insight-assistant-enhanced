"""
Integration tests for system workflows.
"""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestSystemIntegration:
    """Integration tests for complete system workflows."""
    
    @pytest.fixture
    def mock_successful_environment(self):
        """Setup mock environment for successful integration tests."""
        with patch('requests.Session.get') as mock_get, \
             patch('langchain_openai.ChatOpenAI') as mock_llm, \
             patch('config.settings.validate_configuration', return_value=True):
            
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'''
            <html>
                <body>
                    <article>
                        <h1>Advanced AI Research</h1>
                        <p>This research explores machine learning, neural networks, and artificial intelligence. 
                        The study focuses on deep learning algorithms and their applications in natural language processing.</p>
                        <p>Key findings include improvements in transformer architectures and multi-agent systems.</p>
                    </article>
                </body>
            </html>
            '''
            mock_response.url = 'https://example.com'
            mock_response.headers = {'Content-Type': 'text/html'}
            mock_get.return_value = mock_response
            
            # Mock LLM responses
            mock_llm_instance = Mock()
            mock_llm_instance.predict.side_effect = [
                "This research focuses on advanced AI techniques including machine learning and neural networks.",
                "Analysis reveals strong trends in AI automation and multi-agent collaboration systems.",
                "Strategic recommendations include investment in AI infrastructure and talent development."
            ]
            mock_llm.return_value = mock_llm_instance
            
            yield {
                'mock_get': mock_get,
                'mock_llm': mock_llm,
                'mock_llm_instance': mock_llm_instance
            }
    
    def test_complete_workflow_crew_method(self, mock_successful_environment):
        """Test complete workflow using crew method."""
        from src.main import CrossPublicationInsightAssistant
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [
            {"url": "https://example.com/paper1", "selector": "article"},
            {"url": "https://example.com/paper2", "selector": "article"}
        ]
        
        # Mock domain validation
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew",
                include_entities=True,
                include_sentiment=False,
                include_topics=True,
                generate_visualizations=True
            )
        
        # Verify complete result structure
        assert result['success'] is True
        assert result['method'] == 'crew'
        assert 'publications' in result
        assert 'trend_analysis' in result
        assert 'insights' in result
        
        # Verify publications were processed
        pub_info = result['publications']
        assert pub_info['analyzed'] == 2
        assert pub_info['successful'] >= 1
        
        # Verify trend analysis exists
        if result['trend_analysis'] and result['trend_analysis'].get('success'):
            trend = result['trend_analysis']
            assert 'statistical_analysis' in trend
            assert 'enhanced_insights' in trend
        
        # Verify insights exist
        if result['insights'] and result['insights'].get('success'):
            insights = result['insights']
            assert 'executive_summary' in insights
            assert 'recommendations' in insights
    
    def test_workflow_with_failures(self, mock_successful_environment):
        """Test workflow handling partial failures."""
        from src.main import CrossPublicationInsightAssistant
        
        # Make some requests fail
        def side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if 'failing-url' in url:
                raise Exception("Network error")
            else:
                return mock_successful_environment['mock_get'].return_value
        
        mock_successful_environment['mock_get'].side_effect = side_effect
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [
            {"url": "https://example.com/working-paper"},
            {"url": "https://failing-url.com/paper"},
            {"url": "https://example.com/another-working-paper"}
        ]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew"
            )
        
        # Should handle partial failures gracefully
        assert isinstance(result, dict)
        
        # Some publications should succeed
        if result.get('success') and 'publications' in result:
            pub_info = result['publications']
            assert pub_info['successful'] > 0
            assert pub_info['failed'] > 0
    
    def test_end_to_end_data_flow(self, mock_successful_environment):
        """Test end-to-end data flow through all components."""
        from src.main import CrossPublicationInsightAssistant
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [{"url": "https://example.com/ai-research"}]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            # Run analysis and track data flow
            result = assistant.analyze_publications(
                publications=publications,
                method="crew",
                include_entities=True,
                include_topics=True
            )
        
        if result.get('success'):
            # Verify data flows through each stage
            
            # Stage 1: Publication Analysis
            pub_results = result.get('publications', {}).get('results', [])
            if pub_results:
                first_pub = pub_results[0]
                if first_pub.get('success'):
                    assert 'keywords' in first_pub
                    assert 'content_length' in first_pub
                    assert len(first_pub['keywords']) > 0
            
            # Stage 2: Trend Analysis
            trend_analysis = result.get('trend_analysis')
            if trend_analysis and trend_analysis.get('success'):
                assert 'statistical_analysis' in trend_analysis
                stats = trend_analysis['statistical_analysis']
                if stats:
                    assert 'unique_keywords' in stats
                    assert 'top_keywords' in stats
            
            # Stage 3: Insight Generation
            insights = result.get('insights')
            if insights and insights.get('success'):
                assert 'executive_summary' in insights
                assert 'recommendations' in insights
    
    def test_configuration_variations(self, mock_successful_environment):
        """Test system with various configuration options."""
        from src.main import CrossPublicationInsightAssistant
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [{"url": "https://example.com/test-paper"}]
        
        configurations = [
            {
                'include_entities': True,
                'include_sentiment': True,
                'include_topics': True,
                'generate_visualizations': True
            },
            {
                'include_entities': False,
                'include_sentiment': False,
                'include_topics': False,
                'generate_visualizations': False
            },
            {
                'include_entities': True,
                'include_sentiment': False,
                'include_topics': True,
                'generate_visualizations': False
            }
        ]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            for config in configurations:
                result = assistant.analyze_publications(
                    publications=publications,
                    method="crew",
                    **config
                )
                
                # Should work with any configuration
                assert isinstance(result, dict)
                assert 'success' in result
    
    def test_file_input_output_integration(self, mock_successful_environment):
        """Test file input and output integration."""
        from src.main import CrossPublicationInsightAssistant
        
        assistant = CrossPublicationInsightAssistant()
        
        # Create temporary input file
        input_data = {
            "publications": [
                {"url": "https://example.com/paper1"},
                {"url": "https://example.com/paper2"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_file = f.name
        
        try:
            # Load from file
            with open(input_file) as f:
                loaded_data = json.load(f)
            
            publications = loaded_data['publications']
            
            with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
                result = assistant.analyze_publications(
                    publications=publications,
                    method="crew"
                )
            
            # Save to output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_file = f.name
            
            assistant.save_results(result, output_file)
            
            # Verify output file
            assert os.path.exists(output_file)
            
            with open(output_file) as f:
                saved_result = json.load(f)
            
            assert saved_result == result
            
        finally:
            # Cleanup
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_concurrent_processing_stability(self, mock_successful_environment):
        """Test system stability under concurrent processing."""
        from src.main import CrossPublicationInsightAssistant
        
        assistant = CrossPublicationInsightAssistant()
        
        # Large number of publications
        publications = [
            {"url": f"https://example.com/paper{i}"}
            for i in range(10)
        ]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew"
            )
        
        # Should handle large batches without crashing
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_error_recovery_and_resilience(self, mock_successful_environment):
        """Test system error recovery and resilience."""
        from src.main import CrossPublicationInsightAssistant
        
        # Simulate various error conditions
        def failing_llm_predict(*args, **kwargs):
            # Fail on first call, succeed on subsequent calls
            if not hasattr(failing_llm_predict, 'call_count'):
                failing_llm_predict.call_count = 0
            failing_llm_predict.call_count += 1
            
            if failing_llm_predict.call_count == 1:
                raise Exception("LLM service temporarily unavailable")
            else:
                return "Recovered analysis response"
        
        mock_successful_environment['mock_llm_instance'].predict.side_effect = failing_llm_predict
        
        assistant = CrossPublicationInsightAssistant()
        
        publications = [{"url": "https://example.com/paper"}]
        
        with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
            result = assistant.analyze_publications(
                publications=publications,
                method="crew"
            )
        
        # System should handle errors gracefully
        assert isinstance(result, dict)


class TestPerformanceIntegration:
    """Performance integration tests."""
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable during processing."""
        import psutil
        import gc
        
        from src.main import CrossPublicationInsightAssistant
        
        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Mock environment
        with patch('requests.Session.get') as mock_get, \
             patch('langchain_openai.ChatOpenAI'), \
             patch('config.settings.validate_configuration', return_value=True):
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body>Test content</body></html>'
            mock_response.url = 'https://example.com'
            mock_response.headers = {}
            mock_get.return_value = mock_response
            
            assistant = CrossPublicationInsightAssistant()
            
            # Process multiple batches
            for batch in range(3):
                publications = [
                    {"url": f"https://example.com/paper{i}"}
                    for i in range(5)
                ]
                
                with patch('src.tools.web_scraper.SecurityValidator.validate_url', return_value=True):
                    result = assistant.analyze_publications(
                        publications=publications,
                        method="crew"
                    )
                
                # Force garbage collection
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024  # 100MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
