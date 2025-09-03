"""
Enhanced publication analyzer agent using CrewAI framework.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from crewai import Agent
from langchain_openai import ChatOpenAI

from ..tools.web_scraper import WebScrapingTool
from ..tools.keyword_extractor import KeywordExtractionTool
from config.settings import get_llm_config


logger = logging.getLogger(__name__)


class PublicationAnalyzerAgent:
    """CrewAI agent for analyzing publications and extracting insights."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.web_scraper = WebScrapingTool()
        self.keyword_extractor = KeywordExtractionTool()
        
        # Create the LLM instance
        self.llm = ChatOpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            max_tokens=self.llm_config['max_tokens'],
            api_key=self.llm_config['api_key']
        )
        
        # Create the CrewAI agent
        self.agent = Agent(
            role='Publication Content Analyst',
            goal='Extract and analyze content from AI/ML publications to identify key concepts, technologies, and methodologies',
            backstory="""You are an expert content analyst specializing in artificial intelligence and 
            machine learning publications. Your expertise lies in quickly identifying the core concepts, 
            emerging technologies, and research methodologies from academic papers, blog posts, and 
            technical articles. You have a deep understanding of AI/ML terminology and can distinguish 
            between fundamental concepts and novel contributions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.web_scraper, self.keyword_extractor],
            max_iter=3,
            memory=True
        )
    
    def analyze_publication(self, 
                          url: str, 
                          selector: Optional[str] = None,
                          extract_entities: bool = True,
                          extract_sentiment: bool = False,
                          extract_topics: bool = True) -> Dict[str, Any]:
        """
        Analyze a publication and extract comprehensive insights.
        
        Args:
            url: Publication URL to analyze
            selector: CSS selector for targeted content extraction
            extract_entities: Whether to extract named entities
            extract_sentiment: Whether to analyze sentiment
            extract_topics: Whether to classify topics
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting analysis of publication: {url}")
            
            # Step 1: Scrape content
            content = self.web_scraper._run(url, selector)
            
            if content.startswith("Error:"):
                return {
                    'url': url,
                    'success': False,
                    'error': content,
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Extract keywords and detailed analysis
            keyword_result = self.keyword_extractor.extract_detailed(
                content,
                top_k=25,
                include_entities=extract_entities,
                include_sentiment=extract_sentiment,
                include_topics=extract_topics
            )
            
            # Step 3: Use CrewAI agent for enhanced analysis
            enhanced_analysis = self._get_enhanced_analysis(content, keyword_result)
            
            # Step 4: Compile results
            result = {
                'url': url,
                'success': True,
                'content_preview': content[:500] + "..." if len(content) > 500 else content,
                'content_length': len(content),
                'keywords': keyword_result.keywords,
                'keyword_scores': keyword_result.keyword_scores,
                'entities': keyword_result.entities if extract_entities else [],
                'topics': keyword_result.topics if extract_topics else [],
                'sentiment': keyword_result.sentiment if extract_sentiment else None,
                'enhanced_analysis': enhanced_analysis,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully analyzed {url} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing publication {url}: {e}")
            return {
                'url': url,
                'success': False,
                'error': f"Analysis failed: {str(e)}",
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _get_enhanced_analysis(self, content: str, keyword_result) -> Dict[str, Any]:
        """Use CrewAI agent to provide enhanced analysis."""
        try:
            # Create a focused prompt for the agent
            analysis_prompt = f"""
            Analyze this AI/ML publication content and provide insights:
            
            Content Preview: {content[:1000]}...
            
            Extracted Keywords: {', '.join(keyword_result.keywords[:15])}
            
            Please provide:
            1. Main research focus and contributions
            2. Key technologies and methodologies mentioned
            3. Potential impact and significance
            4. Connections to current AI/ML trends
            5. Target audience and use cases
            
            Keep your analysis concise but insightful.
            """
            
            response = self.agent.llm.predict(analysis_prompt)
            
            return {
                'main_focus': self._extract_main_focus(response),
                'technologies': self._extract_technologies(response),
                'impact_assessment': self._extract_impact(response),
                'trend_connections': self._extract_trends(response),
                'target_audience': self._extract_audience(response),
                'full_analysis': response
            }
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            return {
                'error': f"Enhanced analysis failed: {str(e)}",
                'full_analysis': None
            }
    
    def _extract_main_focus(self, analysis: str) -> str:
        """Extract main research focus from analysis."""
        lines = analysis.split('\\n')
        for line in lines:
            if 'focus' in line.lower() or 'main' in line.lower():
                return line.strip()
        return analysis.split('.')[0] if analysis else "Not determined"
    
    def _extract_technologies(self, analysis: str) -> List[str]:
        """Extract mentioned technologies from analysis."""
        tech_keywords = [
            'neural network', 'transformer', 'bert', 'gpt', 'lstm',
            'pytorch', 'tensorflow', 'langchain', 'crewai', 'huggingface',
            'reinforcement learning', 'deep learning', 'machine learning',
            'computer vision', 'natural language', 'multi-agent'
        ]
        
        found_techs = []
        analysis_lower = analysis.lower()
        
        for tech in tech_keywords:
            if tech in analysis_lower:
                found_techs.append(tech)
        
        return found_techs
    
    def _extract_impact(self, analysis: str) -> str:
        """Extract impact assessment from analysis."""
        lines = analysis.split('\\n')
        for line in lines:
            if 'impact' in line.lower() or 'significance' in line.lower():
                return line.strip()
        return "Impact assessment not available"
    
    def _extract_trends(self, analysis: str) -> List[str]:
        """Extract trend connections from analysis."""
        trend_indicators = [
            'trend', 'emerging', 'future', 'direction', 'evolution',
            'advancement', 'development', 'innovation'
        ]
        
        connected_trends = []
        lines = analysis.split('\\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in trend_indicators):
                # Extract potential trend mentions
                words = line.split()
                for i, word in enumerate(words):
                    if any(indicator in word.lower() for indicator in trend_indicators):
                        # Get context around the trend indicator
                        start = max(0, i-2)
                        end = min(len(words), i+3)
                        trend_phrase = ' '.join(words[start:end])
                        connected_trends.append(trend_phrase)
        
        return connected_trends[:5]  # Return top 5 trend connections
    
    def _extract_audience(self, analysis: str) -> str:
        """Extract target audience from analysis."""
        audience_indicators = [
            'researcher', 'developer', 'practitioner', 'engineer',
            'scientist', 'student', 'professional', 'expert'
        ]
        
        analysis_lower = analysis.lower()
        found_audiences = []
        
        for audience in audience_indicators:
            if audience in analysis_lower:
                found_audiences.append(audience)
        
        if found_audiences:
            return ', '.join(found_audiences)
        
        return "AI/ML professionals and researchers"
    
    def batch_analyze(self, publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple publications in batch."""
        results = []
        
        for i, pub in enumerate(publications):
            logger.info(f"Analyzing publication {i+1}/{len(publications)}: {pub['url']}")
            
            result = self.analyze_publication(
                url=pub['url'],
                selector=pub.get('selector'),
                extract_entities=pub.get('extract_entities', True),
                extract_sentiment=pub.get('extract_sentiment', False),
                extract_topics=pub.get('extract_topics', True)
            )
            
            results.append(result)
        
        return results
    
    def get_agent(self) -> Agent:
        """Get the underlying CrewAI agent."""
        return self.agent
