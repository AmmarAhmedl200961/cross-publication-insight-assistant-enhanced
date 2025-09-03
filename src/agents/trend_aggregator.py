"""
Enhanced trend aggregator agent using CrewAI framework.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter, defaultdict

from crewai import Agent
from langchain_openai import ChatOpenAI

from ..tools.data_analyzer import DataAnalysisTool
from config.settings import get_llm_config


logger = logging.getLogger(__name__)


class TrendAggregatorAgent:
    """CrewAI agent for aggregating and analyzing trends across publications."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        self.data_analyzer = DataAnalysisTool()
        
        # Create the LLM instance
        self.llm = ChatOpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'],
            max_tokens=self.llm_config['max_tokens'],
            api_key=self.llm_config['api_key']
        )
        
        # Create the CrewAI agent
        self.agent = Agent(
            role='AI/ML Trend Analyst',
            goal='Aggregate and analyze trends across multiple AI/ML publications to identify patterns, emerging technologies, and research directions',
            backstory="""You are a senior AI/ML trend analyst with extensive experience in identifying 
            patterns across research publications, industry reports, and technical articles. Your expertise 
            includes statistical analysis, trend identification, and the ability to distinguish between 
            temporary fluctuations and significant shifts in the AI/ML landscape. You excel at synthesizing 
            information from multiple sources to provide actionable insights about technology adoption, 
            research directions, and emerging paradigms.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.data_analyzer],
            max_iter=3,
            memory=True
        )
    
    def aggregate_trends(self, publication_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate trends from multiple publication analysis results.
        
        Args:
            publication_results: List of publication analysis results
            
        Returns:
            Dictionary containing aggregated trend analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Aggregating trends from {len(publication_results)} publications")
            
            # Step 1: Extract and validate data
            valid_results = [r for r in publication_results if r.get('success', False)]
            
            if not valid_results:
                return {
                    'success': False,
                    'error': 'No valid publication results to aggregate',
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Collect all keywords and data
            all_keywords = []
            publication_data = []
            
            for result in valid_results:
                keywords = result.get('keywords', [])
                if keywords:
                    all_keywords.append(keywords)
                
                publication_data.append({
                    'url': result.get('url'),
                    'keywords': keywords,
                    'entities': result.get('entities', []),
                    'topics': result.get('topics', []),
                    'sentiment': result.get('sentiment'),
                    'processing_time': result.get('processing_time', 0),
                    'enhanced_analysis': result.get('enhanced_analysis', {})
                })
            
            # Step 3: Perform statistical analysis
            analysis_result = self.data_analyzer._run(all_keywords)
            detailed_analysis = self.data_analyzer.get_detailed_analysis()
            
            # Step 4: Get enhanced trend insights using CrewAI agent
            enhanced_insights = self._get_enhanced_trend_insights(
                publication_data, 
                detailed_analysis
            )
            
            # Step 5: Identify cross-publication patterns
            cross_patterns = self._identify_cross_patterns(publication_data)
            
            # Step 6: Generate trend predictions
            predictions = self._generate_trend_predictions(detailed_analysis, cross_patterns)
            
            # Step 7: Compile comprehensive results
            result = {
                'success': True,
                'publications_analyzed': len(valid_results),
                'total_keywords': len([kw for keywords in all_keywords for kw in keywords]),
                'unique_keywords': len(set([kw for keywords in all_keywords for kw in keywords])),
                'statistical_analysis': detailed_analysis.__dict__ if detailed_analysis else None,
                'enhanced_insights': enhanced_insights,
                'cross_patterns': cross_patterns,
                'trend_predictions': predictions,
                'publication_data': publication_data,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully aggregated trends in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating trends: {e}")
            return {
                'success': False,
                'error': f"Trend aggregation failed: {str(e)}",
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _get_enhanced_trend_insights(self, 
                                   publication_data: List[Dict[str, Any]], 
                                   analysis) -> Dict[str, Any]:
        """Use CrewAI agent to provide enhanced trend insights."""
        try:
            # Prepare data summary for the agent
            all_keywords = []
            all_topics = []
            all_entities = []
            
            for pub in publication_data:
                all_keywords.extend(pub.get('keywords', []))
                all_topics.extend(pub.get('topics', []))
                all_entities.extend([e.get('text', '') for e in pub.get('entities', [])])
            
            # Create keyword frequency summary
            keyword_freq = Counter(all_keywords)
            topic_freq = Counter(all_topics)
            entity_freq = Counter(all_entities)
            
            # Create analysis prompt
            insights_prompt = f"""
            Analyze the following aggregated data from {len(publication_data)} AI/ML publications:
            
            TOP KEYWORDS: {dict(keyword_freq.most_common(15))}
            
            TOP TOPICS: {dict(topic_freq.most_common(10))}
            
            TOP ENTITIES: {dict(entity_freq.most_common(10))}
            
            STATISTICAL METRICS:
            - Total publications: {len(publication_data)}
            - Unique keywords: {len(keyword_freq)}
            - Diversity index: {analysis.diversity_index if analysis else 'N/A'}
            - Trend coherence: {analysis.trend_score if analysis else 'N/A'}
            
            Please provide insights on:
            1. Dominant themes and their significance
            2. Emerging technology patterns
            3. Research direction trends
            4. Industry adoption indicators
            5. Potential future developments
            6. Key players and organizations mentioned
            7. Technology convergence patterns
            
            Focus on actionable insights for AI/ML professionals.
            """
            
            response = self.agent.llm.predict(insights_prompt)
            
            return {
                'dominant_themes': self._extract_themes(response),
                'emerging_patterns': self._extract_patterns(response),
                'research_directions': self._extract_directions(response),
                'adoption_indicators': self._extract_adoption(response),
                'future_developments': self._extract_future(response),
                'key_players': self._extract_players(response),
                'convergence_patterns': self._extract_convergence(response),
                'full_insights': response
            }
            
        except Exception as e:
            logger.error(f"Enhanced trend insights failed: {e}")
            return {
                'error': f"Enhanced insights failed: {str(e)}",
                'full_insights': None
            }
    
    def _identify_cross_patterns(self, publication_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns that appear across multiple publications."""
        
        # Keyword co-occurrence across publications
        keyword_cooccurrence = defaultdict(int)
        for pub in publication_data:
            keywords = pub.get('keywords', [])
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    pair = tuple(sorted([kw1, kw2]))
                    keyword_cooccurrence[pair] += 1
        
        # Topic consistency
        topic_distribution = defaultdict(int)
        for pub in publication_data:
            for topic in pub.get('topics', []):
                topic_distribution[topic] += 1
        
        # Entity mentions across publications
        entity_mentions = defaultdict(int)
        for pub in publication_data:
            for entity in pub.get('entities', []):
                entity_mentions[entity.get('text', '')] += 1
        
        # Technology stack patterns
        tech_patterns = self._identify_tech_patterns(publication_data)
        
        return {
            'keyword_cooccurrence': dict(sorted(keyword_cooccurrence.items(), 
                                               key=lambda x: x[1], reverse=True)[:20]),
            'topic_distribution': dict(sorted(topic_distribution.items(), 
                                            key=lambda x: x[1], reverse=True)),
            'entity_mentions': dict(sorted(entity_mentions.items(), 
                                         key=lambda x: x[1], reverse=True)[:15]),
            'technology_patterns': tech_patterns
        }
    
    def _identify_tech_patterns(self, publication_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify technology patterns across publications."""
        
        # Define technology categories
        tech_categories = {
            'frameworks': ['pytorch', 'tensorflow', 'keras', 'scikit-learn', 'huggingface'],
            'architectures': ['transformer', 'lstm', 'cnn', 'rnn', 'bert', 'gpt'],
            'approaches': ['supervised', 'unsupervised', 'reinforcement', 'federated'],
            'domains': ['nlp', 'cv', 'robotics', 'recommendation', 'speech'],
            'tools': ['jupyter', 'docker', 'kubernetes', 'mlflow', 'wandb']
        }
        
        category_counts = defaultdict(lambda: defaultdict(int))
        
        for pub in publication_data:
            keywords = [kw.lower() for kw in pub.get('keywords', [])]
            
            for category, techs in tech_categories.items():
                for tech in techs:
                    if any(tech in keyword for keyword in keywords):
                        category_counts[category][tech] += 1
        
        return {
            category: dict(sorted(techs.items(), key=lambda x: x[1], reverse=True))
            for category, techs in category_counts.items()
        }
    
    def _generate_trend_predictions(self, analysis, cross_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trend predictions based on analysis."""
        
        predictions = {
            'rising_trends': [],
            'stable_trends': [],
            'declining_trends': [],
            'breakthrough_indicators': [],
            'adoption_forecast': {},
            'confidence_scores': {}
        }
        
        if not analysis:
            return predictions
        
        # Identify rising trends (high frequency, high co-occurrence)
        keyword_freq = analysis.keyword_frequencies
        cooccurrence = cross_patterns.get('keyword_cooccurrence', {})
        
        for keyword, freq in keyword_freq.items():
            # Calculate trend score based on frequency and co-occurrence
            cooccurrence_score = sum(
                count for (kw1, kw2), count in cooccurrence.items() 
                if keyword in (kw1, kw2)
            )
            
            trend_score = freq + (cooccurrence_score * 0.5)
            
            if trend_score >= 5:
                predictions['rising_trends'].append({
                    'keyword': keyword,
                    'score': trend_score,
                    'frequency': freq
                })
            elif trend_score >= 3:
                predictions['stable_trends'].append({
                    'keyword': keyword,
                    'score': trend_score,
                    'frequency': freq
                })
        
        # Sort predictions by score
        predictions['rising_trends'].sort(key=lambda x: x['score'], reverse=True)
        predictions['stable_trends'].sort(key=lambda x: x['score'], reverse=True)
        
        # Identify breakthrough indicators (emerging keywords with high co-occurrence)
        for (kw1, kw2), count in cooccurrence.items():
            if count >= 2:  # Appears together in multiple publications
                predictions['breakthrough_indicators'].append({
                    'pattern': f"{kw1} + {kw2}",
                    'co_occurrence_count': count,
                    'significance': 'high' if count >= 3 else 'medium'
                })
        
        return predictions
    
    def _extract_themes(self, response: str) -> List[str]:
        """Extract dominant themes from agent response."""
        themes = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'theme' in line.lower() or 'dominant' in line.lower():
                # Extract themes from the line
                words = line.split()
                for i, word in enumerate(words):
                    if word.lower() in ['theme', 'themes', 'dominant']:
                        # Get context around theme mentions
                        start = max(0, i-2)
                        end = min(len(words), i+5)
                        theme_phrase = ' '.join(words[start:end])
                        themes.append(theme_phrase.strip('.,'))
        
        return themes[:5]
    
    def _extract_patterns(self, response: str) -> List[str]:
        """Extract emerging patterns from agent response."""
        patterns = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'emerging' in line.lower() or 'pattern' in line.lower():
                patterns.append(line.strip())
        
        return patterns[:5]
    
    def _extract_directions(self, response: str) -> List[str]:
        """Extract research directions from agent response."""
        directions = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'research' in line.lower() or 'direction' in line.lower():
                directions.append(line.strip())
        
        return directions[:5]
    
    def _extract_adoption(self, response: str) -> List[str]:
        """Extract adoption indicators from agent response."""
        indicators = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'adoption' in line.lower() or 'industry' in line.lower():
                indicators.append(line.strip())
        
        return indicators[:3]
    
    def _extract_future(self, response: str) -> List[str]:
        """Extract future developments from agent response."""
        developments = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'future' in line.lower() or 'development' in line.lower():
                developments.append(line.strip())
        
        return developments[:3]
    
    def _extract_players(self, response: str) -> List[str]:
        """Extract key players from agent response."""
        players = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'player' in line.lower() or 'organization' in line.lower():
                players.append(line.strip())
        
        return players[:3]
    
    def _extract_convergence(self, response: str) -> List[str]:
        """Extract convergence patterns from agent response."""
        convergence = []
        lines = response.split('\\n')
        
        for line in lines:
            if 'convergence' in line.lower() or 'combination' in line.lower():
                convergence.append(line.strip())
        
        return convergence[:3]
    
    def get_agent(self) -> Agent:
        """Get the underlying CrewAI agent."""
        return self.agent
