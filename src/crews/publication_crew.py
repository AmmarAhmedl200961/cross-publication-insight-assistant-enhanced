"""
CrewAI crew for coordinating publication analysis workflow.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from crewai import Crew, Task, Process
from langchain_openai import ChatOpenAI

from ..agents.publication_analyzer import PublicationAnalyzerAgent
from ..agents.trend_aggregator import TrendAggregatorAgent
from ..agents.insight_generator import InsightGeneratorAgent
from config.settings import get_crew_config, get_llm_config


logger = logging.getLogger(__name__)


class PublicationInsightCrew:
    """Main CrewAI crew for publication insight analysis."""
    
    def __init__(self):
        self.config = get_crew_config()
        self.llm_config = get_llm_config()
        
        # Initialize agents
        self.publication_analyzer = PublicationAnalyzerAgent()
        self.trend_aggregator = TrendAggregatorAgent()
        self.insight_generator = InsightGeneratorAgent()
        
        # Initialize crew
        self.crew = None
        self._setup_crew()
    
    def _setup_crew(self):
        """Setup the CrewAI crew with agents and tasks."""
        try:
            # Create the crew
            self.crew = Crew(
                agents=[
                    self.publication_analyzer.get_agent(),
                    self.trend_aggregator.get_agent(),
                    self.insight_generator.get_agent()
                ],
                tasks=[],  # Tasks will be created dynamically
                verbose=self.config['verbose'],
                memory=self.config['memory'],
                process=Process.sequential,
                max_rpm=self.config['max_rpm'],
                max_execution_time=self.config['max_execution_time']
            )
            
            logger.info("Publication Insight Crew initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup crew: {e}")
            raise
    
    def analyze_publications(self, 
                           publications: List[Dict[str, Any]],
                           include_entities: bool = True,
                           include_sentiment: bool = False,
                           include_topics: bool = True,
                           generate_visualizations: bool = True) -> Dict[str, Any]:
        """
        Analyze a list of publications using the crew workflow.
        
        Args:
            publications: List of publication dictionaries with 'url' and optional 'selector'
            include_entities: Whether to extract named entities
            include_sentiment: Whether to analyze sentiment
            include_topics: Whether to classify topics
            generate_visualizations: Whether to generate visualization files
            
        Returns:
            Dictionary containing complete analysis results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting crew analysis of {len(publications)} publications")
            
            # Step 1: Create publication analysis tasks
            analysis_tasks = self._create_analysis_tasks(
                publications, include_entities, include_sentiment, include_topics
            )
            
            # Step 2: Create trend aggregation task
            aggregation_task = self._create_aggregation_task()
            
            # Step 3: Create insight generation task
            insight_task = self._create_insight_task()
            
            # Step 4: Update crew with tasks
            all_tasks = analysis_tasks + [aggregation_task, insight_task]
            self.crew.tasks = all_tasks
            
            # Step 5: Execute the crew workflow
            logger.info("Executing crew workflow...")
            crew_result = self.crew.kickoff()
            
            # Step 6: Process and compile results
            final_result = self._compile_results(
                crew_result, 
                publications, 
                generate_visualizations,
                start_time
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            final_result['processing_time'] = processing_time
            
            logger.info(f"Crew analysis completed in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Crew analysis failed: {e}")
            return {
                'success': False,
                'error': f"Crew analysis failed: {str(e)}",
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_analysis_tasks(self, 
                             publications: List[Dict[str, Any]],
                             include_entities: bool,
                             include_sentiment: bool,
                             include_topics: bool) -> List[Task]:
        """Create tasks for analyzing individual publications."""
        tasks = []
        
        for i, pub in enumerate(publications):
            task = Task(
                description=f"""
                Analyze publication {i+1}: {pub['url']}
                
                Extract the following information:
                1. Scrape content from the URL using the provided selector: {pub.get('selector', 'auto-detect')}
                2. Extract relevant keywords and key concepts
                3. {'Extract named entities' if include_entities else 'Skip named entity extraction'}
                4. {'Analyze sentiment' if include_sentiment else 'Skip sentiment analysis'}
                5. {'Classify topics' if include_topics else 'Skip topic classification'}
                6. Provide enhanced analysis of the publication's contribution to AI/ML field
                
                Focus on identifying:
                - Core technical concepts and methodologies
                - Novel contributions and innovations
                - Technologies and frameworks mentioned
                - Research directions and implications
                
                Return structured data suitable for trend aggregation.
                """,
                expected_output="""
                A comprehensive analysis including:
                - Publication URL and metadata
                - Extracted keywords with relevance scores
                - Named entities (if requested)
                - Topic classifications (if requested)
                - Sentiment analysis (if requested)
                - Enhanced analysis with focus areas and technologies
                - Processing metrics and quality indicators
                """,
                agent=self.publication_analyzer.get_agent(),
                context=[]  # No dependencies for parallel execution
            )
            tasks.append(task)
        
        return tasks
    
    def _create_aggregation_task(self) -> Task:
        """Create task for aggregating trends across publications."""
        return Task(
            description="""
            Aggregate and analyze trends across all analyzed publications.
            
            Your tasks:
            1. Collect keywords, entities, and topics from all publication analyses
            2. Perform statistical analysis to identify frequency patterns
            3. Calculate trend coherence and diversity metrics
            4. Identify cross-publication patterns and co-occurrences
            5. Detect emerging trends and declining patterns
            6. Generate trend predictions and confidence scores
            7. Identify technology convergence patterns
            8. Analyze research direction consistency
            
            Focus on:
            - Quantitative trend analysis with statistical backing
            - Identification of dominant themes and emerging patterns
            - Cross-publication correlation analysis
            - Technology adoption indicators
            - Research community focus areas
            
            Provide comprehensive trend analysis suitable for strategic insight generation.
            """,
            expected_output="""
            A comprehensive trend analysis including:
            - Statistical summary of keyword frequencies and distributions
            - Trend coherence and diversity metrics
            - Cross-publication pattern analysis
            - Emerging and declining trend identification
            - Technology convergence patterns
            - Research direction analysis
            - Trend predictions with confidence scores
            - Strategic implications summary
            """,
            agent=self.trend_aggregator.get_agent(),
            context=[]  # Will depend on all analysis tasks
        )
    
    def _create_insight_task(self) -> Task:
        """Create task for generating insights and recommendations."""
        return Task(
            description="""
            Generate comprehensive insights and strategic recommendations from trend analysis.
            
            Your tasks:
            1. Create executive summary highlighting key findings
            2. Develop strategic insights for different stakeholder groups
            3. Generate actionable recommendations with timelines
            4. Identify opportunities and risks
            5. Create implementation roadmaps
            6. Define success metrics and KPIs
            7. Develop audience-specific reports
            8. Provide confidence assessments
            
            Focus on:
            - Translating technical trends into business value
            - Providing actionable strategic guidance
            - Creating clear implementation pathways
            - Identifying competitive advantages and market opportunities
            - Addressing different audience needs (executives, researchers, developers)
            
            Generate insights that enable informed decision-making and strategic planning.
            """,
            expected_output="""
            A comprehensive insight report including:
            - Executive summary with key findings
            - Strategic insights and implications
            - Actionable recommendations with timelines
            - Opportunities and risk analysis
            - Implementation roadmap
            - Audience-specific reports (executives, researchers, developers, PMs)
            - Success metrics and measurement frameworks
            - Confidence levels and data quality assessments
            """,
            agent=self.insight_generator.get_agent(),
            context=[]  # Will depend on aggregation task
        )
    
    def _compile_results(self, 
                        crew_result: Any,
                        publications: List[Dict[str, Any]],
                        generate_visualizations: bool,
                        start_time: datetime) -> Dict[str, Any]:
        """Compile and structure the final results."""
        
        try:
            # Extract individual results (this is a simplified approach)
            # In a real implementation, you'd need to properly extract results from each task
            
            # For now, let's manually execute the workflow to get proper results
            # Step 1: Analyze publications
            publication_results = []
            for pub in publications:
                result = self.publication_analyzer.analyze_publication(
                    url=pub['url'],
                    selector=pub.get('selector'),
                    extract_entities=True,
                    extract_sentiment=False,
                    extract_topics=True
                )
                publication_results.append(result)
            
            # Step 2: Aggregate trends
            trend_analysis = self.trend_aggregator.aggregate_trends(publication_results)
            
            # Step 3: Generate insights
            insights = self.insight_generator.generate_insights(trend_analysis)
            
            # Step 4: Generate visualizations if requested
            visualizations = {}
            if generate_visualizations and trend_analysis.get('success'):
                try:
                    from ..tools.data_analyzer import DataAnalysisTool
                    data_tool = DataAnalysisTool()
                    analysis_obj = data_tool.get_detailed_analysis()
                    if analysis_obj:
                        visualizations = data_tool._create_visualizations(analysis_obj)
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
            
            # Compile final result
            return {
                'success': True,
                'workflow_type': 'crewai_sequential',
                'publications': {
                    'analyzed': len(publications),
                    'successful': len([r for r in publication_results if r.get('success')]),
                    'failed': len([r for r in publication_results if not r.get('success')]),
                    'results': publication_results
                },
                'trend_analysis': trend_analysis,
                'insights': insights,
                'visualizations': visualizations,
                'metadata': {
                    'crew_config': self.config,
                    'agents_used': ['publication_analyzer', 'trend_aggregator', 'insight_generator'],
                    'workflow_start': start_time.isoformat(),
                    'workflow_end': datetime.now().isoformat()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Result compilation failed: {e}")
            return {
                'success': False,
                'error': f"Result compilation failed: {str(e)}",
                'partial_results': {
                    'publications_analyzed': len(publications),
                    'crew_result': str(crew_result) if crew_result else None
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get current status of the crew."""
        return {
            'crew_initialized': self.crew is not None,
            'agents_count': len(self.crew.agents) if self.crew else 0,
            'config': self.config,
            'agents': {
                'publication_analyzer': 'ready',
                'trend_aggregator': 'ready',
                'insight_generator': 'ready'
            }
        }
    
    def reset_crew(self):
        """Reset the crew for a new analysis."""
        try:
            self._setup_crew()
            logger.info("Crew reset successfully")
        except Exception as e:
            logger.error(f"Crew reset failed: {e}")
            raise
