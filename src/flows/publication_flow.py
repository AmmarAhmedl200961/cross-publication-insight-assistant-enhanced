"""
CrewAI Flow for advanced publication analysis workflow orchestration.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import json

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

from ..crews.publication_crew import PublicationInsightCrew
from config.settings import settings


logger = logging.getLogger(__name__)


class PublicationInput(BaseModel):
    """Input model for publication analysis."""
    url: str
    selector: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    metadata: Optional[Dict[str, Any]] = None


class AnalysisConfig(BaseModel):
    """Configuration for analysis workflow."""
    include_entities: bool = True
    include_sentiment: bool = False
    include_topics: bool = True
    generate_visualizations: bool = True
    enable_caching: bool = True
    parallel_processing: bool = True
    max_concurrent_publications: int = 3
    retry_failed_publications: bool = True
    output_format: str = "comprehensive"  # "summary", "detailed", "comprehensive"


class FlowState(BaseModel):
    """State tracking for the flow."""
    publications: List[PublicationInput]
    config: AnalysisConfig
    current_stage: str = "initialized"
    publication_results: List[Dict[str, Any]] = []
    trend_analysis: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class PublicationAnalysisFlow(Flow[FlowState]):
    """Advanced flow for publication analysis with state management and error recovery."""
    
    def __init__(self):
        super().__init__()
        self.crew = PublicationInsightCrew()
    
    @start()
    def initialize_workflow(self, 
                          publications: List[Dict[str, Any]], 
                          config: Optional[Dict[str, Any]] = None) -> FlowState:
        """Initialize the workflow with publications and configuration."""
        
        logger.info(f"Initializing workflow with {len(publications)} publications")
        
        # Convert to Pydantic models
        publication_inputs = []
        for i, pub in enumerate(publications):
            publication_inputs.append(PublicationInput(
                url=pub['url'],
                selector=pub.get('selector'),
                priority=pub.get('priority', 1),
                metadata=pub.get('metadata', {})
            ))
        
        # Setup configuration
        analysis_config = AnalysisConfig(**(config or {}))
        
        # Create initial state
        state = FlowState(
            publications=publication_inputs,
            config=analysis_config,
            current_stage="initialized",
            started_at=datetime.now()
        )
        
        logger.info("Workflow initialized successfully")
        return state
    
    @listen(initialize_workflow)
    def validate_inputs(self, state: FlowState) -> FlowState:
        """Validate input publications and configuration."""
        
        logger.info("Validating inputs...")
        state.current_stage = "validating"
        
        # Validate publications
        valid_publications = []
        for pub in state.publications:
            try:
                # Basic URL validation
                if not pub.url or not pub.url.startswith(('http://', 'https://')):
                    state.errors.append(f"Invalid URL: {pub.url}")
                    continue
                
                # Check if domain is allowed (if restrictions exist)
                if settings.allowed_domains:
                    from urllib.parse import urlparse
                    domain = urlparse(pub.url).netloc.lower()
                    
                    allowed = False
                    for allowed_domain in settings.allowed_domains:
                        if allowed_domain.startswith('*'):
                            if domain.endswith(allowed_domain[1:]):
                                allowed = True
                                break
                        elif domain == allowed_domain or domain.endswith('.' + allowed_domain):
                            allowed = True
                            break
                    
                    if not allowed:
                        state.errors.append(f"Domain not allowed: {domain}")
                        continue
                
                valid_publications.append(pub)
                
            except Exception as e:
                state.errors.append(f"Validation error for {pub.url}: {str(e)}")
        
        state.publications = valid_publications
        
        if not valid_publications:
            state.errors.append("No valid publications to analyze")
            state.current_stage = "failed"
        else:
            logger.info(f"Validation complete: {len(valid_publications)} valid publications")
        
        return state
    
    @listen(validate_inputs)
    def analyze_publications(self, state: FlowState) -> FlowState:
        """Analyze individual publications."""
        
        if state.current_stage == "failed":
            return state
        
        logger.info("Starting publication analysis...")
        state.current_stage = "analyzing_publications"
        
        try:
            if state.config.parallel_processing:
                results = self._analyze_publications_parallel(state)
            else:
                results = self._analyze_publications_sequential(state)
            
            state.publication_results = results
            
            # Check if we have any successful results
            successful_results = [r for r in results if r.get('success', False)]
            if not successful_results:
                state.errors.append("All publication analyses failed")
                state.current_stage = "failed"
            else:
                logger.info(f"Publication analysis complete: {len(successful_results)} successful")
            
        except Exception as e:
            error_msg = f"Publication analysis failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.current_stage = "failed"
        
        return state
    
    @listen(analyze_publications)
    def aggregate_trends(self, state: FlowState) -> FlowState:
        """Aggregate trends from publication analyses."""
        
        if state.current_stage == "failed":
            return state
        
        logger.info("Starting trend aggregation...")
        state.current_stage = "aggregating_trends"
        
        try:
            # Filter successful results
            successful_results = [r for r in state.publication_results if r.get('success', False)]
            
            if not successful_results:
                state.errors.append("No successful publication results for trend aggregation")
                state.current_stage = "failed"
                return state
            
            # Perform trend aggregation
            trend_analysis = self.crew.trend_aggregator.aggregate_trends(successful_results)
            state.trend_analysis = trend_analysis
            
            if not trend_analysis.get('success', False):
                state.errors.append(f"Trend aggregation failed: {trend_analysis.get('error', 'Unknown error')}")
                state.current_stage = "failed"
            else:
                logger.info("Trend aggregation complete")
            
        except Exception as e:
            error_msg = f"Trend aggregation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.current_stage = "failed"
        
        return state
    
    @listen(aggregate_trends)
    def generate_insights(self, state: FlowState) -> FlowState:
        """Generate insights and recommendations."""
        
        if state.current_stage == "failed":
            return state
        
        logger.info("Starting insight generation...")
        state.current_stage = "generating_insights"
        
        try:
            if not state.trend_analysis or not state.trend_analysis.get('success'):
                state.errors.append("No valid trend analysis for insight generation")
                state.current_stage = "failed"
                return state
            
            # Generate insights
            insights = self.crew.insight_generator.generate_insights(state.trend_analysis)
            state.insights = insights
            
            if not insights.get('success', False):
                state.errors.append(f"Insight generation failed: {insights.get('error', 'Unknown error')}")
                state.current_stage = "failed"
            else:
                logger.info("Insight generation complete")
                state.current_stage = "completed"
                state.completed_at = datetime.now()
            
        except Exception as e:
            error_msg = f"Insight generation failed: {str(e)}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.current_stage = "failed"
        
        return state
    
    @listen(generate_insights)
    def finalize_workflow(self, state: FlowState) -> FlowState:
        """Finalize the workflow and prepare output."""
        
        logger.info("Finalizing workflow...")
        
        # Set completion time if not already set
        if not state.completed_at:
            state.completed_at = datetime.now()
        
        # Calculate total processing time
        if state.started_at and state.completed_at:
            processing_time = (state.completed_at - state.started_at).total_seconds()
            logger.info(f"Workflow completed in {processing_time:.2f} seconds")
        
        # Generate final summary
        if state.current_stage == "completed":
            logger.info("Workflow completed successfully")
        else:
            logger.error(f"Workflow failed at stage: {state.current_stage}")
            logger.error(f"Errors: {state.errors}")
        
        return state
    
    def _analyze_publications_parallel(self, state: FlowState) -> List[Dict[str, Any]]:
        """Analyze publications in parallel."""
        
        import concurrent.futures
        from functools import partial
        
        def analyze_single_publication(pub: PublicationInput) -> Dict[str, Any]:
            """Analyze a single publication."""
            try:
                return self.crew.publication_analyzer.analyze_publication(
                    url=pub.url,
                    selector=pub.selector,
                    extract_entities=state.config.include_entities,
                    extract_sentiment=state.config.include_sentiment,
                    extract_topics=state.config.include_topics
                )
            except Exception as e:
                return {
                    'url': pub.url,
                    'success': False,
                    'error': f"Analysis failed: {str(e)}"
                }
        
        # Sort publications by priority
        sorted_pubs = sorted(state.publications, key=lambda x: x.priority)
        
        results = []
        max_workers = min(state.config.max_concurrent_publications, len(sorted_pubs))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pub = {
                executor.submit(analyze_single_publication, pub): pub 
                for pub in sorted_pubs
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_pub):
                pub = future_to_pub[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success'):
                        logger.info(f"Successfully analyzed: {pub.url}")
                    else:
                        logger.warning(f"Failed to analyze: {pub.url} - {result.get('error')}")
                        
                except Exception as e:
                    error_result = {
                        'url': pub.url,
                        'success': False,
                        'error': f"Execution failed: {str(e)}"
                    }
                    results.append(error_result)
                    logger.error(f"Exception analyzing {pub.url}: {e}")
        
        return results
    
    def _analyze_publications_sequential(self, state: FlowState) -> List[Dict[str, Any]]:
        """Analyze publications sequentially."""
        
        results = []
        
        # Sort publications by priority
        sorted_pubs = sorted(state.publications, key=lambda x: x.priority)
        
        for i, pub in enumerate(sorted_pubs):
            logger.info(f"Analyzing publication {i+1}/{len(sorted_pubs)}: {pub.url}")
            
            try:
                result = self.crew.publication_analyzer.analyze_publication(
                    url=pub.url,
                    selector=pub.selector,
                    extract_entities=state.config.include_entities,
                    extract_sentiment=state.config.include_sentiment,
                    extract_topics=state.config.include_topics
                )
                results.append(result)
                
                if result.get('success'):
                    logger.info(f"Successfully analyzed: {pub.url}")
                else:
                    logger.warning(f"Failed to analyze: {pub.url} - {result.get('error')}")
                    
            except Exception as e:
                error_result = {
                    'url': pub.url,
                    'success': False,
                    'error': f"Analysis failed: {str(e)}"
                }
                results.append(error_result)
                logger.error(f"Exception analyzing {pub.url}: {e}")
        
        return results
    
    def get_flow_summary(self, state: FlowState) -> Dict[str, Any]:
        """Get a summary of the flow execution."""
        
        processing_time = None
        if state.started_at and state.completed_at:
            processing_time = (state.completed_at - state.started_at).total_seconds()
        
        successful_publications = len([r for r in state.publication_results if r.get('success')])
        
        return {
            'flow_id': id(self),
            'status': state.current_stage,
            'publications': {
                'total': len(state.publications),
                'successful': successful_publications,
                'failed': len(state.publication_results) - successful_publications
            },
            'has_trend_analysis': state.trend_analysis is not None and state.trend_analysis.get('success'),
            'has_insights': state.insights is not None and state.insights.get('success'),
            'errors': state.errors,
            'processing_time': processing_time,
            'started_at': state.started_at.isoformat() if state.started_at else None,
            'completed_at': state.completed_at.isoformat() if state.completed_at else None,
            'config': state.config.dict()
        }
    
    def export_results(self, state: FlowState, format: str = "json") -> str:
        """Export flow results in specified format."""
        
        if format == "json":
            # Create comprehensive results dictionary
            results = {
                'workflow_summary': self.get_flow_summary(state),
                'publication_results': state.publication_results,
                'trend_analysis': state.trend_analysis,
                'insights': state.insights,
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'flow_version': '1.0.0',
                    'config': state.config.dict()
                }
            }
            
            return json.dumps(results, indent=2, default=str)
        
        elif format == "summary":
            # Create text summary
            summary = f"""
Publication Analysis Workflow Summary
====================================

Status: {state.current_stage}
Publications Analyzed: {len(state.publications)}
Successful Analyses: {len([r for r in state.publication_results if r.get('success')])}

Processing Time: {(state.completed_at - state.started_at).total_seconds():.2f}s if state.started_at and state.completed_at else 'N/A'}

Errors: {len(state.errors)}
{chr(10).join(f"- {error}" for error in state.errors) if state.errors else "None"}

Top Keywords: {', '.join([kw['keyword'] for kw in state.trend_analysis.get('statistical_analysis', {}).get('top_keywords', [])[:5]]) if state.trend_analysis else 'N/A'}

Key Insights: {len(state.insights.get('recommendations', {}).get('immediate_actions', [])) if state.insights else 0} immediate actions identified
"""
            return summary.strip()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Factory function for creating flows
def create_publication_flow() -> PublicationAnalysisFlow:
    """Create a new publication analysis flow."""
    return PublicationAnalysisFlow()
