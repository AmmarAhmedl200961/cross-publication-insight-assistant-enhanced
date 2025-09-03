"""
Main application entry point for the Cross-Publication Insight Assistant.
"""

import logging
import argparse
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings, validate_configuration, setup_logging
from src.crews.publication_crew import PublicationInsightCrew
from src.flows.publication_flow import create_publication_flow


logger = logging.getLogger(__name__)


class CrossPublicationInsightAssistant:
    """Main application class for the Cross-Publication Insight Assistant."""
    
    def __init__(self):
        self.crew = None
        self.flow = None
        
        # Validate configuration
        if not validate_configuration():
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize CrewAI components."""
        try:
            logger.info("Initializing Cross-Publication Insight Assistant...")
            
            # Initialize crew
            self.crew = PublicationInsightCrew()
            logger.info("CrewAI crew initialized")
            
            # Initialize flow
            self.flow = create_publication_flow()
            logger.info("CrewAI flow initialized")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def analyze_publications_crew(self, 
                                publications: List[Dict[str, Any]],
                                **kwargs) -> Dict[str, Any]:
        """
        Analyze publications using CrewAI crew workflow.
        
        Args:
            publications: List of publication dictionaries
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting crew-based analysis of {len(publications)} publications")
            
            result = self.crew.analyze_publications(
                publications=publications,
                include_entities=kwargs.get('include_entities', True),
                include_sentiment=kwargs.get('include_sentiment', False),
                include_topics=kwargs.get('include_topics', True),
                generate_visualizations=kwargs.get('generate_visualizations', True)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Crew analysis failed: {e}")
            return {
                'success': False,
                'error': f"Crew analysis failed: {str(e)}",
                'method': 'crew'
            }
    
    def analyze_publications_flow(self, 
                                publications: List[Dict[str, Any]],
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze publications using CrewAI flow workflow.
        
        Args:
            publications: List of publication dictionaries
            config: Flow configuration options
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting flow-based analysis of {len(publications)} publications")
            
            # Initialize flow state
            state = self.flow.kickoff(
                publications=publications,
                config=config or {}
            )
            
            # Get comprehensive results
            result = {
                'success': state.current_stage == 'completed',
                'method': 'flow',
                'workflow_summary': self.flow.get_flow_summary(state),
                'publication_results': state.publication_results,
                'trend_analysis': state.trend_analysis,
                'insights': state.insights,
                'errors': state.errors
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Flow analysis failed: {e}")
            return {
                'success': False,
                'error': f"Flow analysis failed: {str(e)}",
                'method': 'flow'
            }
    
    def analyze_publications(self, 
                           publications: List[Dict[str, Any]],
                           method: str = "crew",
                           **kwargs) -> Dict[str, Any]:
        """
        Analyze publications using specified method.
        
        Args:
            publications: List of publication dictionaries
            method: Analysis method ("crew" or "flow")
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing analysis results
        """
        if method == "crew":
            return self.analyze_publications_crew(publications, **kwargs)
        elif method == "flow":
            flow_config = {k: v for k, v in kwargs.items() 
                          if k in ['include_entities', 'include_sentiment', 'include_topics', 
                                  'generate_visualizations', 'parallel_processing', 'max_concurrent_publications']}
            return self.analyze_publications_flow(publications, flow_config)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save analysis results to file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the analysis results."""
        print("\\n" + "="*60)
        print("CROSS-PUBLICATION INSIGHT ASSISTANT - ANALYSIS SUMMARY")
        print("="*60)
        
        if not results.get('success'):
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return
        
        # Method and basic stats
        method = results.get('method', 'unknown')
        print(f"📊 Analysis Method: {method.upper()}")
        
        # Publications summary
        if 'publications' in results:
            pub_info = results['publications']
            print(f"📚 Publications: {pub_info.get('analyzed', 0)} analyzed, "
                  f"{pub_info.get('successful', 0)} successful, {pub_info.get('failed', 0)} failed")
        
        # Trend analysis summary
        trend_analysis = results.get('trend_analysis', {})
        if trend_analysis and trend_analysis.get('success'):
            stats = trend_analysis.get('statistical_analysis', {})
            if stats:
                print(f"🔍 Keywords: {stats.get('unique_keywords', 0)} unique, "
                      f"trend score: {stats.get('trend_score', 0):.2f}")
                
                top_keywords = stats.get('top_keywords', [])[:5]
                if top_keywords:
                    keywords_str = ", ".join([kw['keyword'] for kw in top_keywords])
                    print(f"🏷️  Top Keywords: {keywords_str}")
        
        # Insights summary
        insights = results.get('insights', {})
        if insights and insights.get('success'):
            recommendations = insights.get('recommendations', {})
            immediate_actions = recommendations.get('immediate_actions', [])
            if immediate_actions:
                print(f"💡 Immediate Actions: {len(immediate_actions)} recommendations")
            
            exec_summary = insights.get('executive_summary', {})
            if exec_summary and exec_summary.get('overview'):
                print(f"📋 Summary: {exec_summary['overview'][:100]}...")
        
        # Processing time
        processing_time = results.get('processing_time')
        if processing_time:
            print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        
        print("="*60 + "\\n")


def main():
    """Main entry point for the application."""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Cross-Publication Insight Assistant - AI/ML Publication Trend Analysis"
    )
    
    parser.add_argument(
        '--urls', 
        nargs='+', 
        help='URLs of publications to analyze'
    )
    
    parser.add_argument(
        '--input-file', 
        type=str, 
        help='JSON file containing publication URLs and configurations'
    )
    
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='analysis_results.json',
        help='Output file for analysis results (default: analysis_results.json)'
    )
    
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['crew', 'flow'], 
        default='crew',
        help='Analysis method: crew (sequential) or flow (advanced orchestration)'
    )
    
    parser.add_argument(
        '--include-entities', 
        action='store_true', 
        default=True,
        help='Include named entity extraction'
    )
    
    parser.add_argument(
        '--include-sentiment', 
        action='store_true',
        help='Include sentiment analysis'
    )
    
    parser.add_argument(
        '--include-topics', 
        action='store_true', 
        default=True,
        help='Include topic classification'
    )
    
    parser.add_argument(
        '--no-visualizations', 
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Enable parallel processing (flow method only)'
    )
    
    parser.add_argument(
        '--max-concurrent', 
        type=int, 
        default=3,
        help='Maximum concurrent publications for parallel processing'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run with demo publications'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize the assistant
        assistant = CrossPublicationInsightAssistant()
        
        # Prepare publications list
        publications = []
        
        if args.demo:
            # Use demo publications
            publications = [
                {
                    "url": "https://blog.langchain.dev/langgraph-multi-agent-workflows/",
                    "selector": "article"
                },
                {
                    "url": "https://app.readytensor.ai/publications/agentconnect-decentralized-collaboration-framework-for-independent-ai-agents-RLFuglEDiwwS",
                    "selector": "ul._cx"
                }
            ]
            logger.info("Using demo publications")
            
        elif args.input_file:
            # Load from input file
            with open(args.input_file, 'r') as f:
                data = json.load(f)
                publications = data.get('publications', data) if isinstance(data, dict) else data
            logger.info(f"Loaded {len(publications)} publications from {args.input_file}")
            
        elif args.urls:
            # Use provided URLs
            publications = [{"url": url} for url in args.urls]
            logger.info(f"Using {len(publications)} publications from command line")
            
        else:
            logger.error("No publications provided. Use --urls, --input-file, or --demo")
            sys.exit(1)
        
        if not publications:
            logger.error("No publications to analyze")
            sys.exit(1)
        
        # Prepare analysis configuration
        config = {
            'include_entities': args.include_entities,
            'include_sentiment': args.include_sentiment,
            'include_topics': args.include_topics,
            'generate_visualizations': not args.no_visualizations,
            'parallel_processing': args.parallel,
            'max_concurrent_publications': args.max_concurrent
        }
        
        # Run analysis
        logger.info(f"Starting analysis with method: {args.method}")
        results = assistant.analyze_publications(
            publications=publications,
            method=args.method,
            **config
        )
        
        # Save results
        assistant.save_results(results, args.output_file)
        
        # Print summary
        assistant.print_summary(results)
        
        # Exit with appropriate code
        if results.get('success'):
            logger.info("Analysis completed successfully")
            sys.exit(0)
        else:
            logger.error("Analysis failed")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
