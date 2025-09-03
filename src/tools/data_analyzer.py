"""
Enhanced data analysis tool for trend analysis and insights.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from crewai_tools import BaseTool
from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    top_keywords: List[Dict[str, Any]]
    keyword_frequencies: Dict[str, int]
    co_occurrence_matrix: Dict[str, Dict[str, int]]
    trend_score: float
    diversity_index: float
    emerging_keywords: List[str]
    declining_keywords: List[str]
    clusters: List[List[str]]
    summary_stats: Dict[str, Any]
    recommendations: List[str]


@dataclass
class PublicationInsight:
    """Insight from a single publication."""
    url: str
    title: Optional[str]
    keywords: List[str]
    entities: List[Dict[str, str]]
    topics: List[str]
    sentiment: Optional[Dict[str, float]]
    processing_time: float
    timestamp: datetime


@dataclass
class ComprehensiveReport:
    """Comprehensive analysis report."""
    analysis_id: str
    timestamp: datetime
    publications_analyzed: int
    total_keywords: int
    unique_keywords: int
    trend_analysis: TrendAnalysis
    publication_insights: List[PublicationInsight]
    visualizations: Dict[str, str]  # file paths to generated plots
    executive_summary: str


class DataAnalysisTool(BaseTool):
    """Advanced data analysis tool for publication trends."""
    
    name: str = "data_analyzer"
    description: str = "Analyzes publication data to identify trends, patterns, and insights"
    
    def __init__(self):
        super().__init__()
        self.analysis_history = []
    
    def _calculate_diversity_index(self, frequencies: Dict[str, int]) -> float:
        """Calculate Shannon diversity index for keyword distribution."""
        total = sum(frequencies.values())
        if total == 0:
            return 0.0
        
        proportions = [count / total for count in frequencies.values()]
        shannon_index = -sum(p * np.log(p) for p in proportions if p > 0)
        
        # Normalize by maximum possible diversity
        max_diversity = np.log(len(frequencies)) if len(frequencies) > 1 else 1
        return shannon_index / max_diversity if max_diversity > 0 else 0
    
    def _calculate_trend_score(self, keywords: List[List[str]]) -> float:
        """Calculate overall trend strength based on keyword consistency."""
        if len(keywords) < 2:
            return 0.0
        
        # Calculate overlap between consecutive publications
        overlaps = []
        for i in range(len(keywords) - 1):
            set1 = set(keywords[i])
            set2 = set(keywords[i + 1])
            
            if len(set1) == 0 and len(set2) == 0:
                overlap = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                overlap = 0.0
            else:
                overlap = len(set1.intersection(set2)) / len(set1.union(set2))
            
            overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _identify_co_occurrences(self, keywords_lists: List[List[str]]) -> Dict[str, Dict[str, int]]:
        """Identify keyword co-occurrences across publications."""
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for keywords in keywords_lists:
            for i, kw1 in enumerate(keywords):
                for kw2 in keywords[i+1:]:
                    co_occurrence[kw1][kw2] += 1
                    co_occurrence[kw2][kw1] += 1
        
        return {k: dict(v) for k, v in co_occurrence.items()}
    
    def _cluster_keywords(self, keywords: List[str], frequencies: Dict[str, int]) -> List[List[str]]:
        """Simple clustering of keywords based on co-occurrence."""
        # This is a simplified clustering approach
        # In production, you might want to use more sophisticated methods
        
        clusters = []
        used_keywords = set()
        
        # Domain-based clustering
        domain_clusters = {
            'machine_learning': ['neural', 'network', 'deep', 'learning', 'model', 'training', 'algorithm'],
            'ai_agents': ['agent', 'agents', 'multi-agent', 'autonomous', 'collaboration', 'workflow'],
            'nlp': ['language', 'text', 'nlp', 'tokenization', 'embedding', 'transformer'],
            'data_science': ['data', 'analysis', 'visualization', 'statistics', 'prediction'],
            'frameworks': ['pytorch', 'tensorflow', 'langchain', 'crewai', 'huggingface']
        }
        
        for cluster_name, cluster_keywords in domain_clusters.items():
            cluster = []
            for kw in keywords:
                if kw.lower() in cluster_keywords and kw not in used_keywords:
                    cluster.append(kw)
                    used_keywords.add(kw)
            
            if cluster:
                clusters.append(cluster)
        
        # Add remaining keywords as individual clusters or misc cluster
        remaining = [kw for kw in keywords if kw not in used_keywords]
        if remaining:
            if len(remaining) <= 5:
                clusters.append(remaining)
            else:
                # Group by frequency
                high_freq = [kw for kw in remaining if frequencies.get(kw, 0) >= 3]
                low_freq = [kw for kw in remaining if frequencies.get(kw, 0) < 3]
                
                if high_freq:
                    clusters.append(high_freq)
                if low_freq:
                    clusters.append(low_freq)
        
        return clusters
    
    def _identify_emerging_declining(self, 
                                   current_keywords: Dict[str, int],
                                   historical_data: List[Dict[str, int]]) -> Tuple[List[str], List[str]]:
        """Identify emerging and declining keywords based on historical data."""
        if not historical_data:
            return [], []
        
        # Calculate average historical frequency
        historical_avg = defaultdict(float)
        for hist_data in historical_data:
            for kw, freq in hist_data.items():
                historical_avg[kw] += freq / len(historical_data)
        
        emerging = []
        declining = []
        
        for kw, current_freq in current_keywords.items():
            hist_freq = historical_avg.get(kw, 0)
            
            # Keyword is emerging if current frequency is significantly higher
            if current_freq > hist_freq * 1.5 and current_freq >= 2:
                emerging.append(kw)
            
            # Keyword is declining if it was common but now rare
            elif hist_freq >= 3 and current_freq < hist_freq * 0.5:
                declining.append(kw)
        
        return emerging, declining
    
    def _generate_recommendations(self, analysis: TrendAnalysis) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Top trends recommendation
        if analysis.top_keywords:
            top_keyword = analysis.top_keywords[0]['keyword']
            recommendations.append(
                f"Focus on '{top_keyword}' as it appears most frequently across publications"
            )
        
        # Diversity recommendation
        if analysis.diversity_index < 0.3:
            recommendations.append(
                "Content shows low diversity - consider exploring broader range of topics"
            )
        elif analysis.diversity_index > 0.8:
            recommendations.append(
                "High topic diversity detected - consider focusing on core themes for better coherence"
            )
        
        # Emerging trends recommendation
        if analysis.emerging_keywords:
            recommendations.append(
                f"Emerging trends detected: {', '.join(analysis.emerging_keywords[:3])} - consider early adoption"
            )
        
        # Cluster recommendation
        if len(analysis.clusters) > 5:
            recommendations.append(
                "Multiple distinct topic clusters found - consider specialized content for each area"
            )
        
        # Trend strength recommendation
        if analysis.trend_score < 0.3:
            recommendations.append(
                "Low trend coherence detected - ensure consistent terminology and themes"
            )
        
        return recommendations
    
    def _create_visualizations(self, analysis: TrendAnalysis, output_dir: str = "visualizations") -> Dict[str, str]:
        """Create visualization plots and return file paths."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        viz_paths = {}
        
        try:
            # 1. Keyword frequency bar chart
            if analysis.top_keywords:
                keywords = [item['keyword'] for item in analysis.top_keywords[:10]]
                frequencies = [item['frequency'] for item in analysis.top_keywords[:10]]
                
                fig = px.bar(
                    x=frequencies, 
                    y=keywords, 
                    orientation='h',
                    title="Top Keywords by Frequency",
                    labels={'x': 'Frequency', 'y': 'Keywords'}
                )
                fig.update_layout(height=500)
                
                path = os.path.join(output_dir, "keyword_frequency.html")
                fig.write_html(path)
                viz_paths['keyword_frequency'] = path
            
            # 2. Keyword distribution pie chart
            if len(analysis.top_keywords) >= 5:
                keywords = [item['keyword'] for item in analysis.top_keywords[:8]]
                frequencies = [item['frequency'] for item in analysis.top_keywords[:8]]
                
                # Add "Others" category for remaining keywords
                total_freq = sum(analysis.keyword_frequencies.values())
                shown_freq = sum(frequencies)
                if total_freq > shown_freq:
                    keywords.append("Others")
                    frequencies.append(total_freq - shown_freq)
                
                fig = px.pie(
                    values=frequencies,
                    names=keywords,
                    title="Keyword Distribution"
                )
                
                path = os.path.join(output_dir, "keyword_distribution.html")
                fig.write_html(path)
                viz_paths['keyword_distribution'] = path
            
            # 3. Cluster visualization (simple)
            if analysis.clusters:
                cluster_data = []
                for i, cluster in enumerate(analysis.clusters):
                    for keyword in cluster:
                        cluster_data.append({
                            'keyword': keyword,
                            'cluster': f'Cluster {i+1}',
                            'frequency': analysis.keyword_frequencies.get(keyword, 0)
                        })
                
                if cluster_data:
                    df = pd.DataFrame(cluster_data)
                    fig = px.scatter(
                        df,
                        x='cluster',
                        y='frequency',
                        hover_data=['keyword'],
                        title="Keyword Clusters",
                        size='frequency',
                        size_max=20
                    )
                    
                    path = os.path.join(output_dir, "keyword_clusters.html")
                    fig.write_html(path)
                    viz_paths['keyword_clusters'] = path
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return viz_paths
    
    def _run(self, keywords_data: str, **kwargs) -> str:
        """Analyze keyword data and return insights."""
        try:
            # Parse input data
            if isinstance(keywords_data, str):
                if keywords_data.startswith('['):
                    # JSON list of keyword lists
                    data = json.loads(keywords_data)
                else:
                    # Comma-separated keywords
                    data = [keywords_data.split(', ')]
            else:
                data = keywords_data
            
            if not data or all(not keywords for keywords in data):
                return "Error: No keywords provided for analysis"
            
            # Flatten all keywords and count frequencies
            all_keywords = []
            for keyword_list in data:
                if isinstance(keyword_list, list):
                    all_keywords.extend(keyword_list)
                elif isinstance(keyword_list, str):
                    all_keywords.extend(keyword_list.split(', '))
            
            keyword_frequencies = Counter(all_keywords)
            
            # Get top keywords
            top_keywords = [
                {
                    'keyword': kw,
                    'frequency': freq,
                    'percentage': round(freq / len(all_keywords) * 100, 2)
                }
                for kw, freq in keyword_frequencies.most_common(20)
            ]
            
            # Calculate metrics
            diversity_index = self._calculate_diversity_index(keyword_frequencies)
            trend_score = self._calculate_trend_score(data)
            co_occurrence = self._identify_co_occurrences(data)
            clusters = self._cluster_keywords(list(keyword_frequencies.keys()), keyword_frequencies)
            
            # Identify emerging/declining (mock for now without historical data)
            emerging_keywords = [kw for kw, freq in keyword_frequencies.items() if freq >= 3][:5]
            declining_keywords = []
            
            # Create analysis object
            analysis = TrendAnalysis(
                top_keywords=top_keywords,
                keyword_frequencies=dict(keyword_frequencies),
                co_occurrence_matrix=co_occurrence,
                trend_score=trend_score,
                diversity_index=diversity_index,
                emerging_keywords=emerging_keywords,
                declining_keywords=declining_keywords,
                clusters=clusters,
                summary_stats={
                    'total_keywords': len(all_keywords),
                    'unique_keywords': len(keyword_frequencies),
                    'most_common': top_keywords[0]['keyword'] if top_keywords else None,
                    'avg_frequency': np.mean(list(keyword_frequencies.values())),
                    'publications_analyzed': len(data)
                },
                recommendations=[]
            )
            
            # Generate recommendations
            analysis.recommendations = self._generate_recommendations(analysis)
            
            # For CrewAI compatibility, return a formatted string
            result = f"Analysis Complete: {analysis.summary_stats['unique_keywords']} unique keywords found"
            result += f"\\nTop keywords: {', '.join([kw['keyword'] for kw in top_keywords[:5]])}"
            result += f"\\nTrend score: {trend_score:.2f}, Diversity: {diversity_index:.2f}"
            
            # Store analysis for detailed retrieval
            self.last_analysis = analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return f"Error: Data analysis failed - {str(e)}"
    
    def get_detailed_analysis(self) -> Optional[TrendAnalysis]:
        """Get the last detailed analysis result."""
        return getattr(self, 'last_analysis', None)
    
    def create_report(self, 
                     publication_data: List[Dict[str, Any]], 
                     analysis: TrendAnalysis,
                     output_dir: str = "reports") -> ComprehensiveReport:
        """Create a comprehensive analysis report."""
        import os
        import uuid
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        viz_paths = self._create_visualizations(analysis, os.path.join(output_dir, "visualizations"))
        
        # Create executive summary
        exec_summary = self._generate_executive_summary(analysis)
        
        # Convert publication data to insights
        publication_insights = []
        for pub_data in publication_data:
            insight = PublicationInsight(
                url=pub_data.get('url', ''),
                title=pub_data.get('title'),
                keywords=pub_data.get('keywords', []),
                entities=pub_data.get('entities', []),
                topics=pub_data.get('topics', []),
                sentiment=pub_data.get('sentiment'),
                processing_time=pub_data.get('processing_time', 0.0),
                timestamp=datetime.now()
            )
            publication_insights.append(insight)
        
        report = ComprehensiveReport(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            publications_analyzed=len(publication_data),
            total_keywords=analysis.summary_stats.get('total_keywords', 0),
            unique_keywords=analysis.summary_stats.get('unique_keywords', 0),
            trend_analysis=analysis,
            publication_insights=publication_insights,
            visualizations=viz_paths,
            executive_summary=exec_summary
        )
        
        # Save report as JSON
        report_path = os.path.join(output_dir, f"report_{report.analysis_id}.json")
        with open(report_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = asdict(report)
            report_dict['timestamp'] = report.timestamp.isoformat()
            for insight in report_dict['publication_insights']:
                insight['timestamp'] = insight['timestamp'].isoformat() if insight['timestamp'] else None
            
            json.dump(report_dict, f, indent=2)
        
        return report
    
    def _generate_executive_summary(self, analysis: TrendAnalysis) -> str:
        """Generate an executive summary of the analysis."""
        summary_parts = []
        
        # Overview
        summary_parts.append(
            f"Analysis of {analysis.summary_stats.get('publications_analyzed', 0)} publications "
            f"revealed {analysis.summary_stats.get('unique_keywords', 0)} unique keywords."
        )
        
        # Top trends
        if analysis.top_keywords:
            top_3 = [kw['keyword'] for kw in analysis.top_keywords[:3]]
            summary_parts.append(
                f"The most prominent themes are: {', '.join(top_3)}."
            )
        
        # Trend strength
        if analysis.trend_score > 0.7:
            summary_parts.append("Strong thematic consistency observed across publications.")
        elif analysis.trend_score < 0.3:
            summary_parts.append("Publications show diverse, loosely related themes.")
        else:
            summary_parts.append("Moderate thematic alignment detected across publications.")
        
        # Diversity insight
        if analysis.diversity_index > 0.8:
            summary_parts.append("High topic diversity indicates broad coverage of subjects.")
        elif analysis.diversity_index < 0.3:
            summary_parts.append("Low topic diversity suggests focused, specialized content.")
        
        # Emerging trends
        if analysis.emerging_keywords:
            summary_parts.append(
                f"Emerging trends include: {', '.join(analysis.emerging_keywords[:3])}."
            )
        
        # Recommendations summary
        if analysis.recommendations:
            summary_parts.append(
                f"Key recommendation: {analysis.recommendations[0]}"
            )
        
        return " ".join(summary_parts)


# Legacy function for backward compatibility
def analyze_trends(keywords_lists: List[List[str]]) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    tool = DataAnalysisTool()
    result = tool._run(json.dumps(keywords_lists))
    
    # Return detailed analysis if available
    analysis = tool.get_detailed_analysis()
    if analysis:
        return asdict(analysis)
    
    return {"error": result}
