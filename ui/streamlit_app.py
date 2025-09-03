"""
Streamlit UI for the Cross-Publication Insight Assistant.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import CrossPublicationInsightAssistant
from config.settings import settings


# Page configuration
st.set_page_config(
    page_title="Cross-Publication Insight Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_assistant():
    """Initialize the assistant (cached to avoid re-initialization)."""
    try:
        return CrossPublicationInsightAssistant()
    except Exception as e:
        st.error(f"Failed to initialize assistant: {e}")
        return None


def create_keyword_chart(keywords_data: List[Dict[str, Any]]) -> go.Figure:
    """Create a keyword frequency chart."""
    if not keywords_data:
        return go.Figure().add_annotation(
            text="No keyword data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(keywords_data[:10])  # Top 10 keywords
    
    fig = px.bar(
        df, 
        x='frequency', 
        y='keyword',
        orientation='h',
        title="Top Keywords by Frequency",
        labels={'frequency': 'Frequency', 'keyword': 'Keyword'},
        color='frequency',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_topic_distribution_chart(topics_data: Dict[str, int]) -> go.Figure:
    """Create a topic distribution pie chart."""
    if not topics_data:
        return go.Figure().add_annotation(
            text="No topic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    labels = list(topics_data.keys())
    values = list(topics_data.values())
    
    fig = px.pie(
        values=values,
        names=labels,
        title="Topic Distribution"
    )
    
    fig.update_layout(height=400)
    return fig


def create_trend_timeline_chart(publications_data: List[Dict[str, Any]]) -> go.Figure:
    """Create a trend timeline chart."""
    if not publications_data:
        return go.Figure().add_annotation(
            text="No publication data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Extract timestamps and keyword counts
    timeline_data = []
    for i, pub in enumerate(publications_data):
        if pub.get('success'):
            timeline_data.append({
                'publication': f"Pub {i+1}",
                'keywords_count': len(pub.get('keywords', [])),
                'url': pub.get('url', ''),
                'processing_time': pub.get('processing_time', 0)
            })
    
    if not timeline_data:
        return go.Figure().add_annotation(
            text="No successful publication data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.line(
        df,
        x='publication',
        y='keywords_count',
        title="Keywords Extracted per Publication",
        markers=True,
        hover_data=['processing_time']
    )
    
    fig.update_layout(height=400)
    return fig


def display_publication_results(results: Dict[str, Any]):
    """Display publication analysis results."""
    publications = results.get('publications', {})
    pub_results = publications.get('results', [])
    
    if not pub_results:
        st.warning("No publication results to display")
        return
    
    st.markdown('<div class="section-header">📚 Publication Analysis Results</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Publications",
            publications.get('analyzed', 0)
        )
    
    with col2:
        st.metric(
            "Successful",
            publications.get('successful', 0),
            delta=f"{publications.get('successful', 0) - publications.get('failed', 0)}"
        )
    
    with col3:
        st.metric(
            "Failed",
            publications.get('failed', 0)
        )
    
    with col4:
        avg_processing_time = sum(r.get('processing_time', 0) for r in pub_results) / len(pub_results)
        st.metric(
            "Avg Processing Time",
            f"{avg_processing_time:.2f}s"
        )
    
    # Publication details
    st.subheader("Publication Details")
    
    for i, pub in enumerate(pub_results):
        with st.expander(f"Publication {i+1}: {pub.get('url', 'Unknown URL')[:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**URL:** {pub.get('url', 'N/A')}")
                st.write(f"**Status:** {'✅ Success' if pub.get('success') else '❌ Failed'}")
                
                if pub.get('success'):
                    keywords = pub.get('keywords', [])
                    st.write(f"**Keywords ({len(keywords)}):** {', '.join(keywords[:10])}")
                    
                    entities = pub.get('entities', [])
                    if entities:
                        entity_texts = [e.get('text', '') for e in entities[:5]]
                        st.write(f"**Entities:** {', '.join(entity_texts)}")
                    
                    topics = pub.get('topics', [])
                    if topics:
                        st.write(f"**Topics:** {', '.join(topics)}")
                else:
                    st.error(f"Error: {pub.get('error', 'Unknown error')}")
            
            with col2:
                st.write(f"**Processing Time:** {pub.get('processing_time', 0):.2f}s")
                st.write(f"**Content Length:** {pub.get('content_length', 0):,} chars")


def display_trend_analysis(results: Dict[str, Any]):
    """Display trend analysis results."""
    trend_analysis = results.get('trend_analysis', {})
    
    if not trend_analysis or not trend_analysis.get('success'):
        st.warning("No trend analysis data available")
        return
    
    st.markdown('<div class="section-header">📈 Trend Analysis</div>', unsafe_allow_html=True)
    
    # Statistical summary
    stats = trend_analysis.get('statistical_analysis', {})
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Unique Keywords",
                stats.get('unique_keywords', 0)
            )
        
        with col2:
            st.metric(
                "Total Keywords",
                stats.get('total_keywords', 0)
            )
        
        with col3:
            st.metric(
                "Trend Score",
                f"{stats.get('trend_score', 0):.2f}"
            )
        
        with col4:
            st.metric(
                "Diversity Index",
                f"{stats.get('diversity_index', 0):.2f}"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Keyword frequency chart
            top_keywords = stats.get('top_keywords', [])
            if top_keywords:
                fig = create_keyword_chart(top_keywords)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cross patterns
            cross_patterns = trend_analysis.get('cross_patterns', {})
            topic_dist = cross_patterns.get('topic_distribution', {})
            if topic_dist:
                fig = create_topic_distribution_chart(topic_dist)
                st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced insights
    enhanced_insights = trend_analysis.get('enhanced_insights', {})
    if enhanced_insights:
        st.subheader("Enhanced Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dominant Themes:**")
            themes = enhanced_insights.get('dominant_themes', [])
            for theme in themes[:5]:
                st.write(f"• {theme}")
        
        with col2:
            st.write("**Emerging Patterns:**")
            patterns = enhanced_insights.get('emerging_patterns', [])
            for pattern in patterns[:5]:
                st.write(f"• {pattern}")
    
    # Trend predictions
    predictions = trend_analysis.get('trend_predictions', {})
    if predictions:
        st.subheader("Trend Predictions")
        
        rising_trends = predictions.get('rising_trends', [])
        if rising_trends:
            st.write("**Rising Trends:**")
            trend_df = pd.DataFrame(rising_trends[:10])
            st.dataframe(trend_df, use_container_width=True)


def display_insights(results: Dict[str, Any]):
    """Display generated insights."""
    insights = results.get('insights', {})
    
    if not insights or not insights.get('success'):
        st.warning("No insights data available")
        return
    
    st.markdown('<div class="section-header">💡 Generated Insights</div>', unsafe_allow_html=True)
    
    # Executive summary
    exec_summary = insights.get('executive_summary', {})
    if exec_summary:
        st.subheader("Executive Summary")
        
        overview = exec_summary.get('overview', '')
        if overview:
            st.markdown(f'<div class="info-box">{overview}</div>', unsafe_allow_html=True)
        
        key_findings = exec_summary.get('key_findings', [])
        if key_findings:
            st.write("**Key Findings:**")
            for finding in key_findings:
                st.write(f"• {finding}")
    
    # Recommendations
    recommendations = insights.get('recommendations', {})
    if recommendations:
        st.subheader("Recommendations")
        
        tab1, tab2, tab3 = st.tabs(["Immediate Actions", "Short-term Strategy", "Long-term Planning"])
        
        with tab1:
            immediate = recommendations.get('immediate_actions', [])
            for action in immediate:
                st.write(f"• {action}")
        
        with tab2:
            short_term = recommendations.get('short_term_strategy', [])
            for strategy in short_term:
                st.write(f"• {strategy}")
        
        with tab3:
            long_term = recommendations.get('long_term_planning', [])
            for plan in long_term:
                st.write(f"• {plan}")
    
    # Opportunities and risks
    opp_risks = insights.get('opportunities_risks', {})
    if opp_risks:
        st.subheader("Opportunities & Risks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Opportunities:**")
            market_ops = opp_risks.get('market_opportunities', [])
            tech_ops = opp_risks.get('technology_opportunities', [])
            for op in (market_ops + tech_ops)[:5]:
                st.write(f"• {op}")
        
        with col2:
            st.write("**⚠️ Risks:**")
            tech_risks = opp_risks.get('technical_risks', [])
            market_risks = opp_risks.get('market_risks', [])
            for risk in (tech_risks + market_risks)[:5]:
                st.write(f"• {risk}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📊 Cross-Publication Insight Assistant</div>', unsafe_allow_html=True)
    st.markdown("*AI-powered analysis of AI/ML publications to identify trends and insights*")
    
    # Initialize assistant
    assistant = initialize_assistant()
    if not assistant:
        st.error("Failed to initialize the assistant. Please check your configuration.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Analysis method
    method = st.sidebar.selectbox(
        "Analysis Method",
        ["crew", "flow"],
        help="Choose between CrewAI crew (sequential) or flow (advanced orchestration)"
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        include_entities = st.checkbox("Include Named Entities", value=True)
        include_sentiment = st.checkbox("Include Sentiment Analysis", value=False)
        include_topics = st.checkbox("Include Topic Classification", value=True)
        generate_viz = st.checkbox("Generate Visualizations", value=True)
        
        if method == "flow":
            parallel_processing = st.checkbox("Parallel Processing", value=True)
            max_concurrent = st.slider("Max Concurrent Publications", 1, 10, 3)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "📊 Results", "📋 Export"])
    
    with tab1:
        st.header("Publication Analysis")
        
        # Input methods
        input_method = st.radio(
            "Input Method",
            ["Manual URLs", "Upload File", "Demo Data"],
            horizontal=True
        )
        
        publications = []
        
        if input_method == "Manual URLs":
            st.subheader("Enter Publication URLs")
            
            # URL input
            urls_text = st.text_area(
                "URLs (one per line)",
                height=150,
                placeholder="https://blog.langchain.dev/langgraph-multi-agent-workflows/\\nhttps://arxiv.org/abs/2301.00001"
            )
            
            if urls_text:
                urls = [url.strip() for url in urls_text.split('\\n') if url.strip()]
                publications = [{"url": url} for url in urls]
                
                st.info(f"Found {len(publications)} URLs")
        
        elif input_method == "Upload File":
            st.subheader("Upload Publication File")
            
            uploaded_file = st.file_uploader(
                "Choose a JSON file",
                type=['json'],
                help="Upload a JSON file containing publication data"
            )
            
            if uploaded_file:
                try:
                    data = json.load(uploaded_file)
                    publications = data.get('publications', data) if isinstance(data, dict) else data
                    st.success(f"Loaded {len(publications)} publications")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        else:  # Demo Data
            st.subheader("Demo Publications")
            
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
            
            st.info("Using demo publications for analysis")
            
            # Display demo publications
            for i, pub in enumerate(publications):
                st.write(f"{i+1}. {pub['url']}")
        
        # Analysis button
        if publications:
            if st.button("🚀 Start Analysis", type="primary"):
                
                # Prepare configuration
                config = {
                    'include_entities': include_entities,
                    'include_sentiment': include_sentiment,
                    'include_topics': include_topics,
                    'generate_visualizations': generate_viz
                }
                
                if method == "flow":
                    config.update({
                        'parallel_processing': parallel_processing,
                        'max_concurrent_publications': max_concurrent
                    })
                
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing analysis...")
                    progress_bar.progress(10)
                    
                    # Run analysis
                    status_text.text("Analyzing publications...")
                    progress_bar.progress(30)
                    
                    start_time = time.time()
                    
                    results = assistant.analyze_publications(
                        publications=publications,
                        method=method,
                        **config
                    )
                    
                    progress_bar.progress(100)
                    processing_time = time.time() - start_time
                    
                    # Store results in session state
                    st.session_state['analysis_results'] = results
                    st.session_state['analysis_time'] = processing_time
                    
                    if results.get('success'):
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"✅ Analysis completed successfully in {processing_time:.2f} seconds!")
                        st.balloons()
                    else:
                        status_text.empty()
                        progress_bar.empty()
                        st.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    status_text.empty()
                    progress_bar.empty()
                    st.error(f"❌ Analysis failed with exception: {e}")
        else:
            st.info("👆 Please provide publications to analyze using one of the input methods above.")
    
    with tab2:
        st.header("Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            if results.get('success'):
                # Display results sections
                display_publication_results(results)
                display_trend_analysis(results)
                display_insights(results)
                
                # Timeline chart
                st.markdown('<div class="section-header">📈 Analysis Timeline</div>', unsafe_allow_html=True)
                pub_results = results.get('publications', {}).get('results', [])
                if pub_results:
                    fig = create_trend_timeline_chart(pub_results)
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
        
        else:
            st.info("No analysis results available. Please run an analysis first.")
    
    with tab3:
        st.header("Export Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Download Results")
                
                # JSON export
                if st.button("📄 Download JSON"):
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="Download JSON File",
                        data=json_data,
                        file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Summary export
                if st.button("📋 Download Summary"):
                    summary = create_text_summary(results)
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                st.subheader("Share Results")
                
                # Create shareable link (mock)
                if st.button("🔗 Generate Share Link"):
                    st.info("Share link functionality would be implemented in production")
                
                # Export to external systems (mock)
                if st.button("📤 Export to Dashboard"):
                    st.info("Dashboard export functionality would be implemented in production")
        
        else:
            st.info("No results to export. Please run an analysis first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [CrewAI](https://crewai.com/) • "
        "[Streamlit](https://streamlit.io/) • "
        "[OpenAI](https://openai.com/)"
    )


def create_text_summary(results: Dict[str, Any]) -> str:
    """Create a text summary of the results."""
    summary_lines = [
        "CROSS-PUBLICATION INSIGHT ASSISTANT - ANALYSIS SUMMARY",
        "=" * 60,
        ""
    ]
    
    # Basic info
    summary_lines.append(f"Analysis Method: {results.get('method', 'unknown').upper()}")
    summary_lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
    summary_lines.append("")
    
    # Publications
    publications = results.get('publications', {})
    summary_lines.extend([
        "PUBLICATIONS:",
        f"  Total: {publications.get('analyzed', 0)}",
        f"  Successful: {publications.get('successful', 0)}",
        f"  Failed: {publications.get('failed', 0)}",
        ""
    ])
    
    # Trends
    trend_analysis = results.get('trend_analysis', {})
    if trend_analysis and trend_analysis.get('success'):
        stats = trend_analysis.get('statistical_analysis', {})
        summary_lines.extend([
            "TREND ANALYSIS:",
            f"  Unique Keywords: {stats.get('unique_keywords', 0)}",
            f"  Trend Score: {stats.get('trend_score', 0):.2f}",
            f"  Diversity Index: {stats.get('diversity_index', 0):.2f}",
            ""
        ])
        
        top_keywords = stats.get('top_keywords', [])[:5]
        if top_keywords:
            summary_lines.append("TOP KEYWORDS:")
            for kw in top_keywords:
                summary_lines.append(f"  - {kw['keyword']}: {kw['frequency']}")
            summary_lines.append("")
    
    # Insights
    insights = results.get('insights', {})
    if insights and insights.get('success'):
        exec_summary = insights.get('executive_summary', {})
        if exec_summary.get('overview'):
            summary_lines.extend([
                "EXECUTIVE SUMMARY:",
                f"  {exec_summary['overview']}",
                ""
            ])
        
        recommendations = insights.get('recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        if immediate_actions:
            summary_lines.append("IMMEDIATE ACTIONS:")
            for action in immediate_actions[:3]:
                summary_lines.append(f"  - {action}")
            summary_lines.append("")
    
    summary_lines.append("=" * 60)
    
    return "\\n".join(summary_lines)


if __name__ == "__main__":
    main()
