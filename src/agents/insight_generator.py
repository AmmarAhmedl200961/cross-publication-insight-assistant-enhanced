"""
Enhanced insight generator agent using CrewAI framework.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from crewai import Agent
from langchain_openai import ChatOpenAI

from config.settings import get_llm_config


logger = logging.getLogger(__name__)


class InsightGeneratorAgent:
    """CrewAI agent for generating comprehensive insights and reports from trend analysis."""
    
    def __init__(self):
        self.llm_config = get_llm_config()
        
        # Create the LLM instance
        self.llm = ChatOpenAI(
            model=self.llm_config['model'],
            temperature=self.llm_config['temperature'] + 0.1,  # Slightly more creative
            max_tokens=self.llm_config['max_tokens'],
            api_key=self.llm_config['api_key']
        )
        
        # Create the CrewAI agent
        self.agent = Agent(
            role='AI/ML Research Insights Strategist',
            goal='Transform trend analysis data into actionable insights, strategic recommendations, and comprehensive reports for AI/ML professionals',
            backstory="""You are a senior AI/ML research strategist and thought leader with deep 
            expertise in translating complex technical analysis into actionable business and research 
            insights. Your background includes consulting for major tech companies, advising research 
            institutions, and publishing influential reports on AI/ML trends. You excel at identifying 
            the strategic implications of technical developments and can communicate complex insights 
            to both technical and non-technical audiences. Your reports are known for their clarity, 
            depth, and actionable recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            max_iter=3,
            memory=True
        )
    
    def generate_insights(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive insights from trend analysis data.
        
        Args:
            trend_analysis: Output from TrendAggregatorAgent
            
        Returns:
            Dictionary containing generated insights and recommendations
        """
        start_time = datetime.now()
        
        try:
            logger.info("Generating comprehensive insights from trend analysis")
            
            if not trend_analysis.get('success', False):
                return {
                    'success': False,
                    'error': 'Invalid trend analysis data provided',
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }
            
            # Step 1: Generate executive summary
            executive_summary = self._generate_executive_summary(trend_analysis)
            
            # Step 2: Create strategic insights
            strategic_insights = self._generate_strategic_insights(trend_analysis)
            
            # Step 3: Develop actionable recommendations
            recommendations = self._generate_recommendations(trend_analysis)
            
            # Step 4: Identify opportunities and risks
            opportunities_risks = self._identify_opportunities_risks(trend_analysis)
            
            # Step 5: Create implementation roadmap
            roadmap = self._create_implementation_roadmap(trend_analysis, recommendations)
            
            # Step 6: Generate different audience-specific reports
            audience_reports = self._generate_audience_reports(trend_analysis)
            
            # Step 7: Create metrics and KPIs
            metrics = self._define_success_metrics(trend_analysis)
            
            # Step 8: Compile comprehensive results
            result = {
                'success': True,
                'executive_summary': executive_summary,
                'strategic_insights': strategic_insights,
                'recommendations': recommendations,
                'opportunities_risks': opportunities_risks,
                'implementation_roadmap': roadmap,
                'audience_reports': audience_reports,
                'success_metrics': metrics,
                'metadata': {
                    'publications_analyzed': trend_analysis.get('publications_analyzed', 0),
                    'unique_keywords': trend_analysis.get('unique_keywords', 0),
                    'analysis_depth': 'comprehensive',
                    'confidence_level': self._calculate_confidence_level(trend_analysis)
                },
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully generated insights in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'success': False,
                'error': f"Insight generation failed: {str(e)}",
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _generate_executive_summary(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary using CrewAI agent."""
        try:
            # Prepare summary of key data points
            stats = trend_analysis.get('statistical_analysis', {})
            cross_patterns = trend_analysis.get('cross_patterns', {})
            predictions = trend_analysis.get('trend_predictions', {})
            
            summary_prompt = f"""
            Create an executive summary for AI/ML technology trends based on this analysis:
            
            ANALYSIS OVERVIEW:
            - Publications analyzed: {trend_analysis.get('publications_analyzed', 0)}
            - Unique keywords identified: {trend_analysis.get('unique_keywords', 0)}
            - Trend coherence score: {stats.get('trend_score', 'N/A')}
            - Topic diversity: {stats.get('diversity_index', 'N/A')}
            
            TOP EMERGING TRENDS:
            {json.dumps(predictions.get('rising_trends', [])[:5], indent=2)}
            
            KEY TECHNOLOGY PATTERNS:
            {json.dumps(cross_patterns.get('technology_patterns', {}), indent=2)}
            
            Please provide:
            1. 2-3 sentence high-level summary
            2. Top 3 most significant findings
            3. Primary strategic implications
            4. Critical success factors for adoption
            5. Timeline expectations for trend maturation
            
            Keep it concise but impactful for C-level executives.
            """
            
            response = self.agent.llm.predict(summary_prompt)
            
            return {
                'overview': self._extract_overview(response),
                'key_findings': self._extract_key_findings(response),
                'strategic_implications': self._extract_implications(response),
                'success_factors': self._extract_success_factors(response),
                'timeline': self._extract_timeline(response),
                'full_summary': response
            }
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return {'error': f"Summary generation failed: {str(e)}"}
    
    def _generate_strategic_insights(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic insights using CrewAI agent."""
        try:
            enhanced_insights = trend_analysis.get('enhanced_insights', {})
            
            insights_prompt = f"""
            Provide strategic insights based on this AI/ML trend analysis:
            
            DOMINANT THEMES:
            {json.dumps(enhanced_insights.get('dominant_themes', []), indent=2)}
            
            EMERGING PATTERNS:
            {json.dumps(enhanced_insights.get('emerging_patterns', []), indent=2)}
            
            RESEARCH DIRECTIONS:
            {json.dumps(enhanced_insights.get('research_directions', []), indent=2)}
            
            CONVERGENCE PATTERNS:
            {json.dumps(enhanced_insights.get('convergence_patterns', []), indent=2)}
            
            Analyze and provide insights on:
            1. Market positioning opportunities
            2. Competitive advantages from early adoption
            3. Resource allocation priorities
            4. Partnership and collaboration opportunities
            5. Risk mitigation strategies
            6. Innovation investment areas
            
            Focus on actionable strategic insights.
            """
            
            response = self.agent.llm.predict(insights_prompt)
            
            return {
                'market_positioning': self._extract_market_insights(response),
                'competitive_advantages': self._extract_competitive_insights(response),
                'resource_allocation': self._extract_resource_insights(response),
                'partnerships': self._extract_partnership_insights(response),
                'risk_mitigation': self._extract_risk_insights(response),
                'investment_areas': self._extract_investment_insights(response),
                'full_insights': response
            }
            
        except Exception as e:
            logger.error(f"Strategic insights generation failed: {e}")
            return {'error': f"Strategic insights failed: {str(e)}"}
    
    def _generate_recommendations(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        try:
            predictions = trend_analysis.get('trend_predictions', {})
            
            recommendations_prompt = f"""
            Based on the trend analysis, provide specific recommendations:
            
            RISING TRENDS:
            {json.dumps(predictions.get('rising_trends', [])[:5], indent=2)}
            
            BREAKTHROUGH INDICATORS:
            {json.dumps(predictions.get('breakthrough_indicators', [])[:5], indent=2)}
            
            Provide specific recommendations for:
            1. Immediate actions (next 3 months)
            2. Short-term strategy (3-12 months)
            3. Long-term planning (1-3 years)
            4. Technology adoption roadmap
            5. Skill development priorities
            6. Research and development focus
            
            Each recommendation should be specific, measurable, and actionable.
            """
            
            response = self.agent.llm.predict(recommendations_prompt)
            
            return {
                'immediate_actions': self._extract_immediate_actions(response),
                'short_term_strategy': self._extract_short_term(response),
                'long_term_planning': self._extract_long_term(response),
                'technology_roadmap': self._extract_tech_roadmap(response),
                'skill_development': self._extract_skills(response),
                'rd_focus': self._extract_rd_focus(response),
                'full_recommendations': response
            }
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return {'error': f"Recommendations failed: {str(e)}"}
    
    def _identify_opportunities_risks(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify opportunities and risks."""
        try:
            opportunities_prompt = f"""
            Based on the trend analysis data, identify key opportunities and risks:
            
            TREND DATA SUMMARY:
            - Publications analyzed: {trend_analysis.get('publications_analyzed', 0)}
            - Trend coherence: {trend_analysis.get('statistical_analysis', {}).get('trend_score', 'N/A')}
            - Emerging patterns: {len(trend_analysis.get('enhanced_insights', {}).get('emerging_patterns', []))}
            
            Identify:
            1. Market opportunities (new niches, underserved areas)
            2. Technology opportunities (emerging tech combinations)
            3. Business model opportunities (new ways to create value)
            4. Technical risks (potential pitfalls, limitations)
            5. Market risks (competition, timing, adoption barriers)
            6. Strategic risks (wrong investments, missed opportunities)
            
            Provide specific, actionable insights for each category.
            """
            
            response = self.agent.llm.predict(opportunities_prompt)
            
            return {
                'market_opportunities': self._extract_market_opportunities(response),
                'technology_opportunities': self._extract_tech_opportunities(response),
                'business_opportunities': self._extract_business_opportunities(response),
                'technical_risks': self._extract_technical_risks(response),
                'market_risks': self._extract_market_risks(response),
                'strategic_risks': self._extract_strategic_risks(response),
                'full_analysis': response
            }
            
        except Exception as e:
            logger.error(f"Opportunities/risks analysis failed: {e}")
            return {'error': f"Opportunities/risks analysis failed: {str(e)}"}
    
    def _create_implementation_roadmap(self, 
                                     trend_analysis: Dict[str, Any], 
                                     recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation roadmap."""
        try:
            roadmap_prompt = f"""
            Create a detailed implementation roadmap based on:
            
            RECOMMENDATIONS:
            - Immediate actions: {recommendations.get('immediate_actions', [])}
            - Short-term strategy: {recommendations.get('short_term_strategy', [])}
            - Long-term planning: {recommendations.get('long_term_planning', [])}
            
            Create a roadmap with:
            1. Phase 1 (Months 1-3): Foundation and quick wins
            2. Phase 2 (Months 4-12): Strategic implementation
            3. Phase 3 (Years 2-3): Advanced capabilities and optimization
            4. Success milestones for each phase
            5. Resource requirements and dependencies
            6. Risk mitigation checkpoints
            
            Make it specific and executable.
            """
            
            response = self.agent.llm.predict(roadmap_prompt)
            
            return {
                'phase_1': self._extract_phase_details(response, '1'),
                'phase_2': self._extract_phase_details(response, '2'),
                'phase_3': self._extract_phase_details(response, '3'),
                'milestones': self._extract_milestones(response),
                'resources': self._extract_resources(response),
                'dependencies': self._extract_dependencies(response),
                'full_roadmap': response
            }
            
        except Exception as e:
            logger.error(f"Roadmap creation failed: {e}")
            return {'error': f"Roadmap creation failed: {str(e)}"}
    
    def _generate_audience_reports(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reports for different audiences."""
        audiences = {
            'executives': 'Focus on business impact, ROI, and strategic positioning',
            'researchers': 'Focus on technical details, methodologies, and research directions',
            'developers': 'Focus on implementation, tools, and practical applications',
            'product_managers': 'Focus on features, user needs, and market opportunities'
        }
        
        reports = {}
        
        for audience, focus in audiences.items():
            try:
                audience_prompt = f"""
                Create a focused report for {audience.replace('_', ' ')}.
                
                {focus}
                
                TREND DATA: {json.dumps(trend_analysis.get('enhanced_insights', {}), indent=2)[:1000]}...
                
                Provide:
                1. Executive summary tailored to this audience
                2. Top 3 priorities for this audience
                3. Specific action items
                4. Success metrics relevant to their role
                5. Timeline and resource considerations
                
                Keep it concise and role-specific.
                """
                
                response = self.agent.llm.predict(audience_prompt)
                reports[audience] = {
                    'summary': response.split('\\n')[0] if response else '',
                    'priorities': self._extract_priorities(response),
                    'action_items': self._extract_action_items(response),
                    'metrics': self._extract_audience_metrics(response),
                    'timeline': self._extract_audience_timeline(response),
                    'full_report': response
                }
                
            except Exception as e:
                logger.error(f"Report generation for {audience} failed: {e}")
                reports[audience] = {'error': f"Report generation failed: {str(e)}"}
        
        return reports
    
    def _define_success_metrics(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Define success metrics and KPIs."""
        return {
            'adoption_metrics': [
                'Technology implementation rate',
                'Team skill development progress',
                'Project success rate using new technologies'
            ],
            'innovation_metrics': [
                'New product features enabled by trends',
                'Research papers/patents filed',
                'Time-to-market improvements'
            ],
            'business_metrics': [
                'Revenue impact from trend adoption',
                'Cost savings from efficiency gains',
                'Market share in emerging segments'
            ],
            'leading_indicators': [
                'Trend monitoring dashboard updates',
                'Expert network engagement',
                'Technology proof-of-concept completions'
            ]
        }
    
    def _calculate_confidence_level(self, trend_analysis: Dict[str, Any]) -> str:
        """Calculate confidence level of the analysis."""
        publications_count = trend_analysis.get('publications_analyzed', 0)
        unique_keywords = trend_analysis.get('unique_keywords', 0)
        trend_score = trend_analysis.get('statistical_analysis', {}).get('trend_score', 0)
        
        confidence_score = 0
        
        # Publication count factor
        if publications_count >= 10:
            confidence_score += 40
        elif publications_count >= 5:
            confidence_score += 25
        elif publications_count >= 2:
            confidence_score += 15
        
        # Keyword diversity factor
        if unique_keywords >= 50:
            confidence_score += 30
        elif unique_keywords >= 25:
            confidence_score += 20
        elif unique_keywords >= 10:
            confidence_score += 10
        
        # Trend coherence factor
        if trend_score >= 0.7:
            confidence_score += 30
        elif trend_score >= 0.5:
            confidence_score += 20
        elif trend_score >= 0.3:
            confidence_score += 10
        
        if confidence_score >= 80:
            return 'High'
        elif confidence_score >= 60:
            return 'Medium-High'
        elif confidence_score >= 40:
            return 'Medium'
        else:
            return 'Low-Medium'
    
    # Helper methods for extracting specific content from agent responses
    def _extract_overview(self, response: str) -> str:
        """Extract overview from response."""
        lines = response.split('\\n')
        for line in lines[:3]:  # Look in first few lines
            if len(line.strip()) > 50:  # Substantial content
                return line.strip()
        return response.split('.')[0] if response else "Overview not available"
    
    def _extract_key_findings(self, response: str) -> List[str]:
        """Extract key findings from response."""
        findings = []
        lines = response.split('\\n')
        for line in lines:
            if 'finding' in line.lower() or line.strip().startswith(('1.', '2.', '3.')):
                findings.append(line.strip())
        return findings[:3]
    
    def _extract_implications(self, response: str) -> List[str]:
        """Extract strategic implications."""
        implications = []
        lines = response.split('\\n')
        for line in lines:
            if 'implication' in line.lower() or 'strategic' in line.lower():
                implications.append(line.strip())
        return implications[:3]
    
    def _extract_success_factors(self, response: str) -> List[str]:
        """Extract success factors."""
        factors = []
        lines = response.split('\\n')
        for line in lines:
            if 'success' in line.lower() or 'factor' in line.lower():
                factors.append(line.strip())
        return factors[:3]
    
    def _extract_timeline(self, response: str) -> str:
        """Extract timeline information."""
        lines = response.split('\\n')
        for line in lines:
            if 'timeline' in line.lower() or 'month' in line.lower() or 'year' in line.lower():
                return line.strip()
        return "Timeline not specified"
    
    # Additional helper methods for other extraction functions...
    def _extract_market_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['market', 'positioning'])
    
    def _extract_competitive_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['competitive', 'advantage'])
    
    def _extract_resource_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['resource', 'allocation'])
    
    def _extract_partnership_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['partnership', 'collaboration'])
    
    def _extract_risk_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['risk', 'mitigation'])
    
    def _extract_investment_insights(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['investment', 'innovation'])
    
    def _extract_by_keyword(self, response: str, keywords: List[str]) -> List[str]:
        """Generic method to extract content by keywords."""
        results = []
        lines = response.split('\\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                results.append(line.strip())
        return results[:3]
    
    # Implement similar extraction methods for other content types...
    def _extract_immediate_actions(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['immediate', 'next 3'])
    
    def _extract_short_term(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['short-term', '3-12'])
    
    def _extract_long_term(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['long-term', '1-3'])
    
    def _extract_tech_roadmap(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['technology', 'roadmap'])
    
    def _extract_skills(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['skill', 'development'])
    
    def _extract_rd_focus(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['research', 'development'])
    
    def _extract_market_opportunities(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['market opportunity', 'market opportunities'])
    
    def _extract_tech_opportunities(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['technology opportunity', 'technology opportunities'])
    
    def _extract_business_opportunities(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['business opportunity', 'business opportunities'])
    
    def _extract_technical_risks(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['technical risk', 'technical risks'])
    
    def _extract_market_risks(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['market risk', 'market risks'])
    
    def _extract_strategic_risks(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['strategic risk', 'strategic risks'])
    
    def _extract_phase_details(self, response: str, phase: str) -> List[str]:
        return self._extract_by_keyword(response, [f'phase {phase}', f'months {phase}'])
    
    def _extract_milestones(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['milestone', 'milestones'])
    
    def _extract_resources(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['resource', 'resources'])
    
    def _extract_dependencies(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['dependency', 'dependencies'])
    
    def _extract_priorities(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['priority', 'priorities'])
    
    def _extract_action_items(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['action', 'item'])
    
    def _extract_audience_metrics(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['metric', 'metrics'])
    
    def _extract_audience_timeline(self, response: str) -> List[str]:
        return self._extract_by_keyword(response, ['timeline', 'schedule'])
    
    def get_agent(self) -> Agent:
        """Get the underlying CrewAI agent."""
        return self.agent
