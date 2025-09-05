``` mermaid
flowchart LR
  subgraph UI_Layer [UI Layer]
    StreamlitApp["Streamlit App"]
    CLIInterface["CLI Interface"]
    APIEndpoints["API Endpoints"]
  end
  subgraph Agent_Layer [Agent Layer]
    PublicationAnalyzer["Publication Analyzer"]
    TrendAggregator["Trend Aggregator"]
    InsightGenerator["Insight Generator"]
  end
  subgraph Tool_Layer [Tool Layer]
    WebScraper["Web Scraper"]
    KeywordExtractor["Keyword Extractor"]
    DataAnalyzer["Data Analyzer"]
  end
  subgraph CrewAI_Flows [CrewAI & Flows]
    CrewOrchestration["Crew Orchestration"]
    FlowManagement["Flow Management"]
    StateTracking["State Tracking"]
  end
  StreamlitApp -- interacts --> PublicationAnalyzer
  CLIInterface -- interacts --> PublicationAnalyzer
  APIEndpoints -- interacts --> PublicationAnalyzer
  PublicationAnalyzer -- uses --> WebScraper
  PublicationAnalyzer -- uses --> KeywordExtractor
  PublicationAnalyzer -- uses --> DataAnalyzer
  TrendAggregator -- receives --> PublicationAnalyzer
  InsightGenerator -- receives --> TrendAggregator
  CrewOrchestration -- manages --> PublicationAnalyzer
  CrewOrchestration -- manages --> TrendAggregator
  CrewOrchestration -- manages --> InsightGenerator
  FlowManagement -- coordinates --> CrewOrchestration
  StateTracking -- monitors --> CrewOrchestration
```