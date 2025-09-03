# Cross-Publication Insight Assistant - Enhanced

A sophisticated multi-agent system powered by CrewAI for analyzing research publications across multiple sources, identifying trends, and generating actionable insights. This enhanced version features a modern architecture with comprehensive testing, security, and user interface capabilities.

## ✅ Production-Ready Features

This project includes all essential production features:

- ✅ **Complete, Functional System Code**: Fully implemented multi-agent workflow
- ✅ **Clear Setup & Usage Instructions**: Comprehensive documentation below
- ✅ **Testing Suite**: 61+ tests covering unit, integration, and end-to-end scenarios
- ✅ **Security Guardrails**: Input validation, domain restrictions, content sanitization, rate limiting
- ✅ **Modern UI**: Streamlit-based web interface with real-time analysis
- ✅ **Environment Configuration**: `.env.example` file with all required variables
- ✅ **Logging & Monitoring**: Comprehensive logging system with configurable levels
- ✅ **No Hardcoded Secrets**: All sensitive data managed through environment variables

## 🚀 Features

### Core Capabilities
- **Multi-Agent Analysis**: CrewAI-powered agents for publication analysis, trend aggregation, and insight generation
- **Advanced Web Scraping**: Secure scraping with rate limiting, retry mechanisms, and domain validation
- **NLP Processing**: Multi-approach keyword extraction using NLTK, spaCy, and transformers
- **Statistical Analysis**: Comprehensive trend analysis with clustering and diversity calculations
- **Interactive UI**: Streamlit-based web interface with real-time analysis and visualizations

### Enhanced Features
- **Security First**: Input validation, domain restrictions, content sanitization
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Scalable Architecture**: Modular design with configurable components
- **Extensive Testing**: Unit, integration, and performance tests with 61+ test cases
- **Rich Visualizations**: Interactive charts and graphs using Plotly
- **Export Capabilities**: Multiple output formats (JSON, CSV, PDF reports)

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Agent Layer    │    │  Tool Layer     │
│                 │    │                 │    │                 │
│ • Streamlit App │◄──►│ • Publication   │◄──►│ • Web Scraper   │
│ • CLI Interface │    │   Analyzer      │    │ • Keyword       │
│ • API Endpoints │    │ • Trend         │    │   Extractor     │
│                 │    │   Aggregator    │    │ • Data Analyzer │
│                 │    │ • Insight       │    │                 │
│                 │    │   Generator     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ CrewAI & Flows  │
                    │                 │
                    │ • Crew          │
                    │   Orchestration │
                    │ • Flow          │
                    │   Management    │
                    │ • State         │
                    │   Tracking      │
                    └─────────────────┘
```

### Component Structure
```
src/
├── agents/              # CrewAI agents for specialized tasks
│   ├── publication_analyzer.py    # Content analysis agent
│   ├── trend_aggregator.py       # Trend identification agent
│   └── insight_generator.py      # Strategic insights agent
├── crews/              # CrewAI crew configurations
│   └── publication_crew.py       # Main crew orchestration
├── flows/              # Advanced workflow management
│   └── publication_flow.py       # Flow-based processing
├── tools/              # Enhanced analysis tools
│   ├── web_scraper.py            # Secure web scraping
│   ├── keyword_extractor.py      # NLP-based extraction
│   └── data_analyzer.py          # Statistical analysis
├── config.py           # Configuration management
└── main.py            # Application entry point
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- OpenAI API key
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/AmmarAhmedl200961/cross-publication-insight-assistant-enhanced.git
cd cross-publication-insight-assistant-enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration (especially OPENAI_API_KEY)

# Run tests to verify installation
python -m pytest tests/ -v

# Launch the application
python src/main.py --help
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install

# Run comprehensive tests
python -m pytest tests/ -v --cov=src --cov-report=html
```

## 📖 Usage

### Command Line Interface
```bash
# Analyze publications using CrewAI crew
python src/main.py analyze --method crew \
  --publications "https://arxiv.org/abs/2301.00001" \
  --include-entities --include-topics \
  --output results.json

# Analyze from configuration file
python src/main.py analyze --config analysis_config.json

# Run with flow orchestration
python src/main.py analyze --method flow \
  --publications publication_list.json \
  --generate-visualizations
```

### Web Interface
```bash
# Launch Streamlit app
streamlit run ui/streamlit_app.py

# Or use the integrated launcher
python src/main.py ui
```

### Python API
```python
from src.main import CrossPublicationInsightAssistant

# Initialize assistant
assistant = CrossPublicationInsightAssistant()

# Analyze publications
publications = [
    {"url": "https://arxiv.org/abs/2301.00001"},
    {"url": "https://arxiv.org/abs/2301.00002"}
]

result = assistant.analyze_publications(
    publications=publications,
    method="crew",
    include_entities=True,
    include_topics=True,
    generate_visualizations=True
)

# Save results
assistant.save_results(result, "analysis_results.json")
```

## 🔧 Configuration

### Environment Variables
Key configuration options in `.env`:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-4o-mini
WEB_SCRAPING_RATE_LIMIT=1.0
SECURITY_ALLOWED_DOMAINS=arxiv.org,scholar.google.com
UI_PORT=8501
LOG_LEVEL=INFO
```

See `.env.example` for complete configuration options.

### Advanced Configuration
The system supports extensive configuration through the `config/settings.py` file:

- **LLM Settings**: Model selection, temperature, token limits
- **Security Settings**: Domain validation, content filtering, rate limiting
- **Analysis Settings**: Keyword thresholds, confidence levels, batch sizes
- **UI Settings**: Themes, visualization options, export formats

## 🧪 Testing

### Comprehensive Test Suite
The project maintains extensive test coverage with **61+ test cases**:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_agents.py -v      # Agent tests (4 tests)
python -m pytest tests/test_tools.py -v      # Tool tests (9 tests)
python -m pytest tests/test_integration.py -v # Integration tests (8 tests)
python -m pytest tests/test_comprehensive.py -v # Comprehensive tests (40 tests)
```

### Test Categories
- **Unit Tests**: Individual component testing (agents, tools, utilities)
- **Integration Tests**: End-to-end workflow testing with real data flows
- **Security Tests**: Input validation, XSS protection, injection prevention
- **Performance Tests**: Memory usage, processing time, concurrent stability
- **Reliability Tests**: Error handling, timeout scenarios, malformed data
- **Mock Tests**: External service simulation for consistent testing

### Test Results Summary
```
61 tests collected covering:
- Web scraping security and reliability
- Keyword extraction with special characters
- Data analysis with large datasets  
- Agent workflow orchestration
- Flow-based processing
- Error recovery and resilience
- Memory usage stability
- Configuration variations
```

## 🔒 Security & Guardrails

### Input Validation & Sanitization
- **URL Validation**: Comprehensive URL format and domain checking
- **Content Sanitization**: HTML/script tag removal and content cleaning
- **Size Limits**: Configurable maximum content size (default: 10MB)
- **Domain Restrictions**: Whitelist-based domain access control

### Rate Limiting & Abuse Prevention
- **Request Rate Limiting**: Configurable requests per minute with burst protection
- **Retry Logic**: Exponential backoff for failed requests
- **Timeout Protection**: Configurable timeouts for all external calls
- **Resource Limits**: Memory and processing time constraints

### Security Features Implemented
```python
# Domain validation example
SECURITY_ALLOWED_DOMAINS=arxiv.org,scholar.google.com,researchgate.net

# Content size limits
SECURITY_MAX_CONTENT_SIZE=10485760  # 10MB

# Rate limiting
WEB_SCRAPING_RATE_LIMIT=1.0  # 1 request per second
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Protection Against
- ✅ **SQL Injection**: Parameterized queries and input validation
- ✅ **XSS Attacks**: Content sanitization and HTML parsing
- ✅ **SSRF**: Domain whitelisting and URL validation  
- ✅ **DoS**: Rate limiting and resource constraints
- ✅ **Data Exposure**: No hardcoded secrets, environment-based config

## �️ User Interface

### Streamlit Web Application
A modern, interactive web interface for publication analysis:

```bash
# Launch Streamlit app
streamlit run ui/streamlit_app.py

# Access at: http://localhost:8501
```

### UI Features
- **📊 Interactive Dashboard**: Real-time analysis progress and results
- **📁 Multiple Input Methods**: Manual URLs, file upload, or demo data
- **⚙️ Configuration Panel**: Advanced analysis options and settings
- **📈 Live Visualizations**: Dynamic charts and graphs using Plotly
- **💾 Export Options**: Download results in JSON, CSV, or summary formats
- **🔄 Real-time Updates**: Progress bars and status indicators

### UI Components
- **Analysis Tab**: Input methods and execution controls
- **Results Tab**: Comprehensive analysis results display
- **Export Tab**: Download and sharing capabilities
- **Configuration Sidebar**: Method selection and advanced options

### Supported Workflows
- **Crew Method**: Sequential agent collaboration
- **Flow Method**: Advanced orchestration with parallel processing
- **Batch Processing**: Multiple publications simultaneously
- **Real-time Monitoring**: Live progress tracking and error reporting

## 📄 Environment Configuration

### Required Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
# Copy the example file
cp .env.example .env

# Edit with your settings
nano .env
```

### Key Configuration Options
```bash
# --- Required ---
OPENAI_API_KEY=your_openai_api_key_here

# --- Core Settings ---
OPENAI_MODEL=gpt-4o-mini                    # LLM model selection
CREWAI_VERBOSE=true                         # Agent workflow verbosity
WEB_SCRAPING_RATE_LIMIT=1.0                # Requests per second
SECURITY_ALLOWED_DOMAINS=arxiv.org,scholar.google.com  # Allowed domains

# --- Optional Advanced Settings ---
LOG_LEVEL=INFO                              # Logging verbosity
CACHE_ENABLED=true                          # Enable request caching
UI_PORT=8501                                # Streamlit port
ANALYSIS_BATCH_SIZE=10                      # Batch processing size
```

### Security Configuration
```bash
# Domain restrictions (comma-separated)
SECURITY_ALLOWED_DOMAINS=arxiv.org,scholar.google.com,researchgate.net

# Content size limits
SECURITY_MAX_CONTENT_SIZE=10485760          # 10MB limit

# SSL validation
SECURITY_VALIDATE_SSL=true                  # Verify SSL certificates
```

### No Hardcoded Secrets
✅ All sensitive information is managed through environment variables:
- API keys loaded from `.env` file
- Configuration validation prevents hardcoded secrets
- Test mode supports mock credentials
- Graceful fallbacks for missing optional settings

## 📊 Logging & Monitoring

### Comprehensive Logging System
Built-in logging and monitoring capabilities:

```bash
# Log files location
logs/
├── app.log              # Main application logs
├── error.log            # Error-specific logs
└── performance.log      # Performance metrics
```

### Logging Features
- **📝 Structured Logging**: JSON-formatted logs with metadata
- **🔄 Log Rotation**: Automatic file rotation (10MB max, 5 backups)
- **📊 Multiple Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **⏱️ Performance Tracking**: Processing times and resource usage
- **🔍 Request Tracing**: Complete request/response logging

### Configuration Options
```bash
# Logging configuration
LOG_LEVEL=INFO                              # Minimum log level
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=logs/app.log                       # Log file location
LOG_MAX_SIZE=10MB                           # Max file size before rotation
LOG_BACKUP_COUNT=5                          # Number of backup files
```

### Monitoring Capabilities
- **📈 Performance Metrics**: Processing time, memory usage, request counts
- **⚠️ Error Tracking**: Detailed error logs with stack traces
- **🔄 Health Checks**: System status and component availability
- **📊 Usage Analytics**: Request patterns and user behavior (anonymized)

### Example Log Output
```
2025-09-03 19:30:40,854 - src.main - INFO - Initializing Cross-Publication Insight Assistant...
2025-09-03 19:30:41,166 - src.crews.publication_crew - INFO - Publication Insight Crew initialized successfully
2025-09-03 19:30:41,377 - src.main - INFO - All components initialized successfully
```

## 📊 Performance

### Benchmarks
- **Publication Analysis**: ~2-5 seconds per publication
- **Trend Analysis**: ~1-3 seconds for 10 publications
- **Memory Usage**: <100MB baseline, scales linearly
- **Throughput**: 10-50 publications/minute (depending on content)

### Optimization Features
- Async processing where possible
- Intelligent caching
- Batch processing
- Resource pool management

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Maintain test coverage above 90%
- Use type hints where appropriate

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### Documentation
- [Technical Documentation](docs/TECHNICAL.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

### Getting Help
- 📧 Email: support@example.com
<<<<<<< HEAD
- 💬 Discord: <a href="https://discord.com/users/515117034862804992">marto90123</a>
=======
- 💬 Discord: [marto90123](https://discord.com/users/515117034862804992)
>>>>>>> 89c15db (Update README with comprehensive feature documentation)
- 🐛 Issues: [GitHub Issues](https://github.com/AmmarAhmedl200961/cross-publication-insight-assistant-enhanced/issues)

### FAQ

**Q: What's the difference between crew and flow methods?**
A: Crew method uses sequential agent collaboration, while flow method provides more complex workflow orchestration with parallel processing and state management.

**Q: How do I add custom analysis tools?**
A: Extend the base tool classes in `src/tools/` and register them with your agents or crews.

**Q: Can I use other LLM providers besides OpenAI?**
A: Yes, the system supports any LangChain-compatible LLM provider. Update the configuration accordingly.

**Q: How do I handle large-scale analysis?**
A: Use batch processing, enable caching, and consider the flow method for parallel processing of large publication sets.

## 🚀 Roadmap

### Upcoming Features
- [ ] Multi-language publication support
- [ ] Advanced citation analysis
- [ ] Real-time collaboration features
- [ ] Enhanced visualization options
- [ ] Mobile-responsive UI
- [ ] API rate limiting dashboard
- [ ] Custom agent training capabilities

### Long-term Goals
- [ ] Federated learning for collaborative insights
- [ ] Integration with academic databases
- [ ] Automated report generation
- [ ] Machine learning model optimization
- [ ] Enterprise deployment options

---

**Built with ❤️ using CrewAI, LangChain, and Streamlit**
