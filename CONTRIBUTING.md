# Contributing to Cross-Publication Insight Assistant

Thank you for your interest in contributing to the Cross-Publication Insight Assistant! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive experience for all contributors. Please be respectful, constructive, and professional in all interactions.

## 🚀 Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/cross-publication-insight-assistant-enhanced.git
   cd cross-publication-insight-assistant-enhanced
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Verify Setup**
   ```bash
   python -m pytest tests/ -v
   ```

## 📋 Development Guidelines

### Code Style

We follow Python best practices and maintain consistency across the codebase:

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type hints for function parameters and return values
- **Docstrings**: Document all public functions, classes, and modules
- **Line Length**: Maximum 88 characters (Black formatter default)

#### Example Function
```python
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def analyze_publication(
    url: str, 
    include_entities: bool = True,
    max_keywords: int = 50
) -> Dict[str, Any]:
    """
    Analyze a single publication and extract insights.
    
    Args:
        url: The publication URL to analyze
        include_entities: Whether to extract named entities
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        Dictionary containing analysis results with keys:
        - success: Boolean indicating if analysis succeeded
        - keywords: List of extracted keywords
        - entities: List of named entities (if requested)
        - content_length: Length of analyzed content
        
    Raises:
        ValueError: If URL is invalid or inaccessible
        AnalysisError: If content analysis fails
    """
    logger.info(f"Starting analysis for URL: {url}")
    
    # Implementation here
    pass
```

### Testing Requirements

All contributions must include appropriate tests:

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Security Tests**: Validate input sanitization and security measures
- **Performance Tests**: Ensure no significant performance regressions

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from src.tools.web_scraper import WebScrapingTool

class TestWebScrapingTool:
    """Test suite for WebScrapingTool."""
    
    @pytest.fixture
    def tool(self):
        """Create a WebScrapingTool instance for testing."""
        return WebScrapingTool()
    
    def test_successful_scraping(self, tool):
        """Test successful content scraping."""
        with patch('requests.Session.get') as mock_get:
            # Setup mock
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b'<html><body>Test content</body></html>'
            mock_get.return_value = mock_response
            
            # Test
            result = tool._run(url="https://example.com")
            
            # Assertions
            assert result['success'] is True
            assert 'content' in result
            assert len(result['content']) > 0
    
    def test_security_validation(self, tool):
        """Test URL security validation."""
        # Test blocked domain
        result = tool._run(url="https://malicious-site.com")
        assert result['success'] is False
        assert 'security' in result['error'].lower()
```

### Documentation

- **README Updates**: Update README.md for new features
- **Docstrings**: Comprehensive docstrings for all public APIs
- **Type Hints**: Complete type annotations
- **Examples**: Include usage examples for new features

## 🔄 Development Workflow

### Branch Strategy

1. **Main Branch**: Stable, production-ready code
2. **Development Branch**: Integration branch for features
3. **Feature Branches**: Individual feature development
4. **Hotfix Branches**: Critical bug fixes

### Commit Guidelines

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

#### Examples
```
feat(agents): add sentiment analysis to publication analyzer

- Implement sentiment scoring using VADER
- Add confidence metrics for sentiment predictions
- Update tests for new functionality

Closes #123
```

```
fix(security): prevent XSS in content sanitization

- Escape HTML entities in user-provided content
- Add validation for URL parameters
- Update security tests

Fixes #456
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: add new feature"
   
   # Run tests
   python -m pytest tests/ -v
   python -m pytest tests/ --cov=src --cov-report=html
   ```

3. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **PR Requirements**
   - Clear description of changes
   - Reference related issues
   - All tests passing
   - Code coverage maintained
   - Documentation updated

### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Code coverage maintained

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🏗️ Architecture Guidelines

### Component Design

When adding new components, follow these principles:

1. **Single Responsibility**: Each class/function has one clear purpose
2. **Dependency Injection**: Use constructor injection for dependencies
3. **Interface Segregation**: Define clear interfaces for components
4. **Error Handling**: Comprehensive error handling and logging

### Agent Development

For new CrewAI agents:

```python
from crewai import Agent
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import logging

class CustomAgent:
    """Custom agent for specialized analysis tasks."""
    
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        """Initialize the custom agent."""
        self.llm = llm
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        self.agent = Agent(
            role="Custom Analyst",
            goal="Perform specialized analysis on publications",
            backstory="""You are an expert analyst specializing in...""",
            llm=self.llm,
            verbose=self.verbose,
            tools=[],  # Add relevant tools
            max_iter=5,
            memory=True
        )
    
    def analyze(self, content: str, **kwargs) -> Dict[str, Any]:
        """Perform custom analysis on content."""
        try:
            # Implementation
            pass
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e)}
```

### Tool Development

For new analysis tools:

```python
from crewai_tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field

class CustomToolInput(BaseModel):
    """Input schema for custom tool."""
    data: str = Field(..., description="Data to process")
    options: Dict[str, Any] = Field(default={}, description="Processing options")

class CustomTool(BaseTool):
    """Custom tool for specialized processing."""
    
    name: str = "custom_tool"
    description: str = "Performs custom processing on input data"
    args_schema: Type[BaseModel] = CustomToolInput
    
    def _run(self, data: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the custom tool."""
        try:
            # Validate inputs
            if not data:
                return {"success": False, "error": "No data provided"}
            
            # Process data
            result = self._process_data(data, options or {})
            
            return {
                "success": True,
                "result": result,
                "metadata": {
                    "processed_length": len(data),
                    "options_used": options
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_data(self, data: str, options: Dict[str, Any]) -> Any:
        """Internal data processing logic."""
        # Implementation
        pass
```

## 🧪 Testing Guidelines

### Test Organization

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_agents.py     # Agent-specific tests
│   ├── test_tools.py      # Tool-specific tests
│   └── test_utils.py      # Utility function tests
├── integration/           # Integration tests
│   ├── test_workflows.py  # End-to-end workflow tests
│   └── test_api.py        # API endpoint tests
├── performance/           # Performance tests
│   └── test_benchmarks.py # Performance benchmarks
└── fixtures/              # Test data and fixtures
    ├── sample_data.json
    └── mock_responses.py
```

### Test Categories

1. **Unit Tests**: Fast, isolated tests
2. **Integration Tests**: Component interaction tests
3. **Performance Tests**: Speed and memory usage tests
4. **Security Tests**: Input validation and safety tests

### Mock Guidelines

Use mocks for external dependencies:

```python
@patch('src.tools.web_scraper.requests.Session.get')
def test_web_scraping(mock_get):
    """Test web scraping with mocked HTTP requests."""
    # Setup mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'<html>Content</html>'
    mock_get.return_value = mock_response
    
    # Test implementation
    pass
```

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Environment Information**
   - Python version
   - Package versions
   - Operating system

2. **Reproduction Steps**
   - Minimal code to reproduce
   - Expected vs actual behavior
   - Error messages/logs

3. **Context**
   - When did this start happening?
   - Does it happen consistently?
   - Any recent changes?

### Feature Requests

For new features, provide:

1. **Use Case**: Why is this needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Implementation Ideas**: Technical suggestions

## 🔧 Development Tools

### Recommended Tools

- **IDE**: VS Code with Python extension
- **Formatter**: Black for code formatting
- **Linter**: Pylint or Flake8 for code quality
- **Type Checker**: MyPy for type checking
- **Testing**: Pytest for testing framework

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
```

## 📚 Resources

### Documentation
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Learning Materials
- Python Best Practices
- Test-Driven Development
- Software Architecture Patterns
- Multi-Agent Systems

## 🤔 Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search GitHub issues
3. Ask in discussions
4. Contact maintainers

Thank you for contributing to Cross-Publication Insight Assistant! 🎉
