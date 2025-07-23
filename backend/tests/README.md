# RAGFlow Backend Testing

This directory contains comprehensive tests for the RAGFlow backend application.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_api_integration.py        # API endpoint integration tests
├── test_citation_generator.py     # Citation generation unit tests
├── test_citation_validator.py     # Citation validation unit tests
├── test_database_integration.py   # Database integration tests
├── test_end_to_end.py             # End-to-end workflow tests
├── test_logging_config.py         # Logging system unit tests
├── test_monitoring.py             # Monitoring system unit tests
└── README.md                      # This file
```

## Test Categories

### Unit Tests
- **Citation Generator**: Tests for citation generation logic
- **Citation Validator**: Tests for citation quality validation
- **Logging Config**: Tests for structured logging functionality
- **Monitoring**: Tests for health checks and system monitoring

### Integration Tests
- **API Integration**: Tests for API endpoints and request/response handling
- **Database Integration**: Tests for database operations and data persistence

### End-to-End Tests
- **Complete Workflows**: Tests for full document processing workflows
- **Performance Scenarios**: Tests for large document and concurrent processing
- **Error Recovery**: Tests for failure handling and system resilience
- **Security Workflows**: Tests for secure processing and audit trails

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Basic Test Execution

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type e2e

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file tests/test_citation_generator.py

# Run specific test function
python run_tests.py --function test_generate_citations_basic
```

### Advanced Test Execution

```bash
# Run tests in parallel
python run_tests.py --parallel 4

# Run only failed tests from last run
python run_tests.py --failed

# Run with profiling
python run_tests.py --profile

# Verbose output
python run_tests.py --verbose
```

### Predefined Test Suites

```bash
# Run all test suites in sequence
python run_tests.py suites

# Quick tests for development (excludes slow tests)
python run_tests.py quick

# CI/CD optimized tests
python run_tests.py ci
```

### Direct Pytest Usage

```bash
# Run all tests
pytest

# Run with markers
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m "not slow"

# Run specific tests
pytest tests/test_citation_generator.py::TestCitationGenerator::test_generate_citations_basic

# Run with coverage
pytest --cov=app --cov-report=html

# Run with verbose output
pytest -v -s
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Tests that take more than 5 seconds
- `@pytest.mark.performance`: Performance-related tests
- `@pytest.mark.security`: Security-related tests
- `@pytest.mark.database`: Database tests
- `@pytest.mark.api`: API tests
- `@pytest.mark.monitoring`: Monitoring tests
- `@pytest.mark.citation`: Citation system tests

## Test Configuration

### Environment Variables

```bash
# Test environment
export TESTING=true
export DATABASE_URL=sqlite:///:memory:
export REDIS_URL=redis://localhost:6379/15
export LOG_LEVEL=DEBUG
```

### Pytest Configuration

Configuration is defined in `pytest.ini`:

- Test discovery patterns
- Coverage settings
- Async support
- Logging configuration
- Warning filters
- Timeout settings

## Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestMyComponent:
    """Test cases for MyComponent."""
    
    @pytest.fixture
    def my_fixture(self):
        """Create test fixture."""
        return SomeTestData()
    
    @pytest.mark.unit
    def test_basic_functionality(self, my_fixture):
        """Test basic functionality."""
        # Arrange
        component = MyComponent()
        
        # Act
        result = component.do_something(my_fixture)
        
        # Assert
        assert result.success == True
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        component = MyComponent()
        result = await component.do_async_something()
        assert result is not None
```

### Mocking Guidelines

```python
# Mock external services
@patch('app.services.external_service.client')
def test_with_external_service(mock_client):
    mock_client.get_data.return_value = {"test": "data"}
    # Test implementation

# Mock async services
@patch('app.services.async_service.process')
async def test_async_service(mock_process):
    mock_process.return_value = AsyncMock(return_value="result")
    # Test implementation
```

### Fixtures

Common fixtures are available in `conftest.py`:

- `test_db_session`: Database session for testing
- `mock_cache_manager`: Mocked cache manager
- `sample_document_data`: Sample document data
- `sample_chunk_data`: Sample chunk data
- `citation_generator_service`: Citation generator with mocked dependencies

## Coverage Requirements

- Minimum coverage: 80%
- Coverage reports generated in `htmlcov/` directory
- XML coverage report for CI/CD integration

## Performance Testing

Performance tests are marked with `@pytest.mark.performance`:

```python
@pytest.mark.performance
def test_large_document_processing():
    """Test processing performance with large documents."""
    # Performance test implementation
```

## Continuous Integration

For CI/CD environments:

```bash
# Run CI-optimized tests
python run_tests.py ci

# This runs:
# - All tests with coverage
# - JUnit XML output for CI integration
# - Fail if coverage below 80%
```

## Debugging Tests

```bash
# Run with debugging
pytest --pdb

# Run specific test with output
pytest -v -s tests/test_citation_generator.py::test_specific_function

# Run with logging
pytest --log-cli-level=DEBUG
```

## Test Data

Test data is managed through:

- Fixtures in `conftest.py`
- Sample data generators
- Temporary files and directories
- In-memory databases for isolation

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies
3. **Cleanup**: Use fixtures for setup/teardown
4. **Naming**: Use descriptive test names
5. **Documentation**: Document complex test scenarios
6. **Performance**: Mark slow tests appropriately
7. **Coverage**: Aim for high test coverage
8. **Async**: Use proper async test patterns

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the app directory
2. **Database Errors**: Check test database configuration
3. **Async Errors**: Use `@pytest.mark.asyncio` for async tests
4. **Mock Errors**: Verify mock paths and return values
5. **Timeout Errors**: Increase timeout for slow tests

### Debug Commands

```bash
# Check test discovery
pytest --collect-only

# Run with maximum verbosity
pytest -vvv

# Show local variables on failure
pytest --tb=long

# Run without capturing output
pytest -s
```

## Contributing

When adding new tests:

1. Follow the existing test structure
2. Add appropriate markers
3. Include docstrings
4. Mock external dependencies
5. Ensure tests are deterministic
6. Add to appropriate test category
7. Update this README if needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Python Mock](https://docs.python.org/3/library/unittest.mock.html)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)