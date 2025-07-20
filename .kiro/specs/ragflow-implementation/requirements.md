# Requirements Document

## Introduction

This document outlines the requirements for implementing a comprehensive RAG (Retrieval-Augmented Generation) system based on RAGFlow architecture. The system will provide advanced document processing, intelligent chunking, knowledge graph construction, hybrid search capabilities, and high-quality response generation with Russian and English language support.

The system aims to achieve enterprise-grade document understanding and search capabilities comparable to RAGFlow, while integrating with Yandex Cloud ML services and maintaining scalable FastAPI architecture.

## Requirements

### Requirement 1: Core FastAPI Infrastructure

**User Story:** As a system administrator, I want a robust FastAPI backend infrastructure, so that the system can handle concurrent requests reliably and scale horizontally.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL initialize FastAPI application with proper middleware configuration
2. WHEN a request is received THEN the system SHALL log request details including processing time and response status
3. WHEN database connections are needed THEN the system SHALL use connection pooling with automatic reconnection
4. WHEN environment variables are missing THEN the system SHALL fail gracefully with clear error messages
5. IF system health check is requested THEN the system SHALL return status of all critical components (database, vector store, Redis, LLM services)
6. WHEN concurrent requests exceed limits THEN the system SHALL implement proper rate limiting and queue management

### Requirement 2: Deep Document Understanding (DeepDoc)

**User Story:** As a user, I want the system to deeply understand document structure including tables, images, and layout, so that I can get accurate information from complex documents.

#### Acceptance Criteria

1. WHEN a PDF document is uploaded THEN the system SHALL render each page as high-resolution image (zoomin=3)
2. WHEN document layout analysis is performed THEN the system SHALL identify and classify blocks (title, text, table, figure, header, footer) with confidence > 0.5
3. WHEN OCR is performed THEN the system SHALL detect text regions and recognize text with confidence > 0.7
4. WHEN tables are detected THEN the system SHALL recognize table structure (rows, columns, headers, spanning cells) and convert to HTML/Markdown format
5. WHEN images are found THEN the system SHALL extract them and generate descriptions using vision LLM
6. IF ONNX models are unavailable THEN the system SHALL fallback to basic document parsing without failing
7. WHEN text blocks are processed THEN the system SHALL merge them into logical paragraphs using spatial analysis
8. WHEN processing fails THEN the system SHALL log detailed error information and continue with available data

### Requirement 3: Advanced Chunking Strategies

**User Story:** As a user, I want documents to be intelligently split into meaningful chunks, so that search results are contextually relevant and complete.

#### Acceptance Criteria

1. WHEN document type is detected THEN the system SHALL select appropriate chunking strategy (naive, hierarchical, QA, table)
2. WHEN naive chunking is used THEN the system SHALL merge text blocks until token limit (1000 tokens) with overlap (200 tokens)
3. WHEN hierarchical chunking is used THEN the system SHALL preserve document structure based on headings and sections
4. WHEN QA chunking is used THEN the system SHALL identify question-answer pairs and create dedicated chunks
5. WHEN table chunking is used THEN the system SHALL preserve table structure and create chunks with headers + row groups
6. WHEN RAPTOR is enabled THEN the system SHALL cluster chunks semantically and create hierarchical summaries
7. IF chunking strategy fails THEN the system SHALL fallback to naive chunking
8. WHEN chunks are created THEN each chunk SHALL contain metadata (position, type, strategy, confidence)

### Requirement 4: Content Enrichment

**User Story:** As a user, I want document chunks to be enriched with keywords, questions, and tags, so that search accuracy and relevance are improved.

#### Acceptance Criteria

1. WHEN a chunk is processed THEN the system SHALL extract 3-5 relevant keywords using LLM
2. WHEN a chunk is processed THEN the system SHALL generate 2-3 potential questions that the chunk could answer
3. WHEN content tagging is enabled THEN the system SHALL assign relevant tags from predefined taxonomy
4. WHEN enrichment fails THEN the system SHALL continue processing without enrichment data
5. WHEN enrichment is complete THEN the system SHALL store enrichment data with confidence scores
6. IF LLM service is unavailable THEN the system SHALL use fallback keyword extraction methods

### Requirement 5: Knowledge Graph Construction

**User Story:** As a user, I want the system to build a knowledge graph from documents, so that I can discover relationships and connections between entities across documents.

#### Acceptance Criteria

1. WHEN text is processed THEN the system SHALL extract named entities (persons, organizations, locations) with confidence > 0.8
2. WHEN entities are extracted THEN the system SHALL identify relationships between entities with confidence > 0.7
3. WHEN entities and relations are collected THEN the system SHALL build a graph using NetworkX
4. WHEN graph is built THEN the system SHALL detect communities using Leiden algorithm
5. WHEN communities are detected THEN the system SHALL generate summary reports for each community using LLM
6. WHEN duplicate entities are found THEN the system SHALL merge them using entity resolution
7. IF graph construction fails THEN the system SHALL continue without graph features
8. WHEN graph is updated THEN the system SHALL maintain consistency and update related components

### Requirement 6: Hybrid Search System with Elasticsearch

**User Story:** As a user, I want to search documents using both keyword and semantic search, so that I can find relevant information regardless of exact word matching.

#### Acceptance Criteria

1. WHEN search query is received THEN the system SHALL perform both sparse (Elasticsearch) and dense (vector) retrieval simultaneously
2. WHEN sparse retrieval is performed THEN the system SHALL use Elasticsearch with custom analyzers, synonym expansion, and TF-IDF term weighting
3. WHEN query preprocessing occurs THEN the system SHALL apply synonym expansion using WordNet and custom dictionaries
4. WHEN dense retrieval is performed THEN the system SHALL convert query to embedding and find similar chunks using cosine similarity in Infinity DB
5. WHEN both retrievals complete THEN the system SHALL fuse results using weighted scoring algorithms (RRF or weighted sum)
6. WHEN initial results are obtained THEN the system SHALL rerank top 20 results using specialized reranking model or LLM
7. WHEN knowledge graph is available THEN the system SHALL include related entities and community reports in search context
8. IF Elasticsearch is unavailable THEN the system SHALL fallback to PostgreSQL full-text search
9. IF vector store is unavailable THEN the system SHALL continue with sparse search only
10. WHEN search is complete THEN the system SHALL return top 5 most relevant results with confidence scores and fusion metadata

### Requirement 7: Agentic Response Generation

**User Story:** As a user, I want the system to provide thoughtful, well-reasoned answers with proper citations, so that I can trust the information and verify sources.

#### Acceptance Criteria

1. WHEN complex query is detected THEN the system SHALL use agentic reasoning with multi-step analysis
2. WHEN agentic mode is active THEN the system SHALL generate reasoning steps and perform iterative searches if needed
3. WHEN response is generated THEN the system SHALL include specific citations linking answer parts to source chunks
4. WHEN chat history is provided THEN the system SHALL consider context and maintain conversation coherence
5. WHEN knowledge graph context is available THEN the system SHALL incorporate related entities and relationships
6. WHEN answer is generated THEN the system SHALL validate answer quality and flag potential hallucinations
7. IF LLM service fails THEN the system SHALL return error with available context information
8. WHEN citations are generated THEN each citation SHALL include chunk ID, document name, and confidence score

### Requirement 8: Document Management API

**User Story:** As a user, I want to upload, manage, and delete documents through a REST API, so that I can integrate the system with other applications.

#### Acceptance Criteria

1. WHEN multiple documents are uploaded THEN the system SHALL accept multipart/form-data and process files asynchronously
2. WHEN document upload starts THEN the system SHALL return document IDs and processing status immediately
3. WHEN document processing is complete THEN the system SHALL update status and store all extracted data
4. WHEN document list is requested THEN the system SHALL return documents with metadata, status, and processing statistics
5. WHEN document deletion is requested THEN the system SHALL remove all associated data (chunks, embeddings, graph nodes)
6. WHEN processing fails THEN the system SHALL update document status with error details
7. IF storage limits are exceeded THEN the system SHALL reject uploads with appropriate error message
8. WHEN document reprocessing is requested THEN the system SHALL allow strategy changes and update existing data

### Requirement 9: Search and Chat API

**User Story:** As a user, I want to search documents and have conversations about their content, so that I can extract insights and get answers to complex questions.

#### Acceptance Criteria

1. WHEN search request is received THEN the system SHALL validate query and document filters
2. WHEN search is performed THEN the system SHALL return answer with ranked source chunks and confidence scores
3. WHEN chat history is provided THEN the system SHALL maintain conversation context and reference previous exchanges
4. WHEN document filters are specified THEN the system SHALL limit search scope to selected documents
5. WHEN search options are provided THEN the system SHALL respect settings (hybrid search, agentic mode, result limits)
6. WHEN search fails THEN the system SHALL return appropriate error with available partial results
7. IF no relevant documents are found THEN the system SHALL clearly indicate lack of information
8. WHEN response includes sources THEN each source SHALL contain content excerpt, document reference, and relevance score

### Requirement 10: File Storage and Management

**User Story:** As a system administrator, I want reliable file storage with proper organization and access control, so that documents and processed data are safely stored and efficiently retrieved.

#### Acceptance Criteria

1. WHEN documents are uploaded THEN the system SHALL store original files in MinIO with proper organization by document ID
2. WHEN processed images are generated THEN the system SHALL store them in MinIO with appropriate metadata
3. WHEN file access is requested THEN the system SHALL provide secure URLs with configurable expiration
4. WHEN storage cleanup is needed THEN the system SHALL remove orphaned files and maintain storage efficiency
5. IF MinIO is unavailable THEN the system SHALL fallback to local file storage with proper error handling
6. WHEN file integrity is checked THEN the system SHALL validate checksums and detect corruption
7. WHEN backup is needed THEN the system SHALL support file export and import operations
8. WHEN storage limits are approached THEN the system SHALL implement cleanup policies and alerting

### Requirement 11: Caching and Performance

**User Story:** As a system administrator, I want the system to cache expensive operations and maintain good performance, so that users get fast responses and system resources are used efficiently.

#### Acceptance Criteria

1. WHEN LLM requests are made THEN the system SHALL cache responses in Redis with configurable TTL (3600 seconds)
2. WHEN embeddings are generated THEN the system SHALL cache results to avoid recomputation
3. WHEN ONNX models are loaded THEN the system SHALL keep them in memory for reuse
4. WHEN concurrent processing occurs THEN the system SHALL limit simultaneous operations (max 3 documents)
5. WHEN memory usage is high THEN the system SHALL implement graceful degradation
6. WHEN cache is full THEN the system SHALL use LRU eviction policy
7. IF Redis is unavailable THEN the system SHALL continue operation without caching
8. WHEN performance metrics are requested THEN the system SHALL provide processing times and resource usage

### Requirement 12: Monitoring and Logging

**User Story:** As a system administrator, I want comprehensive logging and monitoring, so that I can troubleshoot issues and optimize system performance.

#### Acceptance Criteria

1. WHEN any operation starts THEN the system SHALL log structured information with timestamps and correlation IDs
2. WHEN errors occur THEN the system SHALL log detailed error information with context and stack traces
3. WHEN processing completes THEN the system SHALL log performance metrics (time, resources, results)
4. WHEN system health is checked THEN the system SHALL verify all critical components and return detailed status
5. WHEN log levels are configured THEN the system SHALL respect settings (DEBUG, INFO, WARNING, ERROR)
6. WHEN logs are written THEN the system SHALL use structured JSON format for easy parsing
7. IF logging fails THEN the system SHALL continue operation and attempt to log to fallback destination
8. WHEN metrics are collected THEN the system SHALL track key performance indicators and resource usage

### Requirement 13: Configuration and Deployment

**User Story:** As a system administrator, I want flexible configuration and containerized deployment, so that I can deploy the system in different environments easily.

#### Acceptance Criteria

1. WHEN system starts THEN it SHALL load configuration from environment variables with validation
2. WHEN Docker container is built THEN it SHALL include all required dependencies and ONNX models
3. WHEN docker-compose is used THEN it SHALL orchestrate all required services (API, DB, Redis, Vector Store)
4. WHEN configuration is invalid THEN the system SHALL fail fast with clear error messages
5. WHEN different environments are used THEN the system SHALL support dev/staging/production configurations
6. WHEN secrets are needed THEN the system SHALL load them securely without exposing in logs
7. IF required services are unavailable THEN the system SHALL implement proper health checks and retries
8. WHEN system is deployed THEN it SHALL provide readiness and liveness probes for orchestration

### Requirement 14: Trio Async Infrastructure

**User Story:** As a system administrator, I want trio-based async processing instead of asyncio, so that concurrent LLM requests and heavy operations are properly managed with better performance.

#### Acceptance Criteria

1. WHEN system initializes THEN it SHALL use trio instead of asyncio for all async operations
2. WHEN LLM requests are made THEN the system SHALL use trio.CapacityLimiter for concurrency control (max 10 concurrent)
3. WHEN blocking operations are needed THEN the system SHALL use trio.to_thread.run_sync for thread execution
4. WHEN timing operations THEN the system SHALL use trio.current_time() for accurate measurements
5. WHEN HTTP requests are needed THEN the system SHALL use asks library instead of httpx/aiohttp
6. WHEN FastAPI runs THEN it SHALL be configured to work with trio backend
7. IF trio operations fail THEN the system SHALL implement proper error handling and fallbacks
8. WHEN performance monitoring THEN the system SHALL track trio-specific metrics and resource usage

### Requirement 15: Advanced Document Parsers

**User Story:** As a user, I want specialized parsers for each document type, so that structure, formatting, and metadata are preserved accurately for better search results.

#### Acceptance Criteria

1. WHEN DOCX files are processed THEN the system SHALL use RAGFlowDocxParser with advanced table extraction and structure preservation
2. WHEN Excel files are processed THEN the system SHALL use RAGFlowExcelParser with HTML conversion and data type detection
3. WHEN PowerPoint files are processed THEN the system SHALL use RAGFlowPptParser with slide structure and embedded content extraction
4. WHEN JSON files are processed THEN the system SHALL use RAGFlowJsonParser with intelligent splitting for large files
5. WHEN Markdown files are processed THEN the system SHALL use RAGFlowMarkdownParser with table detection and structure preservation
6. WHEN HTML files are processed THEN the system SHALL use readability-lxml for content cleaning and extraction
7. WHEN parser selection occurs THEN the system SHALL automatically choose appropriate parser based on file type and content
8. IF specialized parsing fails THEN the system SHALL fallback to basic text extraction without failing

### Requirement 16: Custom Tokenization and NLP

**User Story:** As a user, I want proper Russian and English language tokenization with term weighting, so that search quality matches native language patterns and provides accurate results.

#### Acceptance Criteria

1. WHEN text is tokenized THEN the system SHALL use RagTokenizer for both Russian and English languages
2. WHEN fine-grained tokenization is needed THEN the system SHALL provide detailed token splitting with frequency analysis
3. WHEN term weighting is calculated THEN the system SHALL use TF-IDF based weights with custom dictionaries
4. WHEN synonyms are processed THEN the system SHALL expand queries using WordNet and custom synonym dictionaries
5. WHEN Chinese text is encountered THEN the system SHALL handle traditional/simplified conversion and pinyin processing
6. WHEN token frequencies are needed THEN the system SHALL provide frequency lookup from built-in dictionaries
7. WHEN query preprocessing occurs THEN the system SHALL apply language-specific normalization and cleaning
8. IF tokenization fails THEN the system SHALL fallback to basic word splitting without breaking the pipeline

### Requirement 17: External Search Integration

**User Story:** As a user, I want agentic reasoning with external search capabilities, so that answers can include information beyond uploaded documents for comprehensive responses.

#### Acceptance Criteria

1. WHEN agentic mode is enabled THEN the system SHALL integrate Tavily API for external web search
2. WHEN complex queries are detected THEN the system SHALL perform multi-step reasoning with iterative searches
3. WHEN search queries are generated THEN the system SHALL extract search terms using structured prompts and tags
4. WHEN external search is performed THEN the system SHALL retrieve and process web results with confidence scoring
5. WHEN search limits are reached THEN the system SHALL respect maximum search iterations (6 max) and provide appropriate messages
6. WHEN reasoning steps are tracked THEN the system SHALL maintain conversation history and context across iterations
7. WHEN external results are integrated THEN the system SHALL combine internal and external sources with proper attribution
8. IF external search fails THEN the system SHALL continue with internal knowledge base without breaking the response flow

### Requirement 18: Advanced Table Processing

**User Story:** As a user, I want sophisticated table structure recognition and preservation, so that tabular data maintains its meaning and relationships in search results.

#### Acceptance Criteria

1. WHEN tables are detected THEN the system SHALL recognize complex structures including merged cells, headers, and spanning elements
2. WHEN table content is extracted THEN the system SHALL preserve row and column relationships with proper HTML/Markdown formatting
3. WHEN table data types are analyzed THEN the system SHALL automatically detect and classify column types (text, number, date, boolean)
4. WHEN table chunking occurs THEN the system SHALL create chunks that preserve table headers and logical row groupings
5. WHEN table reconstruction happens THEN the system SHALL generate both HTML and Markdown representations
6. WHEN table validation is performed THEN the system SHALL verify structure integrity and data consistency
7. WHEN cross-page tables are encountered THEN the system SHALL handle table continuation across document pages
8. IF table processing fails THEN the system SHALL extract table content as structured text without losing information

### Requirement 19: Performance Optimization

**User Story:** As a system administrator, I want optimized caching and hashing for high-performance operations, so that the system responds quickly and uses resources efficiently.

#### Acceptance Criteria

1. WHEN hashing operations are needed THEN the system SHALL use xxhash for fast cache key generation
2. WHEN LLM responses are cached THEN the system SHALL use optimized hash-based cache keys with proper TTL management
3. WHEN embeddings are cached THEN the system SHALL implement efficient storage and retrieval with hash-based indexing
4. WHEN geometric operations are performed THEN the system SHALL use pyclipper for PDF coordinate processing
5. WHEN string comparisons are needed THEN the system SHALL use editdistance for efficient similarity calculations
6. WHEN JSON repair is required THEN the system SHALL use json-repair for handling malformed JSON data
7. WHEN text processing occurs THEN the system SHALL use optimized algorithms for word-to-number conversion and text normalization
8. IF performance degrades THEN the system SHALL implement monitoring and alerting for cache hit rates and processing times

### Requirement 20: RAPTOR Hierarchical Clustering

**User Story:** As a user, I want RAPTOR hierarchical clustering and summarization, so that documents are organized in multi-level abstractions for better retrieval at different granularities.

#### Acceptance Criteria

1. WHEN RAPTOR is enabled THEN the system SHALL use UMAP for dimensionality reduction with cosine metric
2. WHEN clustering is performed THEN the system SHALL use GaussianMixture with BIC for optimal cluster selection
3. WHEN cluster summaries are generated THEN the system SHALL create hierarchical abstractions using LLM
4. WHEN embeddings are processed THEN the system SHALL reduce dimensions to 10 components with min_dist=0.0
5. WHEN optimal clusters are determined THEN the system SHALL use Bayesian Information Criterion for selection
6. WHEN hierarchical structure is built THEN the system SHALL create tree-organized retrieval from specific to general
7. WHEN cluster validation occurs THEN the system SHALL ensure coherent semantic groupings
8. IF clustering fails THEN the system SHALL fallback to flat chunking without breaking the pipeline

### Requirement 21: Advanced ML and NLP Libraries

**User Story:** As a system administrator, I want comprehensive ML and NLP library support, so that the system can perform sophisticated text analysis and machine learning operations.

#### Acceptance Criteria

1. WHEN ML operations are needed THEN the system SHALL use scikit-learn for clustering and classification algorithms
2. WHEN tokenization is performed THEN the system SHALL use datrie for efficient trie-based token storage
3. WHEN geometric operations are needed THEN the system SHALL use shapely for PDF coordinate processing
4. WHEN token counting is required THEN the system SHALL use tiktoken for OpenAI-compatible tokenization
5. WHEN NLP processing occurs THEN the system SHALL use NLTK for advanced text processing capabilities
6. WHEN caching is implemented THEN the system SHALL use cachetools for advanced caching strategies
7. WHEN serialization is needed THEN the system SHALL use ormsgpack for high-performance MessagePack serialization
8. IF Redis is unavailable THEN the system SHALL use valkey as Redis-compatible alternative

### Requirement 22: Multi-language Text Processing

**User Story:** As a user, I want comprehensive multi-language text processing, so that the system can handle Chinese, Russian, English and other languages with proper normalization and conversion.

#### Acceptance Criteria

1. WHEN Chinese text is processed THEN the system SHALL use hanziconv for traditional/simplified conversion
2. WHEN Chinese numbers are encountered THEN the system SHALL use cn2an for number conversion to Arabic numerals
3. WHEN pinyin is needed THEN the system SHALL use xpinyin for Chinese character pronunciation
4. WHEN Roman numerals are found THEN the system SHALL use roman-numbers for conversion and processing
5. WHEN Markdown conversion is needed THEN the system SHALL use markdown-to-json for structured data extraction
6. WHEN table formatting is required THEN the system SHALL use tabulate for proper table display
7. WHEN word-to-number conversion occurs THEN the system SHALL handle multiple languages appropriately
8. IF language detection fails THEN the system SHALL default to English processing without errors

### Requirement 23: Extended LLM Provider Support

**User Story:** As a user, I want support for multiple LLM providers including Chinese and specialized embedding models, so that I can choose the best models for my specific use case and language requirements.

#### Acceptance Criteria

1. WHEN Chinese LLM is needed THEN the system SHALL support Qwen models through dashscope integration
2. WHEN ChatGLM is required THEN the system SHALL integrate zhipuai for Zhipu AI models
3. WHEN ByteDance models are used THEN the system SHALL support volcengine integration
4. WHEN specialized embeddings are needed THEN the system SHALL support Voyage AI through voyageai
5. WHEN Chinese embeddings are required THEN the system SHALL use bcembedding for Chinese text
6. WHEN BGE embeddings are needed THEN the system SHALL integrate flagembedding models
7. WHEN fast embeddings are required THEN the system SHALL support fastembed for CPU/GPU acceleration
8. IF primary LLM provider fails THEN the system SHALL fallback to alternative providers seamlessly

### Requirement 24: Text File Processing

**User Story:** As a user, I want specialized text file processing with intelligent chunking, so that plain text documents are processed with proper structure detection and formatting.

#### Acceptance Criteria

1. WHEN TXT files are uploaded THEN the system SHALL use RAGFlowTxtParser for processing
2. WHEN text encoding is detected THEN the system SHALL automatically determine file encoding using chardet
3. WHEN text is chunked THEN the system SHALL use configurable delimiters and token limits
4. WHEN text structure is analyzed THEN the system SHALL detect bullet points and hierarchical structure
5. WHEN text is processed THEN the system SHALL preserve formatting and line breaks appropriately
6. WHEN large text files are encountered THEN the system SHALL handle them efficiently with streaming
7. WHEN text chunking occurs THEN the system SHALL maintain context overlap between chunks
8. IF text processing fails THEN the system SHALL fallback to basic line-by-line processing

### Requirement 25: Task Execution System

**User Story:** As a system administrator, I want an asynchronous task execution system with monitoring and concurrency control, so that document processing tasks are managed efficiently and reliably.

#### Acceptance Criteria

1. WHEN tasks are submitted THEN the system SHALL use trio.Semaphore for concurrency control
2. WHEN multiple tasks run THEN the system SHALL limit concurrent document processing (max 5 tasks)
3. WHEN chunk building occurs THEN the system SHALL use trio.CapacityLimiter (max 1 concurrent)
4. WHEN MinIO operations happen THEN the system SHALL limit concurrent file operations (max 10)
5. WHEN tasks are executed THEN the system SHALL provide progress tracking and status updates
6. WHEN tasks are cancelled THEN the system SHALL handle cancellation gracefully with cleanup
7. WHEN worker health is monitored THEN the system SHALL implement heartbeat mechanism
8. IF task execution fails THEN the system SHALL provide detailed error information and retry logic

### Requirement 26: Entity Resolution

**User Story:** As a user, I want automatic entity resolution in the knowledge graph, so that duplicate entities are merged and the graph maintains consistency.

#### Acceptance Criteria

1. WHEN entities are extracted THEN the system SHALL detect potential duplicates using similarity matching
2. WHEN duplicate entities are found THEN the system SHALL merge them using entity resolution algorithms
3. WHEN entities are merged THEN the system SHALL combine their attributes and relationships
4. WHEN entity resolution occurs THEN the system SHALL maintain referential integrity in the graph
5. WHEN resolution confidence is low THEN the system SHALL flag entities for manual review
6. WHEN entities are resolved THEN the system SHALL update all related chunks and references
7. WHEN resolution rules are configured THEN the system SHALL apply custom matching criteria
8. IF resolution fails THEN the system SHALL log conflicts and continue with unresolved entities

### Requirement 27: Advanced NLP Utilities

**User Story:** As a user, I want advanced NLP utilities for text processing, so that documents are analyzed with sophisticated language understanding capabilities.

#### Acceptance Criteria

1. WHEN bullet points are detected THEN the system SHALL use bullets_category for classification
2. WHEN hierarchical merging is needed THEN the system SHALL use hierarchical_merge with level detection
3. WHEN simple merging is required THEN the system SHALL use naive_merge with token limits
4. WHEN tables are tokenized THEN the system SHALL use tokenize_table with proper formatting
5. WHEN language detection is needed THEN the system SHALL use is_english/is_chinese functions
6. WHEN content tables are found THEN the system SHALL use remove_contents_table for cleanup
7. WHEN text structure is analyzed THEN the system SHALL detect titles, headers, and sections
8. IF NLP processing fails THEN the system SHALL fallback to basic text processing methods

### Requirement 28: Image Processing Operators

**User Story:** As a system administrator, I want comprehensive image processing operators for computer vision pipeline, so that document images are processed with professional-grade transformations.

#### Acceptance Criteria

1. WHEN images are decoded THEN the system SHALL use DecodeImage operator with proper format handling
2. WHEN image normalization is needed THEN the system SHALL use NormalizeImage with configurable parameters
3. WHEN channel conversion is required THEN the system SHALL use ToCHWImage for tensor format
4. WHEN image padding is needed THEN the system SHALL use Pad operator with configurable sizes
5. WHEN image resizing is required THEN the system SHALL use LinearResize and Resize operators
6. WHEN detection preprocessing is needed THEN the system SHALL use DetResizeForTest operator
7. WHEN grayscale conversion is required THEN the system SHALL use GrayImageChannelFormat
8. IF image processing fails THEN the system SHALL provide fallback processing with error logging
