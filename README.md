# LangGraph Customer Support Agent

AI-powered customer support automation using LangGraph with 11-stage workflow. Features Gemini AI integration, state persistence, MCP client simulation, and intelligent solution evaluation for enterprise support systems.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow Stages](#workflow-stages)
- [MCP Client System](#mcp-client-system)
- [State Management](#state-management)
- [API Integration](#api-integration)
- [Error Handling](#error-handling)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements "Langie" - a structured LangGraph Agent that processes customer support requests through an 11-stage workflow. The system combines deterministic sequential processing with non-deterministic AI-powered decision points, demonstrating enterprise-grade workflow orchestration patterns.

### Key Capabilities

- End-to-end customer query processing
- Intelligent intent classification and sentiment analysis
- Dynamic solution generation and evaluation
- Automated escalation decision-making
- Professional response generation
- Comprehensive audit trails

## Architecture

The system follows a pipeline architecture with clear separation of concerns:

```
Input → [11 Sequential Stages] → Output
         ↓
    State Persistence
         ↓
    MCP Client System (COMMON/ATLAS)
         ↓
    AI Integration Points
```

### Core Components

- **StateGraph**: LangGraph v2 workflow engine
- **WorkflowState**: TypedDict for state management
- **MCPClient**: Simulated service layer
- **AI Integration**: Strategic Gemini API calls

## Features

- **Graph-Based Workflow**: 11 sequential stages with deterministic and non-deterministic execution modes
- **AI-Powered Decision Making**: Gemini AI integration for query analysis, solution generation, and response creation
- **State Persistence**: Comprehensive state management across all workflow stages
- **MCP Architecture**: Dual-server simulation (COMMON for AI, ATLAS for external systems)
- **Error Resilience**: Robust fallback mechanisms for AI and service failures
- **Audit Trail**: Complete logging and execution history

## Installation

### Prerequisites

- Python 3.8+
- Google AI API key (Gemini)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langgraph-customer-support-agent.git
cd langgraph-customer-support-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Gemini API key
```

### Dependencies

```
langgraph>=0.2.0
google-generativeai>=0.7.0
python-dotenv>=1.0.0
asyncio
logging
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
LOG_LEVEL=INFO
```

### Getting Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Usage

### Basic Usage

```python
import asyncio
from app import run_customer_support_agent

# Define customer query
input_payload = {
    "customer_name": "John Smith",
    "email": "john@example.com",
    "query": "Where is my order #123?",
    "priority": "medium",
    "ticket_id": "T001"
}

# Run the agent
result = await run_customer_support_agent(input_payload)
print(result)
```

### Running the Demo

```bash
python app.py
```

This will execute a sample customer support scenario with complete logging.

## Workflow Stages

| Stage | Name | Mode | Description | AI Integration |
|-------|------|------|-------------|----------------|
| 1 | INTAKE | Deterministic | Accept and validate payload | No |
| 2 | UNDERSTAND | Deterministic | Parse query and extract entities | Yes (Gemini) |
| 3 | PREPARE | Deterministic | Normalize and enrich data | No |
| 4 | ASK | Human Interaction | Request clarification if needed | No |
| 5 | WAIT | Deterministic | Process customer response | No |
| 6 | RETRIEVE | Deterministic | Search knowledge base | No |
| 7 | DECIDE | Non-deterministic | Generate and evaluate solutions | Yes (Gemini) |
| 8 | UPDATE | Deterministic | Update ticket status | No |
| 9 | CREATE | Deterministic | Generate customer response | Yes (Gemini) |
| 10 | DO | Deterministic | Execute actions and notifications | No |
| 11 | COMPLETE | Deterministic | Output final payload | No |

## MCP Client System

The system uses two simulated MCP servers:

### COMMON Server (AI Processing)
- Text parsing and analysis
- Solution generation and evaluation
- Response generation
- Priority calculations

### ATLAS Server (External Integrations)
- Entity extraction
- Database operations
- API integrations
- Notification services

## State Management

The `WorkflowState` TypedDict manages 25+ variables including:

### Input Data
- customer_name, email, query, priority, ticket_id

### Analysis Results
- parsed_intent, sentiment, entities, urgency

### Processing Data
- knowledge_base_results, solutions, chosen_solution

### Output Data
- generated_response, executed_actions, final_status

### Metadata
- stage_history, processing_time, timestamp

## API Integration

### Gemini AI Integration

The system makes strategic AI calls at three key points:

1. **Stage 2 - Query Analysis**: Intent classification, sentiment analysis, entity extraction
2. **Stage 7 - Solution Generation**: Dynamic solution creation and evaluation
3. **Stage 9 - Response Creation**: Professional customer response generation

### Error Handling

Each AI integration includes robust fallback mechanisms:
- JSON parsing with multiple strategies
- Default values for failed operations
- Warning logs for debugging
- Graceful degradation

## Error Handling

### AI Call Failures
- Try-catch blocks around all Gemini API calls
- Fallback logic for each capability
- Detailed error logging

### State Validation
- Required field checks
- Type validation
- Default value assignment

### Workflow Resilience
- Individual stage error isolation
- Continuation with partial data
- Comprehensive audit trails

## Demo

The included demo processes a sample customer query:

```json
{
  "customer_name": "John Smith",
  "email": "",
  "query": "When will my order come? My order id is #456 and my last name is Smith",
  "priority": "medium",
  "ticket_id": "T001"
}
```

Expected output includes:
- Complete workflow execution logs
- AI analysis results
- Generated solutions and evaluations
- Professional customer response
- Final structured payload

## Project Structure

```
├── app.py                 # Main application file
├── .env                   # Environment variables
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── logs/                # Log files (created at runtime)
```

## Development

### Running Tests

```bash
# Run the demo
python app.py

# Enable debug logging
LOG_LEVEL=DEBUG python app.py
```

### Key Design Patterns

- **Pipeline Pattern**: Sequential stage processing
- **Strategy Pattern**: MCP server implementations
- **State Machine**: Workflow state evolution
- **Dependency Injection**: Service layer abstraction

### Extending the System

To add new stages:
1. Define stage function following the pattern
2. Add to workflow builder
3. Update state schema if needed
4. Add appropriate edges

To add new AI capabilities:
1. Implement in `_execute_common_ability`
2. Add fallback logic
3. Update stage functions to use new capability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangGraph for workflow orchestration
- Powered by Google Gemini AI
- Designed for enterprise customer support automation

## Support

For questions or issues:
1. Check the error logs in the console output
2. Verify your Gemini API key is correctly set
3. Ensure all dependencies are installed
4. Review the workflow stage logs for debugging

## Future Enhancements

Potential improvements for production deployment:
- Real MCP server integration
- Database persistence layer
- Web API interface
- Multi-tenant support
- Advanced analytics and reporting
- Integration with existing CRM systems
