
# RAG System - Chat & Learn

## Introduction
This assignment aims to provide insight into the type of technical work you may encounter in our organization while allowing you to showcase your thought processes and problem-solving skills. We want to ensure that our organization is a good fit for you as much as you are for us.

## Important Notes
- **Deep Dive Discussion**: Once shortlisted, we will conduct a detailed discussion to understand your thought process and technical approach.
- **Resource Utilization**: Feel free to leverage any online resources, including assistance from language models (LLMs). If you use an LLM, please share how it contributed to your assignment.
- **Python Libraries**: You are encouraged to use any Python libraries that you find suitable for the tasks.

## Assignment Overview
You are expected to complete Parts 1 and 2 of the assignment outlined below. Bonus points will be awarded for completing Part 3 to test your familiarity with the subject matter.

### Assignment Part 1: Building a RAG System
RAG (Retrieval-Augmented Generation) systems are commonly used patterns powering various AI applications. In this part, you will build a RAG system that utilizes an external data source stored in a vector database alongside a language model. Specifically, you will work with NCERT PDF text.

**Requirements**:
1. Build a RAG system that operates on the NCERT dataset.
2. Serve the RAG system through a FastAPI endpoint, allowing users to send queries and receive responses.
3. Optionally, develop a frontend interface to showcase this endpoint.

### Assignment Part 2: Building an Intelligent Agent
In this part, you will extend the RAG system by building an agent that can perform smart actions based on user queries.

**Requirements**:
1. Create another FastAPI endpoint for the agent.
2. Ensure the agent can determine when to query the VectorDB (e.g., it should not call the VectorDB for simple greetings).
3. Introduce at least one additional action/tool that the agent can invoke based on the user's query. Bonus points for implementing more creative actions!

## Bonus Part 3: (Optional)
Explore additional functionalities or enhancements for your RAG system and intelligent agent. This could include:
- Improving the agent's decision-making capabilities.
- Adding more complex actions or tools.
- Integrating a user feedback mechanism.

## Getting Started
### Prerequisites
- Python 3.x
- FastAPI
- Required libraries (list libraries you plan to use)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI application:
   ```bash
   chainlit run <filename.py>
   ```

### Usage
- Access the API documentation at `http://127.0.0.1:8000/docs` to test the endpoints.
- For the frontend (if applicable), navigate to the respective directory and follow its instructions.

## Contributing
Feel free to contribute by submitting pull requests or opening issues.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
