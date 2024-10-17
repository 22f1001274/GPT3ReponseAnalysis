# Project Title: GPT-3 Conversational Chat with Sentiment Analysis, Reporting, and Database Integration
## Author: Burhan Iqbal Khan
## Contact: 7780898787

## Project Overview
This project involves developing a Python-based application that interacts with users via a conversational chat interface. It leverages OpenAI's GPT-3 model (accessed through Hugging Face API) to generate human-like responses to user inputs. Additionally, the application performs sentiment analysis on the responses, stores both conversations and sentiment data in a PostgreSQL database, and generates visual reports summarizing the sentiment analysis results.

The project employs various frameworks and tools, such as LangChain for GPT-3 conversation handling, Streamlit for the user interface, TextBlob for sentiment analysis, SQLAlchemy for database management, and Plotly for data visualizations.

## Task Description
Develop a Python program that interacts with the user and generates responses using OpenAI's GPT-3 model through the Hugging Face API.
Perform sentiment analysis on the AI-generated responses using the TextBlob library.
Store the user input, AI responses, and sentiment analysis results in a PostgreSQL database using SQLAlchemy ORM.
Generate sentiment analysis reports with data visualizations using Plotly and Streamlit.
Create a summary of the conversation using the GPT-3 model.
Deploy the program using Docker.
## Features
Conversational Chat: The application engages in interactive conversations with the user using GPT-3.
Sentiment Analysis: Each response from GPT-3 is analyzed for sentiment (positive, negative, or neutral) using TextBlob.
Database Integration: User inputs, AI responses, and sentiment results are stored in a PostgreSQL database using SQLAlchemy.
Data Visualization: The application generates pie and bar charts to represent the distribution of sentiments using Plotly.
Conversation Summary: The entire conversation is summarized using GPT-3 for review purposes.
Docker Deployment: A Dockerfile is included to easily deploy the application.
## Tech Stack
Programming Language: Python
Frontend: Streamlit
AI Framework: LangChain, Hugging Face API (OpenAI GPT-3)
Sentiment Analysis: TextBlob
Database: PostgreSQL
ORM: SQLAlchemy
Data Visualization: Plotly, Streamlit
Deployment: Docker
Setup and Installation
Clone this repository:

'''python
git clone [https://github.com/your-repo-name.git](https://github.com/22f1001274/GPT3ReponseAnalysis.git)
'''
Install dependencies from the requirements.txt file:

'''python
pip install -r requirements.txt
'''
Set up your environment variables. Create a .env file with the following:

makefile
'''python
API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_database_url
'''
Run the Streamlit application:

'''python
streamlit run main.py
'''
To deploy the application using Docker, build the Docker image and run the container:

'''python
docker build -t gpt3-sentiment-app .
docker run -p 8501:8501 gpt3-sentiment-app
'''
## How It Works
User Interaction: Users input text into the chat interface, and the application generates responses using GPT-3.
Sentiment Analysis: Sentiment is analyzed for each response, and both the user’s input and AI's response are stored in a PostgreSQL database.
Data Visualization: Sentiment data is visualized in pie and bar charts, providing insights into the overall sentiment of the conversation.
Conversation Summary: A summary of the conversation is generated based on the entire chat history.
Report Generation: Users can generate a report at any time to view the conversation summary and sentiment analysis.
