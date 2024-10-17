# Import necessary libraries

# To load environmental variables from .env file
import os
from dotenv import load_dotenv

# For building user interface with Streamlit
import streamlit as st

# LangChain framework for building LLM applications
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# For sentiment analysis using TextBlob
from textblob import TextBlob

# For data handling and visualization
import pandas as pd
import plotly.express as px

# SQLAlchemy ORM for interacting with PostgreSQL in a declarative way
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# To handle timestamps
from datetime import datetime

# Load environment variables (e.g., API keys, database credentials)
load_dotenv()

# Set the OpenAI API key from the environment variable
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY')

# Initialize SQLAlchemy's declarative base (for defining models)
Base = declarative_base()

# Define the chat_history table as a Python class (SQLAlchemy model)
class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    user_input = Column(Text)  # Stores the user's input
    ai_response = Column(Text)  # Stores the AI-generated response
    sentiment = Column(String(50))  # Sentiment classification (Positive/Negative/Neutral)
    timestamp = Column(DateTime, default=datetime.now)  # Timestamp of the interaction

# Define the sentiment_counts table as a Python class (SQLAlchemy model)
class SentimentCounts(Base):
    __tablename__ = 'sentiment_counts'
    sentiment_type = Column(String(50), primary_key=True)  # Sentiment type (Positive/Negative/Neutral)
    count = Column(Integer)  # Count of how many times each sentiment appeared

def init_db_connection():
    """Initialize the database connection and return a session."""
    try:
        db_url = os.getenv('DATABASE_URL')  # Database URL from environment variables
        engine = create_engine(db_url)  # Create an engine to connect to the database
        Base.metadata.create_all(engine)  # Create tables if they don't exist
        Session = sessionmaker(bind=engine)  # Bind the engine to the session
        return Session()
    except SQLAlchemyError as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def store_chat_data(user_input, ai_response, sentiment):
    """Store chat data (user input, AI response, and sentiment) in the database."""
    session = init_db_connection()
    if session is None:
        return  # Exit if the session couldn't be created
    try:
        new_chat = ChatHistory(
            user_input=user_input,
            ai_response=ai_response,
            sentiment=sentiment,
            timestamp=datetime.now()  # Record the time of the interaction
        )
        session.add(new_chat)  # Add the new chat record
        session.commit()  # Commit the changes to the database
    except SQLAlchemyError as e:
        st.error(f"Error storing chat data: {str(e)}")
    finally:
        session.close()  # Close the session

def retrieve_chat_data():
    """Retrieve all chat history from the database."""
    session = init_db_connection()
    if session is None:
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    try:
        chats = session.query(ChatHistory).all()  # Query all chat history
        chat_df = pd.DataFrame(
            [(chat.user_input, chat.ai_response, chat.sentiment, chat.timestamp) for chat in chats],
            columns=['user_input', 'ai_response', 'sentiment', 'timestamp']
        )
        return chat_df
    except SQLAlchemyError as e:
        st.error(f"Error retrieving chat data: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def update_sentiment_count(sentiment):
    """Update sentiment counts in the database."""
    session = init_db_connection()
    if session is None:
        return  # Exit if the session couldn't be created
    try:
        sentiment_record = session.query(SentimentCounts).filter_by(sentiment_type=sentiment).first()
        if sentiment_record:
            sentiment_record.count += 1  # Increment the count for existing sentiment
        else:
            new_sentiment = SentimentCounts(sentiment_type=sentiment, count=1)  # Create a new record if it doesn't exist
            session.add(new_sentiment)
        session.commit()  # Commit the changes to the database
    except SQLAlchemyError as e:
        st.error(f"Error updating sentiment count: {str(e)}")
    finally:
        session.close()

def retrieve_sentiment_counts():
    """Retrieve sentiment counts from the database."""
    session = init_db_connection()
    if session is None:
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    try:
        sentiment_counts = session.query(SentimentCounts).all()  # Query all sentiment counts
        sentiment_df = pd.DataFrame(
            [(sentiment.sentiment_type, sentiment.count) for sentiment in sentiment_counts],
            columns=['sentiment_type', 'count']
        )
        return sentiment_df
    except SQLAlchemyError as e:
        st.error(f"Error retrieving sentiment counts: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def init_chat_model():
    """Initialize the GPT-3 model using LangChain."""
    return ChatOpenAI(temperature=0.7)

def chat_with_gpt3(user_input, conversation):
    """Handle conversation logic with GPT-3."""
    try:
        response = conversation.predict(input=user_input)  # Get GPT-3's response
        return response
    except Exception as e:
        st.error(f"Error during GPT-3 interaction: {str(e)}")
        return ""

def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob."""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def generate_summary(conversation):
    """Generate a summary of the conversation using GPT-3."""
    try:
        summary_prompt = "Summarize this conversation:\n" + conversation.memory.buffer
        summary = conversation.predict(input=summary_prompt)
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

def generate_sentiment_report():
    """Generate a sentiment analysis report with visualization."""
    st.write("### Sentiment Analysis Report")

    sentiment_df = retrieve_sentiment_counts()

    if sentiment_df.empty:
        st.warning("No sentiment data available.")
        return

    # Display the counts for each sentiment
    for index, row in sentiment_df.iterrows():
        st.write(f"{row['sentiment_type']}: {row['count']} responses")

    # Pie chart visualization using Plotly
    fig = px.pie(sentiment_df, values='count', names='sentiment_type', title='Sentiment Distribution')
    st.plotly_chart(fig)

    # Bar chart visualization using Streamlit's built-in charting tools
    st.bar_chart(sentiment_df.set_index('sentiment_type'))


def main():
    st.title("GPT-3 Conversational Chat with Sentiment Analysis, Reporting, and Database Integration")

    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(
            llm=init_chat_model(),
            memory=ConversationBufferMemory()
        )

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.user_input = st.sidebar.text_input("Let's Chat", st.session_state.user_input)

    if st.sidebar.button("Generate Report"):
        summary = generate_summary(st.session_state.conversation)
        st.write("### Chat Summary")
        st.write(summary)

        generate_sentiment_report()

        st.write("### Chat History (Stored in Database)")
        chat_df = retrieve_chat_data()
        st.write(chat_df)

    if st.session_state.user_input:
        response = chat_with_gpt3(st.session_state.user_input, st.session_state.conversation)
        sentiment = analyze_sentiment(response)

        st.session_state.chat_history.append(f"You: {st.session_state.user_input}")
        st.session_state.chat_history.append(f"AI: {response} (Sentiment: {sentiment})")

        store_chat_data(st.session_state.user_input, response, sentiment)
        update_sentiment_count(sentiment)

        st.session_state.user_input = ""

    for message in st.session_state.chat_history:
        st.write(message)

if __name__ == "__main__":
    main()
