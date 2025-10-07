"""LangChain pipeline orchestrating Gemini responses with career insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage

from firebase import db
from services.utils import get_logger

from .gemini_client import GeminiClient
from .career_insights import CareerDecisionTree, create_user_profile_from_query

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are CareerSaathi, a trusted career coach with access to data-driven career insights. "
    "Provide empathetic, practical guidance grounded in current job market trends. When giving advice, "
    "offer actionable next steps, resources, and encouragement. Use career insights and salary data when relevant. "
    "If you do not have enough information, ask clarifying questions."
)


@dataclass
class CareerCounselorChain:
    gemini_client: GeminiClient
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history_limit: int = 10
    _message_histories: Dict[str, List] = field(default_factory=dict, init=False)
    _career_tree: Optional[CareerDecisionTree] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize and train the career decision tree."""
        try:
            self._career_tree = CareerDecisionTree()
            
            # Try to load existing model, otherwise train new one
            model_path = "career_model.joblib"
            if os.path.exists(model_path):
                self._career_tree.load_model(model_path)
                logger.info("Loaded existing career model")
            else:
                self._career_tree.train_model()
                self._career_tree.save_model(model_path)
                logger.info("Trained and saved new career model")
                
        except Exception as e:
            logger.warning(f"Failed to initialize career insights: {e}")
            self._career_tree = None

    def _get_career_insights(self, user_message: str) -> str:
        """Generate career insights based on user query."""
        if not self._career_tree:
            return "Career insights unavailable."
        
        try:
            # Check if the message is career-related
            career_keywords = ['career', 'job', 'salary', 'role', 'position', 'skills', 'growth', 'recommendation']
            if not any(keyword in user_message.lower() for keyword in career_keywords):
                return ""
            
            # Extract user profile from query
            user_profile = create_user_profile_from_query(user_message)
            
            # Get career predictions
            prediction = self._career_tree.predict_career_path(user_profile)
            insights = self._career_tree.get_career_insights(prediction['primary_recommendation'])
            
            # Format insights
            career_context = f"""
CAREER INSIGHTS:
- Recommended Career Track: {prediction['primary_recommendation']} (Confidence: {prediction['confidence']:.1%})
- Reasoning: {prediction['reasoning']}
- Average Salary in Track: ${insights['avg_salary']:,.0f}
- Salary Range: ${insights['salary_range']['min']:,.0f} - ${insights['salary_range']['max']:,.0f}
- Top Roles in Track:
"""
            for role in prediction['top_roles'][:3]:
                career_context += f"  • {role['role']}: ${role['avg_salary']:,.0f}\n"
            
            return career_context
            
        except Exception as e:
            logger.warning(f"Failed to generate career insights: {e}")
            return ""

    def _get_message_history(self, user_id: str | None) -> List:
        key = user_id or "anonymous"
        if key not in self._message_histories:
            self._message_histories[key] = []
        return self._message_histories[key]

    def _build_chain(self) -> RunnableSequence:
        llm = self.gemini_client.get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{user_message}\n\nContext:\n{context}"),
            ]
        )
        parser = StrOutputParser()
        return prompt | llm | parser

    def invoke(
        self,
        user_id: Optional[str],
        user_message: str,
        *,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Run the chain, persist history, and return the AI's reply."""
        message_history = self._get_message_history(user_id)
        chain = self._build_chain()
        
        # Generate career insights if relevant
        career_insights = self._get_career_insights(user_message)
        
        # Combine context with career insights
        combined_context = context or "No additional context provided."
        if career_insights:
            combined_context += f"\n\n{career_insights}"
        
        inputs = {
            "user_message": user_message,
            "context": combined_context,
            "history": message_history[-self.history_limit:],  # Keep only recent messages
        }
        logger.info("Invoking LangChain pipeline for user %s", user_id or "anonymous")
        response_text = chain.invoke(inputs)
        
        # Add messages to history
        message_history.append(HumanMessage(content=user_message))
        message_history.append(AIMessage(content=response_text))

        if user_id:
            try:
                db.save_history_entry(
                    user_id,
                    user_message,
                    response_text,
                    metadata={
                        "context": combined_context,
                        "model": self.gemini_client.model_name,
                        "has_career_insights": bool(career_insights),
                        **(metadata or {}),
                    },
                )
            except Exception as exc:  # pragma: no cover - external dependency
                logger.warning("Failed to save history for user %s: %s", user_id, exc)

        return response_text

    def get_recent_history(self, user_id: Optional[str]) -> List[Dict[str, str]]:
        if not user_id:
            message_history = self._get_message_history(None)
            return [
                {"role": msg.type, "content": msg.content}
                for msg in message_history[-self.history_limit :]
            ]
        try:
            return db.list_history(user_id, limit=self.history_limit)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.warning("Falling back to in-memory history for user %s: %s", user_id, exc)
            message_history = self._get_message_history(user_id)
            return [
                {"role": msg.type, "content": msg.content}
                for msg in message_history[-self.history_limit :]
            ]
