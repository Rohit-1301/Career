"""LangChain pipeline orchestrating Gemini responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage

from firebase import db
from services.utils import get_logger

from .gemini_client import GeminiClient

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are CareerSaathi, a trusted career coach. Provide empathetic, practical guidance "
    "grounded in current job market trends. When giving advice, offer actionable next steps, "
    "resources, and encouragement. If you do not have enough information, ask clarifying questions."
)


@dataclass
class CareerCounselorChain:
    gemini_client: GeminiClient
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history_limit: int = 10
    _message_histories: Dict[str, List] = field(default_factory=dict, init=False)

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
        inputs = {
            "user_message": user_message,
            "context": context or "No additional context provided.",
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
                        "context": context,
                        "model": self.gemini_client.model_name,
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
