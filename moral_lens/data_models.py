from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"

    @classmethod
    def choices(cls):
        return [messagerole.value for messagerole in cls]

@dataclass
class ChatMessage:
    """
    Represents a chat message to be sent to a language model.
    """
    role: MessageRole
    content: str

    def format(self) -> Dict[str, str]:
        """
        Returns the template format for the message.
        """
        return {"role": self.role.value, "content": self.content}

@dataclass
class Prompt:
    """
    Represents a prompt consisting of a list of chat messages.

    Example:
        prompt = Prompt(messages=[
            ChatMessage(role=MessageRole.system, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.user, content="What is the capital of France?"),
        ])
    """
    messages: list[ChatMessage]

    def __getitem__(self, index):
        """
        Makes the prompt subscriptable to access messages by index.

        Example:
            prompt = Prompt(messages=[...])
            first_message = prompt[0]
        """
        return self.messages[index]

    def get_role_message(self, role: MessageRole | str) -> ChatMessage | None:
        """
        Returns the first message with the specified role.

        Args:
            role (MessageRole): The role of the message to retrieve.

        Returns:
            ChatMessage: The first message with the specified role, or None if not found.
        """
        for message in self.messages:
            if str(message.role) == str(role):
                return message
        return None

    def get_system_message(self) -> ChatMessage | None:
        """Returns the first message with the system role."""
        return self.get_role_message(MessageRole.system)

    def get_user_message(self) -> ChatMessage | None:
        """Returns the first message with the user role."""
        return self.get_role_message(MessageRole.user)

    def openai_format(self) -> List[Dict[str, str]]:
        """
        Formats the prompt for use in OpenAIModel (and OpenRouterModel).

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        """
        return [message.format() for message in self.messages]

    def anthropic_format(self):
        """
        Formats the prompt for use in AnthropicModel.

        "You are a helpful assistant.", [{"role": "user", "content": "What is the capital of France?"}]
        """
        system_prompt = self.messages[0]
        message = self.messages[1]

        assert system_prompt.role == MessageRole.system
        assert message.role == MessageRole.user

        return system_prompt.content, [message.format()]

    def gemini_format(self):
        """
        Formats the prompt for use in GeminiModel.

        "You are a helpful assistant.", "What is the capital of France?"
        """
        system_prompt = self.messages[0]
        message = self.messages[1]

        assert system_prompt.role == MessageRole.system
        assert message.role == MessageRole.user

        return system_prompt.content, message.content

    def __len__(self):
        return len(self.messages)


class Provider(str, Enum):
    """
    Supported providers for language models.
    Includes: openai, anthropic, gemini, openrouter, huggingface.
    """
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"
    openrouter = "openrouter"
    huggingface = "huggingface"

    @classmethod
    def choices(cls):
        return [provider.value for provider in cls]


@dataclass
class LLMResponse:
    """
    Represents a response from a language model.
    """
    model_id: str
    completion: str
    content: str
    thinking_content: str = ""

    decision: str = ""
    reasoning: str = ""

    attempts: int = 0

    two_choices: List[str] = None

    def is_valid(self) -> bool:
        """
        Check if the response is valid based on the presence of completion or content.
        """
        if self.completion is None or self.content is None or self.content.strip() == "":
            return False
        return True