# encoding = "utf-8"

REWARD_MODEL_SYSTEM_PROMPT = """You are a helpful assistant. You should generate responses tailored to the user's preferences based on the given criteria. Each criterion is equally important in shaping the response.

The criteria are:
{criteria}
"""

PAIR_REWARD_MODEL_PROMPT = """# Instruction

You are an experienced judge. Your task is to assess which AI model performs better in the conversation and would be preferred by the user based on the given criteria.
We will provide you with the criteria, the initial user query, the follow-up conversation with two different AI models. You should first read the provided information carefully before making the final the verdict of the user's preference.

# Rules

* Every criterion is equally important in shaping the response.
* Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
* Do not favor certain indexes of the responses. Be as objective as possible.

# Information

## Criteria
{criteria}

## User Query
{user_query}

## Follow-up Conversation with Model A
{model_a}

## Follow-up Conversation with Model B
{model_b}
"""


