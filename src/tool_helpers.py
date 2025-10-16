
# math tools
from langchain.tools import tool
import math
from typing import (List, Annotated,Sequence,TypedDict,Any,Optional,Dict,Union,Literal)
from prompt_helpers import model_call_output

@tool("add", return_direct=True)
def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


@tool("subtract", return_direct=True)
def subtract(a: float, b: float) -> float:
    """Return the result of subtracting b from a."""
    return a - b


@tool("multiply", return_direct=True)
def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


@tool("divide", return_direct=True)
def divide(a: float, b: float) -> float:
    """Return the result of dividing a by b. Raises an error if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool("sqrt", return_direct=True)
def sqrt(x: float) -> float:
    """Return the square root of a number. Raises an error if the input is negative."""
    if x < 0:
        raise ValueError("Cannot take square root of a negative number.")
    return math.sqrt(x)


@tool("power", return_direct=True)
def power(a: float, b: float) -> float:
    """Return a raised to the power of b."""
    return math.pow(a, b)

@tool(args_schema=model_call_output)
def model_call_tool(Thought: List[str], References:Optional[List[str]] = None, Final_answer: Optional[str] = None) :
    """Output the model's thought process, references and the final answer."""
    return {"Thought": Thought, "References": References or [],"Final_answer": Final_answer or ""}


# output shcema tools

tools=[add,subtract,multiply,divide,sqrt,power]

tools_by_name={ tool.name:tool for tool in tools}

tool_names=[tool.name for tool in tools]