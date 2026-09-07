import logging
import sys
from typing import Optional, Dict, Any

from app.domain.models import AgentEvent, AgentEventType

def setup_logger():
    logger = logging.getLogger("voyage")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

voyage_logger = setup_logger()

def log_agent_event(
    event_type: AgentEventType, 
    message: str, 
    node: Optional[str] = None, 
    tool: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Logs a structured Voyage event. This leverages our Pydantic AgentEvent model.
    """
    event = AgentEvent(
        event_type=event_type,
        message=message,
        node=node,
        tool=tool,
        metadata=metadata or {}
    )
    # Output the structured JSON string
    voyage_logger.info(event.model_dump_json())
