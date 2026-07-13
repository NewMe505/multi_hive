from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
import hive_prompts
from llm_factory import get_llm

def sprint_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_llm("planner")
    
    # TRC-T4: Filter explicitly for HumanMessage 
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    objective = human_msgs[0].content if human_msgs else ""
    
    sys_prompt = hive_prompts.get_sprint_planner_prompt()
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=objective)])
    return {"sprint_plan": response.content}
