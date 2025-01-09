import time
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LatencyMetrics:
    request_id: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    success: bool
    error_message: str = None

class BedrockLatencyTracker:
    def __init__(self):
        self.metrics: List[LatencyMetrics] = []
    
    def measure_latency(self, func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                response = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)
                response = None
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            metric = LatencyMetrics(
                request_id=f"req_{int(time.time()*1000)}",
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=duration_ms,
                success=success,
                error_message=error_msg
            )
            self.metrics.append(metric)
            return response
        return wrapper

    def get_statistics(self) -> Dict:
        if not self.metrics:
            return {}
            
        successful_latencies = [m.total_duration_ms for m in self.metrics if m.success]
        
        return {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful_latencies),
            "failed_requests": len(self.metrics) - len(successful_latencies),
            "avg_latency_ms": sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0,
            "min_latency_ms": min(successful_latencies) if successful_latencies else 0,
            "max_latency_ms": max(successful_latencies) if successful_latencies else 0
        }



# # example usage
# # Initialize tracker
# latency_tracker = BedrockLatencyTracker()

# # Decorate the Bedrock API calls
# @latency_tracker.measure_latency
# def invoke_bedrock_agent(prompt: str):
#     response = bedrock_agent_runtime_client.invoke_agent(
#         agentId='YOUR_AGENT_ID',
#         agentAliasId='YOUR_ALIAS_ID',
#         input={
#             'text': prompt
#         }
#     )
#     return response

# # Example usage
# response = invoke_bedrock_agent("What are the total sales?")
# stats = latency_tracker.get_statistics()
# print(f"Latency Statistics: {json.dumps(stats, indent=2)}")


# @metrics_tracker.measure_latency
# def query_agent(agent_name, user_query, kb_id):
#     return agents.invoke(
#         agent_name=agent_name, 
#         input_text=user_query, 
#         verbose=True, 
#         enable_trace=True,
#         kb_id=kb_id,
#         num_results=5,
#         # tra