import json
from algo_1 import Algorithm1

class NS3PipelineConnector:
    """
    A connector class to interface between the ns-3 simulation (running in WSL)
    and the Python-based Algorithm 1 logic.
    """
    def __init__(self):
        self.algo = Algorithm1()
        self.metrics = {
            "total_steps": 0,
            "handovers_triggered": 0,
            "rlf_risks_high": 0, # How many timesteps RLF risk was >= 0.5
            "ping_pong_risks_high": 0 # Instances of high ping-pong risk
        }

    def reset(self):
        self.algo = Algorithm1()
        self.metrics = {
            "total_steps": 0,
            "handovers_triggered": 0,
            "rlf_risks_high": 0,
            "ping_pong_risks_high": 0
        }

    def process_step(self, state_dict: dict, ttt_eff: int = 160, hys_eff: float = 3.0) -> dict:
        """
        Takes a state dictionary exported from ns-3 and computes the Algorithm 1 step.
        """
        required_keys = [
            "rsrp_serving", "sinr_serving", "cqi_serving", "distance_serving",
            "rsrp_neighbors", "neighbor_ids", "distance_neighbors", 
            "velocity", "now_s"
        ]
        
        # Ensure all keys are available, otherwise provide reasonable defaults or throw error
        for k in required_keys:
            if k not in state_dict:
                raise ValueError(f"Missing required state variable: {k}")

        decision = self.algo.step(
            rsrp_serving=state_dict["rsrp_serving"],
            sinr_serving=state_dict["sinr_serving"],
            cqi_serving=state_dict["cqi_serving"],
            distance_serving=state_dict["distance_serving"],
            rsrp_neighbors=state_dict["rsrp_neighbors"],
            neighbor_ids=state_dict["neighbor_ids"],
            distance_neighbors=state_dict["distance_neighbors"],
            velocity=state_dict["velocity"],
            now_s=state_dict["now_s"],
            TTT_eff=ttt_eff,
            HYS_eff=hys_eff,
            los_probability=state_dict.get("los_probability", 1.0)
        )
        
        self._update_metrics(decision)
        return decision

    def _update_metrics(self, decision: dict):
        self.metrics["total_steps"] += 1
        
        if decision["action"] != 0:
            self.metrics["handovers_triggered"] += 1
            
        if decision["rlf_risk"] >= 0.50:
            self.metrics["rlf_risks_high"] += 1
            
        if decision["ping_pong_risk"] >= 0.70:
            self.metrics["ping_pong_risks_high"] += 1

    def get_summary_metrics(self):
        """
        Returns Phase 1 baseline metrics over the course of the simulation.
        """
        return self.metrics

# Example placeholder for a ZMQ server if you plan to pipe live from ns-3
def run_zmq_server(bind_address="tcp://127.0.0.1:5555"):
    try:
        import zmq
    except ImportError:
        print("Please run `pip install pyzmq` to use the ZMQ server")
        return
        
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(bind_address)
    
    pipeline = NS3PipelineConnector()
    print(f"Algorithm 1 ZMQ Server listening on {bind_address}")
    
    while True:
        message = socket.recv_json()
        
        if message.get("command") == "reset":
            pipeline.reset()
            socket.send_json({"status": "reset_ok"})
            continue
            
        if message.get("command") == "get_metrics":
            socket.send_json(pipeline.get_summary_metrics())
            continue

        try:
            decision = pipeline.process_step(
                state_dict=message["state"], 
                ttt_eff=message.get("ttt", 160), 
                hys_eff=message.get("hys", 3.0)
            )
            socket.send_json(decision)
        except Exception as e:
            socket.send_json({"error": str(e)})

if __name__ == "__main__":
    # Standard entry point if run directly
    print("NS3 Pipeline Connector for Algorithm 1 loaded.")
    # uncomment below to run server directly
    # run_zmq_server()
