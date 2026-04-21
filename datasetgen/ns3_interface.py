import json
from phase_1.algo_1 import Algorithm1


class NS3PipelineConnector:
    """
    Connector between ns-3 exported state and Python Algorithm1 logic.
    """

    def __init__(self):
        self.algo = Algorithm1()
        self.metrics = {
            "total_steps": 0,
            "handovers_triggered": 0,
            "rlf_risks_high": 0,       # rlf_risk >= 0.5 count
            "ping_pong_risks_high": 0  # ping_pong_risk >= 0.7 count
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
        Process one state tick from ns-3 and return Algorithm1 decision.
        """
        required_keys = [
            "rsrp_serving",
            "sinr_serving",
            "cqi_serving",
            "distance_serving",
            "rsrp_neighbors",
            "neighbor_ids",
            "distance_neighbors",
            "velocity",
            "now_s",
        ]

        for k in required_keys:
            if k not in state_dict:
                raise ValueError(f"Missing required state variable: {k}")

        decision = self.algo.step(
            rsrp_serving=float(state_dict["rsrp_serving"]),
            sinr_serving=float(state_dict["sinr_serving"]),
            cqi_serving=int(state_dict["cqi_serving"]),
            distance_serving=float(state_dict["distance_serving"]),
            rsrp_neighbors=list(state_dict["rsrp_neighbors"]),
            neighbor_ids=list(state_dict["neighbor_ids"]),
            distance_neighbors=list(state_dict["distance_neighbors"]),
            velocity=float(state_dict["velocity"]),
            now_s=float(state_dict["now_s"]),
            TTT_eff=int(ttt_eff),
            HYS_eff=float(hys_eff),
        )

        self._update_metrics(decision)
        return decision

    def _update_metrics(self, decision: dict):
        self.metrics["total_steps"] += 1

        if int(decision.get("action", 0)) != 0:
            self.metrics["handovers_triggered"] += 1

        if float(decision.get("rlf_risk", 0.0)) >= 0.50:
            self.metrics["rlf_risks_high"] += 1

        if float(decision.get("ping_pong_risk", 0.0)) >= 0.70:
            self.metrics["ping_pong_risks_high"] += 1

    def get_summary_metrics(self):
        return self.metrics


def run_zmq_server(bind_address="tcp://127.0.0.1:5555"):
    """
    Optional ZMQ server wrapper for live ns-3 integration.
    """
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
                hys_eff=message.get("hys", 3.0),
            )
            socket.send_json(decision)
        except Exception as e:
            socket.send_json({"error": str(e)})


if __name__ == "__main__":
    print("NS3 Pipeline Connector for Algorithm 1 loaded.")