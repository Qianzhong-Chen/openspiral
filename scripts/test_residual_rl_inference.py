"""Quick test that the residual RL policy server responds correctly."""

import numpy as np
from openpi_client import websocket_client_policy


def main():
    # Connect to the serving policy
    client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8111)
    print(f"Connected. Server metadata: {client.get_server_metadata()}")

    # Create dummy observation matching YAM robot format:
    # 3 cameras (H, W, 3) uint8 + state (14,) float32
    obs = {
        "state": np.random.randn(14).astype(np.float32),
        "left_camera-images-rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "right_camera-images-rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "top_camera-images-rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }

    # Run inference
    print("Running inference...")
    result = client.infer(obs)

    actions = result["actions"]
    print(f"Actions shape: {actions.shape}")  # expect (action_horizon, action_dim)
    print(f"Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"Timing: {result.get('policy_timing', 'N/A')}")

    # Run a few more to check consistency and speed
    import time
    times = []
    for i in range(20):
        t0 = time.time()
        result = client.infer(obs)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Step {i+1}: {elapsed*1000:.0f}ms, actions norm={np.linalg.norm(result['actions']):.4f}")

    print(f"Average inference time: {np.mean(times)*1000:.0f}ms (excl. first call)")
    print("Test passed!")


if __name__ == "__main__":
    main()
