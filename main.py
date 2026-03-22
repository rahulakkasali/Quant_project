import os
import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    if os.path.exists(dashboard_path):
        logger.info("Starting Streamlit Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    else:
        logger.error(f"Dashboard not found at {dashboard_path}")

def display_help():
    print("Intelligent Trading System Orchestrator")
    print("Usage: python main.py [command]")
    print("Commands:")
    print("  dashboard    Run the Streamlit visualization dashboard")
    print("  live         Launch the real-time continuous trading execution loop")
    print("  help         Show this help message")
    print("\nNote: Individual modules (data, models, rl, execution) can be run independently for training and testing.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "dashboard":
            run_dashboard()
        elif command == "live":
            from execution.live_trader import run_live_trader
            logger.info("Starting Live Trader...")
            run_live_trader()
        else:
            display_help()
    else:
        display_help()
