import argparse
from app.tui.app import VoyageTUI

def main():
    parser = argparse.ArgumentParser(description="Project Voyage TUI")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API calls")
    parser.add_argument("--replay", type=str, help="Path to a recorded run JSON file to replay")
    
    args = parser.parse_args()
    
    app = VoyageTUI(demo_mode=args.demo, replay_file=args.replay)
    app.run()

if __name__ == "__main__":
    main()
