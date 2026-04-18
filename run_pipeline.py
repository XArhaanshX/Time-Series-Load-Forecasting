import os
import subprocess
import sys

def run_stage(script_path):
    print(f"\n>>> Running Stage: {os.path.basename(script_path)}")
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {script_path}: {e}")
        sys.exit(1)

def main():
    # Define absolute paths to stages
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    stages = [
        os.path.join(base_dir, 'src', 'data_processing', 'ingest_and_preprocess.py'),
        # Add future stages here
    ]
    
    print("="*50)
    print("HARYANA POWER GRID - LOAD FORECASTING PIPELINE")
    print("="*50)
    
    for stage in stages:
        if os.path.exists(stage):
            run_stage(stage)
        else:
            print(f"Stage script not found: {stage}")
            
    print("\n" + "="*50)
    print("PIPELINE EXECUTION FINISHED")
    print("="*50)

if __name__ == "__main__":
    main()
