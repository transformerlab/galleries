#!/usr/bin/env python3
"""
Nanochat Training Script with TransformerLab SDK Integration
Runs the nanochat speedrun.sh training pipeline with proper SDK integration
Based on: https://github.com/karpathy/nanochat
"""

import os
import subprocess
import sys
from datetime import datetime

from lab import lab


def run_command(command, description, stream_output=True):
    """Execute a command and log the output"""
    lab.log(f"🔧 {description}")
    try:
        if stream_output:
            # Stream output in real-time for long-running commands
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    lab.log(line.rstrip())
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        else:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                lab.log(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        lab.log(f"❌ Error running command: {command}")
        if hasattr(e, 'stderr') and e.stderr:
            lab.log(f"Error output: {e.stderr}")
        raise


def main():
    """Main training function - runs nanochat speedrun with SDK integration"""
    
    start_time = datetime.now()
    
    try:
        # Initialize TransformerLab SDK
        lab.init()
        lab.log("🚀 Starting Nanochat Training with TransformerLab SDK")
        
        # Prepare training configuration
        training_config = {
            "experiment_name": "nanochat-speedrun",
            "model_name": "nanochat-d20",
            "template_name": "nanochat-speedrun",
            "log_to_wandb": True,
            "_config": {
                "depth": 20
            },
        }
        lab.set_config(training_config)
        lab.log(f"Training started at {start_time}")

    except Exception as e:
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Setup environment
        lab.log("📦 Setting up environment...")
        
        # Use job-specific directory for NANOCHAT_BASE_DIR
        job_data_dir = lab.job.get_dir()
        base_dir = os.path.join(job_data_dir, "nanochat_data")
        os.makedirs(base_dir, exist_ok=True)
        os.environ["NANOCHAT_BASE_DIR"] = base_dir
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # Ensure CUDA libraries are in the library path for PyTorch
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ['LD_LIBRARY_PATH']}"
        else:
            os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
        
        lab.log(f"NANOCHAT_BASE_DIR: {base_dir}")
        
        # Ensure we're in the nanochat directory
        nanochat_dir = os.path.expanduser("~/nanochat")
        if not os.path.exists(nanochat_dir):
            raise RuntimeError("Nanochat directory not found. Setup may have failed.")
        
        os.chdir(nanochat_dir)
        lab.log(f"Working directory: {nanochat_dir}")
        
        # Setup WANDB run name
        # Note: If WANDB_API_KEY is set in environment, wandb will automatically use it
        # Otherwise, speedrun.sh will use "dummy" which skips wandb logging
        wandb_run = f"nanochat-{lab.job.id}"
        os.environ["WANDB_RUN"] = wandb_run
        
        if os.environ.get("WANDB_API_KEY"):
            lab.log(f"🔑 WANDB_API_KEY found - wandb will log with run name: {wandb_run}")
        else:
            lab.log("ℹ️  No WANDB_API_KEY found, training will run without wandb logging")
            lab.log("Set WANDB_API_KEY environment variable to enable wandb logging")
        
        # Deactivate conda if active (to avoid conflicts with uv venv)
        # speedrun.sh will create and activate its own .venv
        if os.environ.get("CONDA_PREFIX"):
            lab.log("⚠️  Conda environment detected, will deactivate before running speedrun.sh")
            # Unset conda variables to avoid conflicts with uv
            conda_vars_to_unset = ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER", 
                                    "CONDA_SHLVL", "CONDA_PYTHON_EXE", "CONDA_EXE"]
            for var in conda_vars_to_unset:
                if var in os.environ:
                    del os.environ[var]
            lab.log("✅ Conda environment variables cleared")
        
        lab.log("🚀 Running nanochat speedrun.sh...")
        lab.log("This will take approximately 4 hours on 8xH100...")
        lab.update_progress(10)
        
        # Run speedrun.sh in a clean shell without conda
        run_command(
            "bash speedrun.sh",
            "Running nanochat speedrun pipeline",
            stream_output=True
        )
                
        # Save artifacts
        lab.log("💾 Saving training artifacts...")
        
        # Save the report
        report_file = os.path.join(nanochat_dir, "report.md")
        if os.path.exists(report_file):
            report_artifact_path = lab.save_artifact(report_file, "nanochat_report.md")
            lab.log(f"✅ Saved training report: {report_artifact_path}")
        
        # Save model checkpoints from base_dir
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            saved_model_path = lab.save_model(checkpoint_dir, name="nanochat_d20_model")
            lab.log(f"✅ Model checkpoints saved: {saved_model_path}")
        
        # Training completed
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"🎉 Training completed in {training_duration}")
        
        # Get wandb URL if available
        try:
            job_data = lab.job.get_job_data()
            captured_wandb_url = job_data.get("wandb_run_url", "Not available")
            lab.log(f"� Wandb URL: {captured_wandb_url}")
        except Exception:
            captured_wandb_url = "Not available"
        
        # Finish wandb run
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                lab.log("✅ Wandb run finished")
        except Exception:
            pass
        
        # Complete the job
        lab.finish("✅ Nanochat training completed successfully!")
        
        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "model": "nanochat-d20",
            "wandb_url": captured_wandb_url,
        }

    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        print("Training interrupted by user")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "error": error_msg}


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n✅ Nanochat Training Result: {result}")        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
