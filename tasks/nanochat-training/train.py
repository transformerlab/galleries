#!/usr/bin/env python3
"""
Nanochat Training Script with TransformerLab SDK Integration
Runs the nanochat speedrun training pipeline with proper SDK integration
Based on: https://github.com/karpathy/nanochat/blob/master/speedrun.sh
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from lab import lab


def run_command(command, description, stream_output=True, cwd=None, env=None):
    """Execute a command and log the output"""
    lab.log(f"🔧 {description}")
    try:
        # Merge environment variables
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        
        if stream_output:
            # Stream output in real-time for long-running commands
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=cmd_env
            )
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())  # Print to stdout
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
                text=True,
                cwd=cwd,
                env=cmd_env
            )
            if result.stdout:
                lab.log(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        lab.log(f"❌ Error running command: {command}")
        if hasattr(e, 'stderr') and e.stderr:
            lab.log(f"Error output: {e.stderr}")
        raise


def setup_environment(base_dir, nanochat_dir, nproc):
    """Setup environment variables for training"""
    lab.log("🔧 Phase: Environment Setup")
    
    # Set environment variables
    os.environ["NANOCHAT_BASE_DIR"] = base_dir
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NPROC_PER_NODE"] = str(nproc)
    
    lab.log(f"NANOCHAT_BASE_DIR: {base_dir}")
    lab.log(f"Using {nproc} GPU(s)")
    
    # Install uv if not available
    if subprocess.run("command -v uv", shell=True, capture_output=True).returncode != 0:
        lab.log("📦 Installing uv package manager...")
        run_command(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "Installing uv",
            stream_output=False
        )
    
    # Create venv and install dependencies
    lab.log("🔧 Creating virtual environment...")
    if not os.path.exists(os.path.join(nanochat_dir, ".venv")):
        run_command("uv venv", "Creating venv", cwd=nanochat_dir)
    
    lab.log("📦 Installing dependencies...")
    run_command("uv sync --extra gpu", "Installing dependencies", cwd=nanochat_dir)
    
    # Install transformerlab-sdk in the venv
    lab.log("📦 Installing transformerlab-sdk...")
    run_command("uv pip install transformerlab", "Installing SDK", cwd=nanochat_dir)
    

def train_tokenizer(base_dir, nanochat_dir):
    """Train the tokenizer"""
    lab.log("🔧 Phase: Tokenizer Training")
    
    # Install Rust/Cargo
    lab.log("🦀 Installing Rust/Cargo...")
    cargo_home = os.path.expanduser("~/.cargo")
    if not os.path.exists(cargo_home):
        run_command(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            "Installing Rust",
            stream_output=False
        )
    
    # Source cargo env
    cargo_bin = os.path.join(cargo_home, "bin")
    if cargo_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cargo_bin}:{os.environ.get('PATH', '')}"
    
    # Build rustbpe tokenizer
    lab.log("🔨 Building rustbpe tokenizer...")
    run_command(
        "uv run maturin develop --release --manifest-path rustbpe/Cargo.toml",
        "Building tokenizer",
        cwd=nanochat_dir
    )    
    # Download initial dataset
    lab.log("📥 Downloading initial dataset shards (8)...")
    run_command(
        "uv run python -m nanochat.dataset -n 8",
        "Downloading dataset shards",
        cwd=nanochat_dir
    )
    
    # Start downloading additional shards in background
    lab.log("📥 Starting background download of additional shards (240)...")
    dataset_process = subprocess.Popen(
        "uv run python -m nanochat.dataset -n 240",
        shell=True,
        cwd=nanochat_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    
    # Train tokenizer
    lab.log("🎯 Training tokenizer...")
    run_command(
        "uv run python -m scripts.tok_train --max_chars=2000000000",
        "Training tokenizer",
        cwd=nanochat_dir
    )
    
    # Evaluate tokenizer
    lab.log("📊 Evaluating tokenizer...")
    run_command(
        "uv run python -m scripts.tok_eval",
        "Evaluating tokenizer",
        cwd=nanochat_dir
    )
    
    # Save tokenizer checkpoint
    tokenizer_path = os.path.join(base_dir, "tokenizer.tok")
    if os.path.exists(tokenizer_path):
        lab.save_checkpoint(tokenizer_path, "tokenizer.tok")
        lab.log("✅ Tokenizer checkpoint saved")
    
    return dataset_process


def train_base_model(base_dir, nanochat_dir, dataset_process, nproc, wandb_run):
    """Train the base model"""
    lab.log("🔧 Phase: Base Model Pretraining")
    
    # Wait for dataset download
    lab.log("⏳ Waiting for dataset download to complete...")
    dataset_process.wait()
    lab.log("✅ Dataset download complete")
        
    # Train base model
    lab.log("🎯 Training base d20 model (this will take a while)...")
    nproc_env = {"NPROC_PER_NODE": str(nproc)}
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.base_train --depth=20 --run={wandb_run}",
        "Training base model",
        cwd=nanochat_dir,
        env=nproc_env
    )
        
    # Save base model checkpoint
    model_path = os.path.join(base_dir, "checkpoints", "model_d20.pt")
    if os.path.exists(model_path):
        lab.save_checkpoint(model_path, "base_model_d20.pt")
        lab.log("✅ Base model checkpoint saved")
    
    # Evaluate on train/val
    lab.log("📊 Evaluating base model on train/val data...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.base_loss",
        "Evaluating base model",
        cwd=nanochat_dir
    )
        
    # Evaluate on CORE tasks
    lab.log("📊 Evaluating base model on CORE tasks...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.base_eval",
        "Evaluating on CORE tasks",
        cwd=nanochat_dir
    )
    

def train_midtraining(base_dir, nanochat_dir, nproc, wandb_run):
    """Run midtraining phase"""
    lab.log("🔧 Phase: Midtraining")
    
    # Download identity conversations
    lab.log("📥 Downloading identity conversations...")
    identity_file = os.path.join(base_dir, "identity_conversations.jsonl")
    run_command(
        f"curl -L -o {identity_file} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
        "Downloading conversations",
        stream_output=False
    )
        
    # Run midtraining
    lab.log("🎯 Running midtraining...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.mid_train --run={wandb_run}",
        "Midtraining",
        cwd=nanochat_dir
    )
        
    # Save midtraining checkpoint
    model_path = os.path.join(base_dir, "checkpoints", "model_mid.pt")
    if os.path.exists(model_path):
        lab.save_checkpoint(model_path, "mid_model.pt")
        lab.log("✅ Midtraining checkpoint saved")
    
    # Evaluate
    lab.log("📊 Evaluating midtrained model...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_eval -i mid",
        "Evaluating midtrained model",
        cwd=nanochat_dir
    )
    

def train_sft(base_dir, nanochat_dir, nproc, wandb_run):
    """Run supervised finetuning"""
    lab.log("🔧 Phase: Supervised Finetuning")
    
    # Run SFT
    lab.log("🎯 Running supervised finetuning...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_sft --run={wandb_run}",
        "Supervised finetuning",
        cwd=nanochat_dir
    )
        
    # Save SFT checkpoint
    model_path = os.path.join(base_dir, "checkpoints", "model_sft.pt")
    if os.path.exists(model_path):
        lab.save_checkpoint(model_path, "sft_model.pt")
        lab.log("✅ SFT checkpoint saved")
    
    # Evaluate
    lab.log("📊 Evaluating SFT model...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_eval -i sft",
        "Evaluating SFT model",
        cwd=nanochat_dir
    )
    
    
    # Save final model to Model Zoo
    if os.path.exists(model_path):
        lab.log("💾 Saving final SFT model to Model Zoo...")
        lab.save_model(
            source_path=model_path,
            name="nanochat_d20_sft",
            architecture="gpt2",
            pipeline_tag="text-generation"
        )
        lab.log("✅ SFT model saved to Model Zoo")


def train_rl(base_dir, nanochat_dir, nproc, wandb_run, enable_rl=False):
    """Run reinforcement learning (optional)"""
    if not enable_rl:
        lab.log("⏭️  Skipping RL training (disabled)")
        return
    
    lab.log("🔧 Phase: Reinforcement Learning (GSM8K)")
    
    # Run RL
    lab.log("🎯 Running reinforcement learning on GSM8K...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_rl --run={wandb_run}",
        "Reinforcement learning",
        cwd=nanochat_dir
    )
    
    # Save RL checkpoint
    model_path = os.path.join(base_dir, "checkpoints", "model_rl.pt")
    if os.path.exists(model_path):
        lab.save_checkpoint(model_path, "rl_model.pt")
        lab.log("✅ RL checkpoint saved")
    
    # Evaluate RL model only on GSM8K
    lab.log("📊 Evaluating RL model on GSM8K...")
    run_command(
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m scripts.chat_eval -i rl -a GSM8K",
        "Evaluating RL model",
        cwd=nanochat_dir
    )
    
    # Save RL model to Model Zoo
    if os.path.exists(model_path):
        lab.log("💾 Saving final RL model to Model Zoo...")
        lab.save_model(
            source_path=model_path,
            name="nanochat_d20_rl",
            architecture="gpt2",
            pipeline_tag="text-generation"
        )
        lab.log("✅ RL model saved to Model Zoo")


def generate_report(nanochat_dir):
    """Generate final training report"""
    lab.log("📝 Generating final report...")
    run_command(
        "uv run python -m nanochat.report generate",
        "Generating report",
        cwd=nanochat_dir
    )
        
    # Save report as artifact
    report_path = os.path.join(nanochat_dir, "report.md")
    if os.path.exists(report_path):
        lab.save_artifact(report_path, "nanochat_training_report.md")
        lab.log("✅ Training report saved")


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
                "depth": 20,
                "nproc_per_node": 8,
                "enable_rl": True,  # Set to True to enable RL training on GSM8K
            },
        }
        lab.set_config(training_config)
        lab.log(f"Training started at {start_time}")

    except Exception as e:
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Setup directories
        job_data_dir = lab.job.get_dir()
        base_dir = os.path.join(job_data_dir, "nanochat_data")
        os.makedirs(base_dir, exist_ok=True)
        
        lab.log(f"Job directory: {job_data_dir}")
        lab.log(f"Training data directory: {base_dir}")
        
        # Clone nanochat repository if not exists
        nanochat_dir = os.path.expanduser("~/nanochat")
        if not os.path.exists(nanochat_dir):
            lab.log("📥 Cloning nanochat repository...")
            run_command(
                "git clone https://github.com/karpathy/nanochat.git ~/nanochat",
                "Cloning nanochat repository",
                stream_output=True
            )
        else:
            lab.log(f"✅ Nanochat directory found: {nanochat_dir}")
        
        # Setup WANDB run name
        wandb_run = f"nanochat-{lab.job.id}"
        if os.environ.get("WANDB_API_KEY"):
            lab.log(f"🔑 WANDB_API_KEY found - wandb will log with run name: {wandb_run}")
        else:
            wandb_run = "dummy"  # Skip wandb logging
            lab.log("ℹ️  No WANDB_API_KEY found, training will run without wandb logging")
        
        # Get GPU count
        nproc = training_config.get("_config", {}).get("nproc_per_node", 8)
        
        # Deactivate conda if active (to avoid conflicts with uv venv)
        if os.environ.get("CONDA_PREFIX"):
            lab.log("⚠️  Conda environment detected, clearing conda variables...")
            conda_vars_to_unset = ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER", 
                                    "CONDA_SHLVL", "CONDA_PYTHON_EXE", "CONDA_EXE"]
            for var in conda_vars_to_unset:
                if var in os.environ:
                    del os.environ[var]
            lab.log("✅ Conda environment variables cleared")
        
        # Check if RL is enabled
        enable_rl = training_config.get("_config", {}).get("enable_rl", False)
        
        lab.log("🚀 Starting Nanochat Speedrun Training Pipeline")
        lab.log("This will take approximately 4 hours on 8xH100...")
        if enable_rl:
            lab.log("✅ RL training is ENABLED (will train on GSM8K)")
        else:
            lab.log("⏭️  RL training is DISABLED (set enable_rl=True in config to enable)")
        
        # Initialize report
        run_command(
            "uv run python -m nanochat.report reset",
            "Initializing report",
            cwd=nanochat_dir
        )
        
        # Run all training phases
        setup_environment(base_dir, nanochat_dir, nproc)
        dataset_process = train_tokenizer(base_dir, nanochat_dir)
        train_base_model(base_dir, nanochat_dir, dataset_process, nproc, wandb_run)
        train_midtraining(base_dir, nanochat_dir, nproc, wandb_run)
        train_sft(base_dir, nanochat_dir, nproc, wandb_run)
        train_rl(base_dir, nanochat_dir, nproc, wandb_run, enable_rl)
        generate_report(nanochat_dir)
        
        lab.update_progress(100)
        lab.log("✅ All training phases completed!")
        
        # Additional post-processing and validation
        lab.log("🔍 Validating training artifacts...")
        
        # Check for final model in checkpoints
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        models_found = []
        if os.path.exists(checkpoint_dir):
            model_files = ["model_sft.pt", "model_mid.pt", "model_d20.pt"]
            if enable_rl:
                model_files.append("model_rl.pt")
            for model_file in model_files:
                model_path = os.path.join(checkpoint_dir, model_file)
                if os.path.exists(model_path):
                    models_found.append(model_file)
            lab.log(f"✅ Found model checkpoints: {', '.join(models_found) if models_found else 'none'}")
        
        # Check for report
        report_path = os.path.join(nanochat_dir, "report.md")
        if os.path.exists(report_path):
            lab.log("✅ Training report generated successfully")
        else:
            lab.log("⚠️  Training report not found")
        
        # Training completed
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"🎉 Training completed in {training_duration}")
        
        # Get wandb URL if available
        try:
            job_data = lab.job.get_job_data()
            captured_wandb_url = job_data.get("wandb_run_url", "Not available")
            lab.log(f"📊 Wandb URL: {captured_wandb_url}")
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
        
        lab.finish()

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "model": "nanochat-d20",
            "wandb_url": captured_wandb_url,
            "checkpoints": models_found,
            "base_dir": base_dir,
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
