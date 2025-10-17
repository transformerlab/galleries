import os
from datetime import datetime
from time import sleep


from lab import lab


def train():
    """Fake training function that runs locally but reports to TransformerLab"""

    # Training configuration
    training_config = {
        "experiment_name": "alpha",
        "model_name": "HuggingFaceTB/SmolLM-135M-Instruct",
        "dataset": "Trelis/touch-rugby-rules",
        "template_name": "full-demo",
        "output_dir": "./output",
        "log_to_wandb": False,
        "_config": {
            "dataset_name": "Trelis/touch-rugby-rules",
            "lr": 2e-5,
            "num_train_epochs": 1,
            "batch_size": 8,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "max_seq_length": 512,
        },
    }

    try:
        # Initialize lab with default/simple API
        lab.init()
        lab.set_config(training_config)

        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")

        # Create output directory if it doesn't exist
        os.makedirs(training_config["output_dir"], exist_ok=True)

        # Load the dataset
        lab.log("Loading dataset...")
        sleep(0.1)
        lab.log("Loaded dataset")

        # Report initial progress
        lab.job.update_progress(10)

        # Train the model
        lab.log("Starting training...")
        print("Starting training")
        for i in range(8):
            sleep(1)
            lab.log(f"Iteration {i + 1}/8")
            lab.job.update_progress(10 + (i + 1) * 10)
            print(f"Iteration {i + 1}/8")
            
            # Save fake checkpoint every 2 iterations
            if (i + 1) % 2 == 0:
                checkpoint_file = os.path.join(training_config["output_dir"], f"checkpoint_epoch_{i + 1}.txt")
                with open(checkpoint_file, "w") as f:
                    f.write(f"Fake checkpoint for epoch {i + 1}\n")
                    f.write(f"Model state: iteration_{i + 1}\n")
                    f.write(f"Loss: {0.5 - (i + 1) * 0.05:.3f}\n")
                    f.write(f"Accuracy: {0.6 + (i + 1) * 0.04:.3f}\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                
                # Save checkpoint using lab facade
                saved_checkpoint_path = lab.save_checkpoint(checkpoint_file, f"epoch_{i + 1}_checkpoint.txt")
                lab.log(f"Saved checkpoint: {saved_checkpoint_path}")
                
                # Save some fake artifacts
                artifact_file = os.path.join(training_config["output_dir"], f"training_metrics_epoch_{i + 1}.json")
                with open(artifact_file, "w") as f:
                    f.write('{\n')
                    f.write(f'  "epoch": {i + 1},\n')
                    f.write(f'  "loss": {0.5 - (i + 1) * 0.05:.3f},\n')
                    f.write(f'  "accuracy": {0.6 + (i + 1) * 0.04:.3f},\n')
                    f.write(f'  "learning_rate": {2e-5},\n')
                    f.write(f'  "batch_size": {8},\n')
                    f.write(f'  "timestamp": "{datetime.now().isoformat()}"\n')
                    f.write('}\n')
                
                # Save artifact using lab facade
                saved_artifact_path = lab.save_artifact(artifact_file, f"metrics_epoch_{i + 1}.json")
                lab.log(f"Saved artifact: {saved_artifact_path}")

        # Calculate training time
        end_time = datetime.now()
        training_duration = end_time - start_time
        lab.log(f"Training completed in {training_duration}")
        
        # Save final artifacts
        final_model_file = os.path.join(training_config["output_dir"], "final_model_summary.txt")
        with open(final_model_file, "w") as f:
            f.write("Final Model Summary\n")
            f.write("==================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write("Final Loss: 0.15\n")
            f.write("Final Accuracy: 0.92\n")
            f.write(f"Model: {training_config['model_name']}\n")
            f.write(f"Dataset: {training_config['dataset']}\n")
            f.write(f"Completed at: {end_time}\n")
        
        # Save final model as artifact
        final_model_path = lab.save_artifact(final_model_file, "final_model_summary.txt")
        lab.log(f"Saved final model summary: {final_model_path}")
        
        # Save training configuration as artifact
        config_file = os.path.join(training_config["output_dir"], "training_config.json")
        import json
        with open(config_file, "w") as f:
            json.dump(training_config, f, indent=2)
        
        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")
        
        print("Complete")

        # Complete the job in TransformerLab via facade
        lab.finish("Training completed successfully")

        return {
            "status": "success",
            "job_id": lab.job.id,
            "duration": str(training_duration),
            "output_dir": os.path.join(
                training_config["output_dir"], f"final_model_{lab.job.id}"
            ),
        }

    except KeyboardInterrupt:
        lab.error("Stopped by user or remotely")
        return {"status": "stopped", "job_id": lab.job.id}

    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")

        import traceback

        traceback.print_exc()
        lab.error(error_msg)
        return {"status": "error", "job_id": lab.job.id, "error": error_msg}


if __name__ == "__main__":
    result = train()
    print(result)


