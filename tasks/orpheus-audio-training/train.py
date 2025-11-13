#!/usr/bin/env python3
"""
Orpheus TTS (Text-to-Speech) Fine-tuning Script
Fine-tunes Orpheus audio models using Unsloth with TransformerLab SDK integration
"""
from unsloth import FastLanguageModel
import torch
from transformers import TrainerCallback, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from snac import SNAC
from unsloth import is_bfloat16_supported

import argparse
import os
import json
from datetime import datetime
from abc import ABC


from lab import lab

class OrpheusAudioTrainer(ABC):
    def __init__(
        self,
        model_name,
        context_length,
        device,
        speaker_key,
        lora_r,
        lora_alpha,
        lora_dropout,
        sampling_rate,
        max_audio_length,
        batch_size,
        audio_column_name="audio",
        text_column_name="text",
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.device = device
        self.speaker_key = speaker_key
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.audio_column_name = audio_column_name
        self.text_column_name = text_column_name
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Load main model first
        self.model, self.processor = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.context_length,
            dtype=None,
            load_in_4bit=False,
        )
        
        # Load SNAC model to device (will share GPU with main model)
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lab.log(f"Trainable parameters: {num_trainable}")

        # Define special tokens
        self.tokenizer_length = 128256
        self.start_of_text = 128000
        self.end_of_text = 128009
        self.start_of_speech = self.tokenizer_length + 1
        self.end_of_speech = self.tokenizer_length + 2
        self.start_of_human = self.tokenizer_length + 3
        self.end_of_human = self.tokenizer_length + 4
        self.start_of_ai = self.tokenizer_length + 5
        self.end_of_ai = self.tokenizer_length + 6
        self.audio_tokens_start = self.tokenizer_length + 10
        self.pad_token = self.tokenizer_length + 7
        self.ds_sample_rate = 24000
        self.batch_size = batch_size

    def _tokenize_audio(self, waveform):
        """Convert audio waveform to SNAC tokens."""
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(dtype=torch.float32)

        waveform = waveform.unsqueeze(0).unsqueeze(0).to(self.device)

        # Generate SNAC codes
        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Interleave codes according to Orpheus format
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.extend(
                [
                    codes[0][0][i].item() + 128266,
                    codes[1][0][2 * i].item() + 128266 + 4096,
                    codes[2][0][4 * i].item() + 128266 + (2 * 4096),
                    codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096),
                    codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096),
                    codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096),
                    codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096),
                ]
            )

        # Clear GPU memory after encoding
        del waveform, codes
        torch.cuda.empty_cache()
        
        return all_codes

    def _remove_duplicate_frames(self, codes_list):
        """Remove consecutive duplicate audio frames to reduce redundancy."""
        if len(codes_list) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")

        result = codes_list[:7]

        for i in range(7, len(codes_list), 7):
            current_first = codes_list[i]
            previous_first = result[-7]

            if current_first != previous_first:
                result.extend(codes_list[i : i + 7])

        return result

    def preprocess_dataset(self, example):
        """
        Preprocess a single example for Orpheus training.
        """
        try:
            # Extract and tokenize audio
            audio_array = example[self.audio_column_name]["array"]
            codes_list = self._tokenize_audio(audio_array)

            if not codes_list:
                print(f"Warning: Empty codes list for example with text '{example[self.text_column_name][:50]}...'")
                return None

            # Remove duplicate frames for efficiency
            codes_list = self._remove_duplicate_frames(codes_list)

            # Create text prompt (multi-speaker or single-speaker)
            if self.speaker_key in example and example[self.speaker_key]:
                text_prompt = f"{example[self.speaker_key]}: {example[self.text_column_name]}"
            else:
                text_prompt = example[self.text_column_name]

            text_ids = self.processor.encode(text_prompt, add_special_tokens=True)
            text_ids.append(self.end_of_text)

            # Construct input sequence with special tokens
            input_ids = (
                [self.start_of_human]
                + text_ids
                + [self.end_of_human]
                + [self.start_of_ai]
                + [self.start_of_speech]
                + codes_list
                + [self.end_of_speech]
                + [self.end_of_ai]
            )

            # Use tokenizer to handle truncation and padding properly
            if len(input_ids) > self.context_length:
                input_ids = input_ids[: self.context_length]

            labels = input_ids.copy()

            # Pad to context_length using the pad_token
            padding_length = self.context_length - len(input_ids)
            if padding_length > 0 and self.batch_size > 1:
                input_ids.extend([self.pad_token] * padding_length)
                labels.extend([-100] * padding_length)
                attention_mask = [1] * (len(input_ids) - padding_length) + [0] * padding_length
            else:
                attention_mask = [1] * len(input_ids)

            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        except Exception as e:
            print(f"Error processing example with text '{example[self.text_column_name][:50]}...': {e}")
            return None


class TransformerLabCallback(TrainerCallback):
    """Callback to integrate with TransformerLab SDK for progress tracking and logging"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.training_started = False
    
    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        """Called when training begins"""
        lab.log("🚀 Training started with Unsloth Trainer")
        self.training_started = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics"""
        if logs:
            # Log key metrics to TransformerLab
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    lab.log(f"{key}: {value:.4f}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        self.current_step = state.global_step
        # Update progress (0-95%)
        progress = min(int((self.current_step / self.total_steps) * 95), 95)
        lab.update_progress(progress)

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        if state.global_step > 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                # Use SDK to save checkpoint
                lab.save_checkpoint(checkpoint_dir, name=f"checkpoint-{state.global_step}")
                lab.log(f"💾 Checkpoint saved at step {state.global_step}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if state.epoch:
            lab.log(f"📊 Completed epoch {int(state.epoch)} / {args.num_train_epochs}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        lab.log("✅ Training completed successfully")


def main():
    """Main training function with TransformerLab SDK integration"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fine-tune Orpheus TTS model on custom voice dataset")
    parser.add_argument(
        "--model_id",
        type=str,
        default="unsloth/orpheus-3b-0.1-ft",
        help="The model ID to use for training (default: unsloth/orpheus-3b-0.1-ft)")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="MrDragonFox/Elise",
        help="The dataset ID to use for training (default: MrDragonFox/Elise)")
    parser.add_argument(
        "--max_dataset_samples",
        type=int,
        default=None,
        help="Maximum number of dataset samples to use (default: None, use full dataset)")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device (default: 1)")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoints every N steps (default: 100)")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs if set)")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = None
    try:
        # Initialize TransformerLab SDK
        lab.init()
        lab.log("TransformerLab SDK initialized")
        
        # Configure GPU usage - use only GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Prepare training configuration for TransformerLab
        training_config = {
            "experiment_name": "orpheus-audio-training",
            "model_name": args.model_id,
            "dataset": args.dataset_id,
            "template_name": "orpheus-tts-training",
            "log_to_wandb": True,
            "output_dir": args.output_dir,
            "_config": {
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_seq_length": args.max_seq_length,
                "warmup_steps": args.warmup_steps,
                "save_steps": args.save_steps,
                "max_steps": args.max_steps,
            },
        }
        lab.set_config(training_config)
        
        # Check if we should resume from a checkpoint
        # checkpoint = lab.get_checkpoint_to_resume()
        if checkpoint:
            lab.log(f"📁 Resuming training from checkpoint: {checkpoint}")
        
        # Log start time
        start_time = datetime.now()
        lab.log(f"Training started at {start_time}")
        lab.log(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All available')}")
        
        os.makedirs(args.output_dir, exist_ok=True)

    except Exception as e:
        # If SDK initialization fails, continue without it
        print(f"Warning: TransformerLab SDK initialization failed: {e}")
        start_time = datetime.now()
        training_config = {}

    try:
        # Load the dataset
        dataset = load_dataset(args.dataset_id, split="train")
        lab.log(f"Loaded dataset with {len(dataset)} samples")
        
        if args.max_dataset_samples is not None and len(dataset) > args.max_dataset_samples:
            dataset = dataset.select(range(args.max_dataset_samples))
            lab.log(f"⚠️ Limited dataset to {args.max_dataset_samples} samples to save memory")
        
        # Ensure all audio is at 24 kHz sampling rate
        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
            lab.log("✅ Dataset audio resampled to 24kHz")

        max_audio_length = max(len(example["audio"]["array"]) for example in dataset)
        
    except Exception as e:
        lab.log(f"❌ Error loading dataset: {e}")
        lab.finish("Training failed - dataset loading error")
        raise
    
    try:
        model_trainer = OrpheusAudioTrainer(
            model_name=args.model_id,
            speaker_key="source",
            context_length=args.max_seq_length,
            device=device,
            lora_r=64,
            lora_alpha=64,
            lora_dropout=0,
            sampling_rate=24000,
            max_audio_length=max_audio_length,
            batch_size=args.per_device_train_batch_size,
            audio_column_name="audio",
            text_column_name="text",
        )

        processed_ds = dataset.map(
        model_trainer.preprocess_dataset,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset",
        )
        processed_ds = processed_ds.filter(lambda x: x is not None)

        lab.log(f"Processed dataset length: {len(processed_ds)}")

        
    except Exception as e:
        lab.log(f"❌ Error loading model: {e}")
        lab.finish("Training failed - model loading error")
        raise
    
    try:
        
        # Calculate total training steps for progress tracking
        steps_per_epoch = len(processed_ds) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        if args.max_steps is not None:
            total_steps = args.max_steps
        else:
            total_steps = steps_per_epoch * args.num_train_epochs
        
        lab.log(f"Total training steps: {total_steps}")
        
        # Create TransformerLab callback for progress tracking
        tlab_callback = TransformerLabCallback(total_steps)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to=["wandb"],
            run_name=f"orpheus-audio-{lab.job.id}",
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            resume_from_checkpoint=checkpoint if checkpoint else None,
        )
        
        # Override with max_steps if provided
        if args.max_steps is not None:
            training_args.max_steps = args.max_steps
                
    except Exception as e:
        lab.log(f"❌ Error setting up training arguments: {e}")
        lab.finish("Training failed - training arguments error")
        raise

    # Create trainer
    lab.log("Creating trainer...")
    
    try:
        trainer = Trainer(
            model=model_trainer.model,
            train_dataset=processed_ds,
            args=training_args,
            callbacks=[tlab_callback],
        )
        
        lab.log("✅ Trainer created successfully")
        
    except Exception as e:
        lab.log(f"❌ Error creating trainer: {e}")
        lab.finish("Training failed - trainer creation error")
        raise

    # Start training
    lab.log("🚀 Starting training...")
    
    try:
        trainer.train(resume_from_checkpoint=checkpoint if checkpoint else None)        
    except Exception as e:
        lab.log(f"❌ Error during training: {e}")
        lab.finish("Training failed - training error")
        raise

    # Training completed
    end_time = datetime.now()
    training_duration = end_time - start_time
    lab.log(f"Training completed in {training_duration}")
    
    # Save training artifacts
    lab.log("Saving training artifacts...")
    
    try:
        # Save training configuration as artifact
        config_file = os.path.join(args.output_dir, "training_config.json")
        with open(config_file, "w") as f:
            json.dump(training_config, f, indent=2)
        config_artifact_path = lab.save_artifact(config_file, "training_config.json")
        lab.log(f"Saved training config: {config_artifact_path}")
        
        # Save training progress summary
        progress_file = os.path.join(args.output_dir, "training_progress_summary.json")
        with open(progress_file, "w") as f:
            json.dump({
                "training_type": "Orpheus TTS Fine-tuning with Unsloth",
                "total_epochs": args.num_train_epochs,
                "total_steps": total_steps,
                "final_step": trainer.state.global_step,
                "model_name": args.model_id,
                "dataset": args.dataset_id,
                "completed_at": end_time.isoformat(),
                "duration": str(training_duration),
            }, f, indent=2)
        progress_artifact_path = lab.save_artifact(progress_file, "training_progress_summary.json")
        lab.log(f"Saved training progress: {progress_artifact_path}")
        
        # Save final model summary
        final_model_file = os.path.join(args.output_dir, "final_model_summary.txt")
        with open(final_model_file, "w") as f:
            f.write("Final Orpheus TTS Model Summary\n")
            f.write("================================\n")
            f.write(f"Training Duration: {training_duration}\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Dataset: {args.dataset_id}\n")
            f.write(f"Completed at: {end_time}\n")
            f.write(f"Total steps: {trainer.state.global_step}\n")
            f.write(f"Max sequence length: {args.max_seq_length}\n")
        final_model_path = lab.save_artifact(final_model_file, "final_model_summary.txt")
        lab.log(f"Saved final model summary: {final_model_path}")
        
    except Exception as e:
        lab.log(f"⚠️ Warning: Error saving artifacts: {e}")
    
    # Save LoRA model
    lab.log("Saving LoRA model...")
    
    try:
        lora_output_dir = os.path.join(args.output_dir, "lora_model")
        os.makedirs(lora_output_dir, exist_ok=True)
        
        model_trainer.model.save_pretrained(lora_output_dir)
        model_trainer.processor.save_pretrained(lora_output_dir)
        
        lab.log(f"✅ LoRA model saved to: {lora_output_dir}")
        
        # Save using SDK for tracking
        saved_path = lab.save_model(lora_output_dir, name="orpheus_lora_model")
        lab.log(f"✅ Model saved to job models directory: {saved_path}")
        
    except Exception as e:
        lab.log(f"❌ Error saving model: {e}")
        lab.finish("Training failed - model saving error")
        raise
    
    # Get the captured wandb URL from job data for reporting
    try:
        job_data = lab.job.get_job_data()
        captured_wandb_url = job_data.get("wandb_run_url", "None")
        lab.log(f"📋 Final wandb URL stored in job data: {captured_wandb_url}")
    except Exception:
        captured_wandb_url = "None"
    
    # Finish wandb run if it was initialized
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            lab.log("✅ Wandb run finished")
    except Exception:
        pass
    
    # Store model metadata in job data
    try:
        lab.job.update_job_data_field("final_model_path", lora_output_dir)
    except Exception:
        pass
    
    # Complete the job
    lab.finish("Training completed successfully")
    lab.update_progress(100)
    
    return {
        "status": "success",
        "job_id": lab.job.id if hasattr(lab.job, 'id') else None,
        "duration": str(training_duration),
        "output_dir": args.output_dir,
        "saved_model_path": saved_path,
        "wandb_url": captured_wandb_url,
        "trainer_type": "Unsloth Trainer",
    }


if __name__ == "__main__":
    try:
        result = main()
        print(f"\nTraining Result: {result}")
    except KeyboardInterrupt:
        lab.error("Training stopped by user or remotely")
        print("Training interrupted by user")
    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        import traceback
        traceback.print_exc()
        lab.error(error_msg)
