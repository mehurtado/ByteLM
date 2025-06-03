import unittest
from unittest.mock import MagicMock, patch, call
import torch
from pathlib import Path
import sys
import shutil 

if '..' not in sys.path:
    sys.path.insert(0, '..')

from trainer import Trainer 
from config import CONFIG_V3 


class MinimalMockDataLoader:
    def __init__(self, num_batches=10, batch_data=None):
        self.num_batches = num_batches
        if batch_data is None:
            self.batch_data = (torch.randn(4, 16), torch.randn(4, 16)) 
        else:
            self.batch_data = batch_data
        self.current_batch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            return self.batch_data
        else:
            raise StopIteration

class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock(spec=torch.nn.Module)
        self.mock_model.config = {"model_specific_param": 123, "sequence_length": 16} 
        self.mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
        self.mock_model.to = MagicMock(return_value=self.mock_model) # Ensure model.to(device) returns the model

        self.mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.mock_optimizer.param_groups = [{'lr': 0.001}] 

        self.mock_criterion = MagicMock(spec=torch.nn.Module)
        self.mock_device = torch.device("cpu")
        
        self.checkpoint_dir = Path("./test_checkpoints_trainer") 
        self.model_name = "test_model"

        self.base_train_config = CONFIG_V3.copy()
        self.base_train_config.update({
            "num_epochs": 1,
            "print_every": 500, 
            "test_every_batches": 0, 
            "checkpoint_every_batches": 0, 
            "validate_every_batches": 0, 
            "use_amp": False, 
            "use_lr_warmup": False, 
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_name": self.model_name,
            "clip_grad_norm_value": 0.0, 
            "vocab_size": 256, 
            "sequence_length": 16,
            "embedding_dim": 32,
            "attention_d_model": 32,
            "num_attention_layers": 1,
            "attention_num_heads": 2,
        })
        
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._reinitialize_trainer(self.base_train_config.copy())

    def _reinitialize_trainer(self, config_override):
        current_config = self.base_train_config.copy()
        current_config.update(config_override)
        
        # Ensure dataloaders are re-created if num_batches changes
        train_dl_batches = current_config.get("simulated_num_batches", 10)
        val_dl_needed = current_config.get("validate_every_batches",0) > 0 or \
                        current_config.get("run_end_of_epoch_eval", False) or \
                        current_config.get("test_loading_val_dl_present", False) # For specific tests

        self.trainer = Trainer(
            model=self.mock_model,
            train_dataloader=MinimalMockDataLoader(num_batches=train_dl_batches),
            val_dataloader=MinimalMockDataLoader(num_batches=5) if val_dl_needed else None,
            optimizer=self.mock_optimizer,
            criterion=self.mock_criterion,
            device=self.mock_device,
            checkpoint_dir=str(self.checkpoint_dir),
            model_name=self.model_name,
            scheduler=None, 
            train_config=current_config
        )
        self.trainer.best_val_loss = float('inf')


    def tearDown(self):
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir) 
            
    @patch('trainer.Trainer._run_training_step') 
    @patch('trainer.Trainer.save_checkpoint') 
    def test_intermittent_checkpointing_and_naming(self, mock_save_checkpoint, mock_run_step):
        mock_run_step.return_value = 0.1 

        config_update = {
            "checkpoint_every_batches": 3,
            "simulated_num_batches": 10 
        }
        self._reinitialize_trainer(config_update)
        
        self.trainer.train_epoch(epoch_num=0) 
        
        self.assertEqual(mock_save_checkpoint.call_count, 3) 
        
        expected_calls = [
            call(epoch=0, val_loss=mock_run_step.return_value, is_best=False, 
                 custom_filename=f"{self.model_name}_epoch_1_batch_3_step_3.pth"),
            call(epoch=0, val_loss=mock_run_step.return_value, is_best=False, 
                 custom_filename=f"{self.model_name}_epoch_1_batch_6_step_6.pth"),
            call(epoch=0, val_loss=mock_run_step.return_value, is_best=False, 
                 custom_filename=f"{self.model_name}_epoch_1_batch_9_step_9.pth"),
        ]
        mock_save_checkpoint.assert_has_calls(expected_calls, any_order=False)
        # Test 5 part: check is_best=False and custom_filename is not None
        for c in mock_save_checkpoint.mock_calls:
            self.assertFalse(c.kwargs['is_best'])
            self.assertIsNotNone(c.kwargs['custom_filename'])


    @patch('trainer.Trainer._run_training_step')
    @patch('trainer.Trainer.evaluate_epoch')
    def test_intermittent_validation(self, mock_evaluate_epoch, mock_run_step):
        mock_run_step.return_value = 0.1
        mock_evaluate_epoch.return_value = 0.5 

        config_update = {
            "validate_every_batches": 2,
            "simulated_num_batches": 5,
        }
        self._reinitialize_trainer(config_update)
        
        self.trainer.train_epoch(epoch_num=0)
        
        self.assertEqual(mock_evaluate_epoch.call_count, 2)
        mock_evaluate_epoch.assert_has_calls([call(epoch_num=0), call(epoch_num=0)], any_order=False)

    @patch('trainer.Trainer._run_training_step')
    @patch('trainer.Trainer.save_checkpoint')
    @patch('trainer.Trainer.evaluate_epoch')
    def test_best_model_update_intermittent_val_and_naming(self, mock_evaluate_epoch, mock_save_checkpoint, mock_run_step):
        mock_run_step.return_value = 0.1 
        mock_evaluate_epoch.side_effect = [0.5, 0.3, 0.4, 0.2] 

        config_update = {
            "validate_every_batches": 2, 
            "simulated_num_batches": 8 
        }
        self._reinitialize_trainer(config_update)
        self.trainer.best_val_loss = float('inf') 

        self.trainer.train_epoch(epoch_num=0)
        
        self.assertEqual(mock_evaluate_epoch.call_count, 4)
        self.assertAlmostEqual(self.trainer.best_val_loss, 0.2)
        
        # Test 5 part: Check save_checkpoint calls for best model
        # Expected calls with is_best=True
        # Call 1: val_loss=0.5, is_best=True
        # Call 2: val_loss=0.3, is_best=True
        # Call 3: val_loss=0.2, is_best=True
        
        best_model_save_calls = [c for c in mock_save_checkpoint.mock_calls if c.kwargs.get('is_best')]
        self.assertEqual(len(best_model_save_calls), 3)

        expected_best_val_losses = [0.5, 0.3, 0.2]
        for i, actual_call in enumerate(best_model_save_calls):
            self.assertTrue(actual_call.kwargs['is_best'])
            self.assertAlmostEqual(actual_call.kwargs['val_loss'], expected_best_val_losses[i])
            # Default filename for best model is model_name_best.pth
            self.assertEqual(Path(self.checkpoint_dir / f"{self.model_name}_best.pth").name, Path(actual_call.args[0] if actual_call.args else actual_call.kwargs.get('filepath', "")).name)


    @patch('torch.load')
    @patch('pathlib.Path.is_file')
    @patch('trainer.Trainer.train_epoch') # Mock out full epoch training
    @patch('trainer.Trainer.evaluate_epoch') # Mock out end of epoch eval
    def test_loading_best_val_loss_from_checkpoint(self, mock_eval_epoch, mock_train_epoch, mock_path_is_file, mock_torch_load):
        mock_path_is_file.return_value = True # Simulate checkpoint file exists
        
        # Case 1: 'best_val_loss' key exists
        dummy_checkpoint_1 = {
            'epoch': 0,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': 0.3, # Epoch loss
            'best_val_loss': 0.25, # Actual best loss
            'current_global_step': 100
        }
        mock_torch_load.return_value = dummy_checkpoint_1
        config_update = {
            "resume_from_checkpoint": str(self.checkpoint_dir / "dummy_ckpt.pth"),
            "reset_best_val_loss_on_resume": False,
            "num_epochs": 1, # To run the train loop once
            "simulated_num_batches": 0, # No training steps needed for this test
            "run_end_of_epoch_eval": True, # To make it run evaluate_epoch once
            "test_loading_val_dl_present": True, # Ensure val_dataloader is present
        }
        self._reinitialize_trainer(config_update)
        
        self.trainer.train(num_epochs=1) # Call train to trigger loading logic
        self.assertAlmostEqual(self.trainer.best_val_loss, 0.25)

        # Case 2: 'best_val_loss' key missing, fallback to 'loss'
        dummy_checkpoint_2 = {
            'epoch': 0,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': 0.28, # Fallback best loss
            'current_global_step': 100
        }
        mock_torch_load.return_value = dummy_checkpoint_2
        self._reinitialize_trainer(config_update) # Reinitialize with the same config

        self.trainer.train(num_epochs=1)
        self.assertAlmostEqual(self.trainer.best_val_loss, 0.28)

        # Case 3: reset_best_val_loss_on_resume = True
        mock_torch_load.return_value = dummy_checkpoint_1 # reset_best_val_loss should override this
        config_update_reset = config_update.copy()
        config_update_reset["reset_best_val_loss_on_resume"] = True
        self._reinitialize_trainer(config_update_reset)

        self.trainer.train(num_epochs=1)
        self.assertEqual(self.trainer.best_val_loss, float('inf'))


    @patch('trainer.Trainer._run_training_step')
    @patch('trainer.Trainer.evaluate_epoch')
    @patch('trainer.Trainer.save_checkpoint')
    def test_end_of_epoch_checkpoint_naming(self, mock_save_checkpoint, mock_evaluate_epoch, mock_run_step):
        mock_run_step.return_value = 0.1 # train loss
        mock_evaluate_epoch.return_value = 0.5 # val loss

        config_update = {
            "simulated_num_batches": 2, # Run a few batches
            "run_end_of_epoch_eval": True, # Ensure end of epoch eval runs
            "num_epochs": 1,
             "validate_every_batches": 0, # Disable interim validation for this test
             "checkpoint_every_batches": 0, # Disable interim checkpointing for this test
        }
        self._reinitialize_trainer(config_update)
        self.trainer.best_val_loss = 0.6 # Existing best loss is worse

        # We need to call the main train() method to test its checkpoint saving logic
        self.trainer.train(num_epochs=1)

        # Expected calls:
        # 1. End of epoch (is_best=True because 0.5 < 0.6)
        # 2. End of epoch (is_best=False, custom_filename for epoch_1.pth)
        
        # Call for the best model at end of epoch
        call_for_best = call(epoch=0, val_loss=0.5, is_best=True)
        # Call for the regular epoch checkpoint
        call_for_epoch_specific = call(epoch=0, val_loss=0.5, is_best=False, 
                                       custom_filename=f"{self.model_name}_epoch_1.pth")

        # Check if both calls were made. Order might vary if save_checkpoint for best and save_checkpoint for epoch are two separate calls.
        # Based on current trainer.py logic:
        # - if current_val_loss < self.best_val_loss, it calls save_checkpoint (is_best=True)
        # - then it calls save_checkpoint (is_best=False, custom_filename=epoch_X)
        mock_save_checkpoint.assert_any_call(epoch=0, val_loss=0.5, is_best=True)
        mock_save_checkpoint.assert_any_call(epoch=0, val_loss=0.5, is_best=False, custom_filename=f"{self.model_name}_epoch_1.pth")
        self.assertEqual(mock_save_checkpoint.call_count, 2)

    @patch('trainer.Trainer._run_training_step')
    def test_lr_scheduler_step_per_batch_after_warmup(self, mock_run_training_step_patch):
        config_update = {
            "use_lr_warmup": True,
            "lr_warmup_steps": 2, # Warmup for global_step 0 and 1
            "learning_rate": 0.1, # Target LR for warmup and initial for scheduler
            "lr_warmup_init_factor": 0.1, # Warmup starts at 0.1 * 0.1 = 0.01
            "simulated_num_batches": 10,
            "num_epochs": 1,
            "print_every": 1,
        }
        self._reinitialize_trainer(config_update)

        # Ensure the optimizer used by the trainer has the correct initial LR for the scheduler
        # self.trainer.optimizer is self.mock_optimizer as per _reinitialize_trainer structure
        self.trainer.optimizer.param_groups[0]['lr'] = config_update['learning_rate']

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.trainer.optimizer,
            step_size=3,
            gamma=0.1
        )
        self.trainer.scheduler = scheduler

        lrs_over_time = []

        mock_run_training_step_patch.side_effect = lambda inputs, targets: (lrs_over_time.append(self.trainer.optimizer.param_groups[0]['lr']), 0.1)[1]

        self.trainer.train_epoch(epoch_num=0)

        expected_lrs = [
            0.01,
            0.055,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001
        ]

        # print(f"Collected LRs for test_lr_scheduler_step_per_batch_after_warmup: {lrs_over_time}") # Optional debug line

        self.assertEqual(len(lrs_over_time), len(expected_lrs), f"Mismatch in number of LR samples. Got {len(lrs_over_time)}, expected {len(expected_lrs)}. All LRs: {lrs_over_time}")
        for i in range(len(expected_lrs)):
            self.assertAlmostEqual(lrs_over_time[i], expected_lrs[i], places=5,
                                 msg=f"LR mismatch at step {i} (global_step={i}). Got {lrs_over_time[i]}, expected {expected_lrs[i]}. All LRs: {lrs_over_time}")


if __name__ == '__main__':
    unittest.main()
