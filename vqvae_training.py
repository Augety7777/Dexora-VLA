import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import wandb
from data.bson_vla_dataset import BsonVLADataset


class VQVAEActionEncoder(nn.Module):
    """
    2D CNN Encoder for action chunks with global compression - Larger version
    """
    def __init__(self, input_dim: int, chunk_size: int, hidden_dim: int = 512, 
                 feature_multiplier: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Base channel sizes (multiplied by feature_multiplier for larger models)
        base_channels = [64, 128, 256, 512, 512]
        channels = [c * feature_multiplier for c in base_channels]
        
        # Deeper CNN layers with residual connections
        self.conv_layers = nn.ModuleList([
            # First block
            nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
            ),
            # Second block
            nn.Sequential(
                nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
            ),
            # Third block
            nn.Sequential(
                nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[2], channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(inplace=True),
            ),
            # Fourth block
            nn.Sequential(
                nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[3]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[3]),
                nn.ReLU(inplace=True),
            ),
            # Fifth block
            nn.Sequential(
                nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[4]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[4], channels[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[4]),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.Conv2d(1, channels[0], kernel_size=1),  # Skip connection for block 1
            nn.Conv2d(channels[0], channels[1], kernel_size=1),  # Skip connection for block 2
            nn.Conv2d(channels[1], channels[2], kernel_size=1),  # Skip connection for block 3
            nn.Conv2d(channels[2], channels[3], kernel_size=1),  # Skip connection for block 4
            nn.Conv2d(channels[3], channels[4], kernel_size=1),  # Skip connection for block 5
        ])
        
        # Multiple global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Attention mechanism for better feature aggregation
        self.attention = nn.Sequential(
            nn.Linear(channels[4] * 2, channels[4]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[4], channels[4]),
            nn.Sigmoid()
        )
        
        # Larger projection network
        self.projection = nn.Sequential(
            nn.Linear(channels[4], hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, chunk_size, input_dim] action chunk
        Returns:
            [B, hidden_dim] global encoded features
        """
        batch_size = x.shape[0]
        
        # Add channel dimension: [B, chunk_size, input_dim] -> [B, 1, chunk_size, input_dim]
        x = x.unsqueeze(1)
        identity = x
        
        # Apply conv layers with residual connections
        for i, (conv_block, downsample) in enumerate(zip(self.conv_layers, self.downsample_layers)):
            # Apply convolution block
            out = conv_block(x)
            
            # Skip connection
            identity = downsample(identity)
            x = out + identity
            identity = x
        
        # Multiple pooling strategies
        avg_pooled = self.global_avg_pool(x).view(batch_size, -1)  # [B, channels[4]]
        max_pooled = self.global_max_pool(x).view(batch_size, -1)  # [B, channels[4]]
        
        # Combine features
        combined = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, channels[4] * 2]
        
        # Apply attention
        attention_weights = self.attention(combined)  # [B, channels[4]]
        weighted_features = avg_pooled * attention_weights + max_pooled * (1 - attention_weights)
        
        # Final projection
        encoded = self.projection(weighted_features)
        
        return encoded


class VQVAEActionDecoder(nn.Module):
    """
    Decoder that reconstructs action chunks from global features - Larger version
    """
    def __init__(self, input_dim: int, chunk_size: int, hidden_dim: int = 512,
                 feature_multiplier: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        
        # Base channel sizes (multiplied by feature_multiplier for larger models)
        base_channels = [512, 512, 256, 128, 64]
        channels = [c * feature_multiplier for c in base_channels]
        
        # Compute intermediate feature map sizes
        self.initial_size = max(8, min(chunk_size, input_dim) // 4)  # Adaptive initial size
        
        # Larger projection network
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, channels[0] * self.initial_size * self.initial_size)
        )
        
        # Deeper transpose convolution blocks with residual connections
        self.deconv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.ConvTranspose2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[1], channels[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
            ),
            # Block 2
            nn.Sequential(
                nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[2], channels[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[2]),
                nn.ReLU(inplace=True),
            ),
            # Block 3
            nn.Sequential(
                nn.ConvTranspose2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[3]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[3], channels[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[3]),
                nn.ReLU(inplace=True),
            ),
            # Block 4
            nn.Sequential(
                nn.ConvTranspose2d(channels[3], channels[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[4]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels[4], channels[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[4]),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Skip connection layers for upsampling
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels[3], channels[4], kernel_size=1),
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[4], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        
        # Adaptive pooling to get exact target size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((chunk_size, input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_dim] global encoded features
        Returns:
            [B, chunk_size, input_dim] reconstructed action chunk
        """
        batch_size = x.shape[0]
        
        # Project to feature map: [B, hidden_dim] -> [B, channels[0]*H*W]
        x = self.projection(x)
        
        # Reshape to feature map: [B, channels[0]*H*W] -> [B, channels[0], H, W]
        channels_0 = len(self.deconv_blocks[0]) * 32 if hasattr(self, 'channels') else 512 * 2  # feature_multiplier * base
        x = x.view(batch_size, -1, self.initial_size, self.initial_size)
        identity = x
        
        # Apply deconv blocks with residual connections
        for i, (deconv_block, upsample) in enumerate(zip(self.deconv_blocks, self.upsample_layers)):
            # Apply deconvolution block
            out = deconv_block(x)
            
            # Skip connection with upsampling
            identity = upsample(identity)
            x = out + identity
            identity = x
        
        # Final convolution
        x = self.final_conv(x)
        
        # Adaptive pooling to exact target size
        x = self.adaptive_pool(x)
        
        # Remove channel dimension: [B, 1, chunk_size, input_dim] -> [B, chunk_size, input_dim]
        x = x.squeeze(1)
        
        return x


class VectorQuantizerGlobal(nn.Module):
    """
    Vector Quantization layer for global features
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: [B, embedding_dim] global feature vectors
        Returns:
            quantized: [B, embedding_dim] quantized vectors
            loss: scalar tensor with VQ loss
            perplexity: scalar tensor with codebook perplexity
        """
        # Calculate distances to all codebook vectors
        # inputs: [B, D], embeddings: [K, D]
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embeddings.weight.t()))
        
        # Find closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)  # [B]
        
        # Create one-hot encodings
        encodings = torch.zeros(inputs.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)  # [B, D]
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity


class VQVAEAction(nn.Module):
    """
    VQ-VAE for action chunk compression and reconstruction with global compression - Larger version
    """
    def __init__(self, input_dim: int, chunk_size: int, hidden_dim: int = 512, 
                 num_embeddings: int = 1024, commitment_cost: float = 0.25,
                 feature_multiplier: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.feature_multiplier = feature_multiplier
        
        self.encoder = VQVAEActionEncoder(input_dim, chunk_size, hidden_dim, feature_multiplier)
        self.vq_layer = VectorQuantizerGlobal(num_embeddings, hidden_dim, commitment_cost)
        self.decoder = VQVAEActionDecoder(input_dim, chunk_size, hidden_dim, feature_multiplier)
        
        # Add a learnable scale factor for better reconstruction
        self.reconstruction_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, chunk_size, input_dim] action chunk
        Returns:
            reconstructed: [B, chunk_size, input_dim] reconstructed action chunk
            vq_loss: scalar tensor with VQ loss
            perplexity: scalar tensor with codebook perplexity
        """
        # Encode to global features: [B, chunk_size, input_dim] -> [B, hidden_dim]
        encoded = self.encoder(x)
        
        # Vector quantization: [B, hidden_dim] -> [B, hidden_dim]
        quantized, vq_loss, perplexity = self.vq_layer(encoded)
        
        # Decode back to action chunk: [B, hidden_dim] -> [B, chunk_size, input_dim]
        reconstructed = self.decoder(quantized)
        
        # Apply learnable scaling
        reconstructed = reconstructed * self.reconstruction_scale
        
        return reconstructed, vq_loss, perplexity
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode action chunk to quantized global features
        Args:
            x: [B, chunk_size, input_dim] action chunk
        Returns:
            [B, hidden_dim] quantized global features
        """
        encoded = self.encoder(x)
        quantized, _, _ = self.vq_layer(encoded)
        return quantized
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features back to action chunk
        Args:
            z: [B, hidden_dim] quantized global features
        Returns:
            [B, chunk_size, input_dim] reconstructed action chunk
        """
        reconstructed = self.decoder(z)
        return reconstructed * self.reconstruction_scale
    
    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get codebook indices for input action chunks
        Args:
            x: [B, chunk_size, input_dim] action chunk
        Returns:
            [B] codebook indices
        """
        encoded = self.encoder(x)
        
        # Calculate distances to codebook
        distances = (torch.sum(encoded**2, dim=1, keepdim=True) 
                    + torch.sum(self.vq_layer.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(encoded, self.vq_layer.embeddings.weight.t()))
        
        # Get indices
        indices = torch.argmin(distances, dim=1)
        return indices
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode action chunks from codebook indices
        Args:
            indices: [B] codebook indices
        Returns:
            [B, chunk_size, input_dim] reconstructed action chunks
        """
        # Get quantized vectors from indices
        quantized = self.vq_layer.embeddings(indices)  # [B, hidden_dim]
        
        # Decode
        return self.decode(quantized)
    
    def get_model_size(self):
        """
        Get model size information
        """
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        vq_params = sum(p.numel() for p in self.vq_layer.parameters())
        
        return {
            'total_params': total_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'vq_params': vq_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }


class ActionChunkDataset(Dataset):
    """
    Dataset wrapper for action chunks from BsonVLADataset
    """
    def __init__(self, bson_dataset: BsonVLADataset, num_samples: int = 10000):
        self.bson_dataset = bson_dataset
        self.num_samples = num_samples
        
        # Pre-cache some samples to avoid repeated file I/O in workers
        print("Pre-caching action samples...")
        self.cached_samples = []
        for _ in tqdm(range(num_samples), desc="Caching samples"):
            sample = self.bson_dataset.get_item(action_only=True)
            actions = sample['actions']  # [chunk_size, input_dim]
            self.cached_samples.append(torch.FloatTensor(actions))
        
        if not self.cached_samples:
            raise ValueError("No valid samples could be cached!")
            
        print(f"Successfully cached {len(self.cached_samples)} samples")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Randomly select from cached samples
        cache_idx = np.random.randint(0, len(self.cached_samples))
        return self.cached_samples[cache_idx].clone()


def train_vqvae(config: Dict):
    """
    Training function for VQ-VAE
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    print("Initializing dataset...")
    bson_dataset = BsonVLADataset(
        bson_dir=config['data']['bson_dir'],
        sub_sample=config['data']['sub_sample']
    )
    
    # Get action dimensions from a sample
    sample = bson_dataset.get_item()
    chunk_size, input_dim = sample['actions'].shape
    print(f"Action dimensions: chunk_size={chunk_size}, input_dim={input_dim}")
    
    # Create dataset and dataloader
    train_dataset = ActionChunkDataset(bson_dataset, config['training']['num_samples'])
    
    print(f"Testing with {config['training']['num_workers']} workers...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=min(config['training']['num_workers'], 2),  # Limit workers
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2
    )
    # Test multiprocess dataloader
    test_batch = next(iter(train_loader))
    print(f"Successfully loaded test batch with multiprocessing: {test_batch.shape}")
    
    # Initialize model
    model = VQVAEAction(
        input_dim=input_dim,
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_embeddings=config['model']['num_embeddings'],
        commitment_cost=config['model']['commitment_cost']
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['scheduler_step'],
        gamma=config['training']['scheduler_gamma']
    )
    
    # Initialize wandb if enabled
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            config=config,
            name=config['logging']['run_name']
        )
        wandb.watch(model)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        epoch_perplexity = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, actions in enumerate(pbar):
            actions = actions.to(device)
            
            # Forward pass
            reconstructed, vq_loss, perplexity = model(actions)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, actions)
            
            # Total loss
            total_loss = recon_loss + vq_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            epoch_perplexity += perplexity.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'VQ': f'{vq_loss.item():.4f}',
                'Perp': f'{perplexity.item():.2f}'
            })
        
        # Calculate average losses
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_vq_loss = epoch_vq_loss / len(train_loader)
        avg_perplexity = epoch_perplexity / len(train_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, "
              f"VQ={avg_vq_loss:.4f}, Perplexity={avg_perplexity:.2f}")
        
        if config['logging']['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'recon_loss': avg_recon_loss,
                'vq_loss': avg_vq_loss,
                'perplexity': avg_perplexity,
                'lr': scheduler.get_last_lr()[0]
            })
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
            torch.save(checkpoint, os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt'))
            print(f"Saved best model with loss: {best_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['logging']['save_freq'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['logging']['checkpoint_dir'], f'epoch_{epoch+1}.pt'))
    
    print("Training completed!")
    
def evaluate_model(config: Dict, checkpoint_path: str = None):
    """
    Evaluation function for trained VQ-VAE model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    print("Initializing dataset...")
    bson_dataset = BsonVLADataset(
        bson_dir=config['data']['bson_dir'],
        sub_sample=config['data']['sub_sample']
    )
    
    # Get action dimensions from a sample
    sample = bson_dataset.get_item()
    chunk_size, input_dim = sample['actions'].shape
    print(f"Action dimensions: chunk_size={chunk_size}, input_dim={input_dim}")
    
    # Initialize model
    model = VQVAEAction(
        input_dim=input_dim,
        chunk_size=chunk_size,
        hidden_dim=config['model']['hidden_dim'],
        num_embeddings=config['model']['num_embeddings'],
        commitment_cost=config['model']['commitment_cost']
    ).to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    # Create evaluation dataset
    eval_dataset = ActionChunkDataset(bson_dataset, 100)  # Small evaluation set
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Evaluation
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    all_original = []
    all_reconstructed = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for actions in tqdm(eval_loader, desc="Evaluation"):
            actions = actions.to(device)
            
            # Forward pass
            reconstructed, vq_loss, perplexity = model(actions)
            recon_loss = F.mse_loss(reconstructed, actions)
            total_loss_batch = recon_loss + vq_loss
            
            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Store for visualization
            all_original.append(actions.cpu())
            all_reconstructed.append(reconstructed.cpu())
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_vq_loss = total_vq_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"Average VQ Loss: {avg_vq_loss:.4f}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    
    # Concatenate all samples
    all_original = torch.cat(all_original, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    
    # Visualization
    visualize_reconstruction(all_original, all_reconstructed, config, save_dir=config['logging']['checkpoint_dir'])
    
    return {
        'avg_loss': avg_loss,
        'avg_recon_loss': avg_recon_loss,
        'avg_vq_loss': avg_vq_loss,
        'avg_perplexity': avg_perplexity
    }


def visualize_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, 
                           config: Dict, save_dir: str, sample_idx: int = 0, 
                           axes_to_plot: list = None):
    """
    Visualize reconstruction results for specific sample and axes
    
    Args:
        original: [N, chunk_size, input_dim] original action chunks
        reconstructed: [N, chunk_size, input_dim] reconstructed action chunks
        config: configuration dictionary
        save_dir: directory to save plots
        sample_idx: which sample to visualize
        axes_to_plot: list of axis indices to plot, if None plot first 6 axes
    """
    if axes_to_plot is None:
        # Default: plot first 6 axes (left arm: 6 joints, right arm: 6 joints, etc.)
        axes_to_plot = list(range(min(12, original.shape[-1])))  # First 12 axes
    
    sample_original = original[sample_idx].numpy()  # [chunk_size, input_dim]
    sample_reconstructed = reconstructed[sample_idx].numpy()  # [chunk_size, input_dim]
    
    chunk_size, input_dim = sample_original.shape
    print(f"Visualizing sample {sample_idx}: chunk_size={chunk_size}, input_dim={input_dim}")
    
    # Create time steps
    time_steps = np.arange(chunk_size)
    
    # Calculate number of subplots
    n_axes = len(axes_to_plot)
    n_cols = min(4, n_axes)
    n_rows = (n_axes + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [[axes]]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define axis labels (assuming standard robot configuration)
    axis_labels = [
        'L_shoulder_pan', 'L_shoulder_lift', 'L_elbow', 'L_wrist1', 'L_wrist2', 'L_wrist3',  # Left arm
        'L_thumb', 'L_index', 'L_middle', 'L_ring', 'L_pinky', 'L_wrist_rot',              # Left hand
        'R_shoulder_pan', 'R_shoulder_lift', 'R_elbow', 'R_wrist1', 'R_wrist2', 'R_wrist3',  # Right arm  
        'R_thumb', 'R_index', 'R_middle', 'R_ring', 'R_pinky', 'R_wrist_rot',              # Right hand
    ]
    
    # Ensure we have enough labels
    while len(axis_labels) < input_dim:
        axis_labels.append(f'Axis_{len(axis_labels)}')
    
    # Plot each axis
    for i, axis_idx in enumerate(axes_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if axis_idx < input_dim:
            # Plot original and reconstructed
            ax.plot(time_steps, sample_original[:, axis_idx], 'b-', label='Original', linewidth=2, alpha=0.8)
            ax.plot(time_steps, sample_reconstructed[:, axis_idx], 'r--', label='Reconstructed', linewidth=2, alpha=0.8)
            
            # Calculate reconstruction error
            mse = np.mean((sample_original[:, axis_idx] - sample_reconstructed[:, axis_idx]) ** 2)
            ax.set_title(f'{axis_labels[axis_idx]}\nMSE: {mse:.4f}', fontsize=10)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')  # Hide extra subplots
    
    # Remove extra subplots
    for i in range(n_axes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'reconstruction_sample_{sample_idx}_axes_{"-".join(map(str, axes_to_plot))}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction plot: {save_path}")
    
    # Also create a summary plot with overall statistics
    create_summary_plot(original, reconstructed, save_dir)
    
    plt.show()


def create_summary_plot(original: torch.Tensor, reconstructed: torch.Tensor, save_dir: str):
    """
    Create summary plots showing overall reconstruction quality
    """
    original_np = original.numpy()
    reconstructed_np = reconstructed.numpy()
    
    # Calculate per-axis MSE across all samples
    mse_per_axis = np.mean((original_np - reconstructed_np) ** 2, axis=(0, 1))
    
    # Calculate per-sample MSE
    mse_per_sample = np.mean((original_np - reconstructed_np) ** 2, axis=(1, 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Per-axis MSE
    axes[0, 0].bar(range(len(mse_per_axis)), mse_per_axis)
    axes[0, 0].set_title('Reconstruction MSE per Axis')
    axes[0, 0].set_xlabel('Axis Index')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Per-sample MSE histogram
    axes[0, 1].hist(mse_per_sample, bins=20, alpha=0.7)
    axes[0, 1].set_title('Distribution of Sample MSE')
    axes[0, 1].set_xlabel('MSE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation between original and reconstructed
    original_flat = original_np.flatten()
    reconstructed_flat = reconstructed_np.flatten()
    axes[1, 0].scatter(original_flat, reconstructed_flat, alpha=0.1, s=1)
    axes[1, 0].plot([original_flat.min(), original_flat.max()], 
                    [original_flat.min(), original_flat.max()], 'r--', alpha=0.8)
    axes[1, 0].set_title('Original vs Reconstructed Values')
    axes[1, 0].set_xlabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Error distribution over time
    error_per_timestep = np.mean((original_np - reconstructed_np) ** 2, axis=(0, 2))
    axes[1, 1].plot(error_per_timestep, 'b-', linewidth=2)
    axes[1, 1].set_title('Reconstruction Error over Time')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(save_dir, 'reconstruction_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE for action chunk compression')
    parser.add_argument('--config', type=str, default='vqvae_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--bson_dir', type=str, default='data/ours',
                       help='Path to BSON dataset directory')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation (skip training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize (default: 0)')
    parser.add_argument('--axes', type=str, default=None,
                       help='Comma-separated list of axis indices to plot (e.g., "0,1,2,6,7,8")')
    
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'data': {
                'bson_dir': args.bson_dir,
                'sub_sample': 1.0
            },
            'model': {
                'hidden_dim': 128,
                'num_embeddings': 512,
                'commitment_cost': 0.25
            },
            'training': {
                'num_epochs': 100,
                'batch_size': 16,  # Reduced batch size
                'lr': 1e-3,
                'num_workers': 0,  # Start with 0 workers to avoid segfault
                'num_samples': 5000,  # Reduced for faster testing
                'scheduler_step': 30,
                'scheduler_gamma': 0.5
            },
            'logging': {
                'use_wandb': False,
                'wandb_project': 'vqvae-action-chunks',
                'run_name': 'vqvae_training',
                'checkpoint_dir': 'checkpoints/vqvae',
                'save_freq': 10
            }
        }
        
        # Save default config
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created default config file: {args.config}")
    
    # Override bson_dir from command line
    if args.bson_dir != 'data/ours':
        config['data']['bson_dir'] = args.bson_dir
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Parse axes to plot
    axes_to_plot = None
    if args.axes:
        try:
            axes_to_plot = [int(x.strip()) for x in args.axes.split(',')]
            print(f"Will plot axes: {axes_to_plot}")
        except ValueError:
            print(f"Invalid axes format: {args.axes}. Using default axes.")
    
    if args.eval_only:
        # Only run evaluation
        print("Running evaluation only...")
        results = evaluate_model(config, args.checkpoint)
        print(f"Evaluation completed. Results: {results}")
        
        # Additional custom visualization if requested
        if axes_to_plot is not None or args.sample_idx != 0:
            print(f"Creating custom visualization for sample {args.sample_idx}...")
            # Load data for custom visualization
            bson_dataset = BsonVLADataset(config['data']['bson_dir'], config['data']['sub_sample'])
            eval_dataset = ActionChunkDataset(bson_dataset, 100)
            eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0)
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            sample = bson_dataset.get_item()
            chunk_size, input_dim = sample['actions'].shape
            
            model = VQVAEAction(
                input_dim=input_dim,
                chunk_size=chunk_size,
                hidden_dim=config['model']['hidden_dim'],
                num_embeddings=config['model']['num_embeddings'],
                commitment_cost=config['model']['commitment_cost'],
                feature_multiplier=config['model']['feature_multiplier']
            ).to(device)
            
            checkpoint_path = args.checkpoint or os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get samples for visualization
            all_original = []
            all_reconstructed = []
            with torch.no_grad():
                for actions in eval_loader:
                    actions = actions.to(device)
                    reconstructed, _, _ = model(actions)
                    all_original.append(actions.cpu())
                    all_reconstructed.append(reconstructed.cpu())
                    if len(all_original) * actions.shape[0] > args.sample_idx + 10:
                        break
            
            all_original = torch.cat(all_original, dim=0)
            all_reconstructed = torch.cat(all_reconstructed, dim=0)
            
            if args.sample_idx < len(all_original):
                visualize_reconstruction(
                    all_original, all_reconstructed, config,
                    save_dir=config['logging']['checkpoint_dir'],
                    sample_idx=args.sample_idx,
                    axes_to_plot=axes_to_plot
                )
            else:
                print(f"Sample index {args.sample_idx} out of range. Available: 0-{len(all_original)-1}")
    else:
        # Run training
        train_vqvae(config)
        
        # Run evaluation after training
        print("\nRunning final evaluation...")
        results = evaluate_model(config)
        print(f"Final evaluation results: {results}")


if __name__ == "__main__":
    main()