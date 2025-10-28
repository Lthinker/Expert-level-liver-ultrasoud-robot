import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import torch.nn.functional as F

class FixedMaskConvexDomainRandomization:
    def __init__(self, 
                 mask_path="default_convex_mask.npz",
                 noise_std_range=(0.01, 0.1),
                 brightness_range=(0.7, 1.3),
                 contrast_range=(0.7, 1.3),
                 speckle_intensity_range=(0.0, 0.3),
                 beam_pattern_range=(0.3, 0.7),
                 translation_range=10,
                 deformation_intensity_range=(0.15, 0.5)):
        
        # Load fixed mask
        self.load_mask(mask_path)
        
        # Domain randomization parameters
        self.noise_std_range = noise_std_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.speckle_intensity_range = speckle_intensity_range
        self.beam_pattern_range = beam_pattern_range
        self.translation_range = translation_range
        self.deformation_intensity_range = deformation_intensity_range
        
        print(f"✅ Fixed mask domain randomizer initialized")
        print(f"   Mask size: {self.mask.shape}")
        print(f"   Coverage: {self.mask_info['coverage_ratio']*100:.1f}%")
    
    def load_mask(self, mask_path):
        """Load saved mask"""
        try:
            data = np.load(mask_path, allow_pickle=True)
            self.mask = data['mask']
            self.mask_info = data['mask_info'].item()
            self.mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
            
            # Extract geometric parameters
            self.center_x = self.mask_info['center_x']
            self.center_y = self.mask_info['center_y'] 
            self.inner_radius = self.mask_info['inner_radius']
            self.outer_radius = self.mask_info['outer_radius']
            self.sector_angle = self.mask_info['sector_angle']
            
            print(f"🔍 Loaded mask geometric parameters:")
            print(f"   Center: ({self.center_x:.1f}, {self.center_y:.1f})")
            print(f"   Inner radius: {self.inner_radius:.1f}")
            print(f"   Outer radius: {self.outer_radius:.1f}")
            print(f"   Sector angle: {self.sector_angle:.1f}°")
            
        except Exception as e:
            print(f"❌ Failed to load mask: {e}")
            print("Using default mask...")
            self._create_default_mask()
    
    def _create_default_mask(self):
        """Create default mask (if loading fails)"""
        h, w = 256, 256
        center_x, center_y = w//2, h//8
        inner_radius, outer_radius = h//6, h*0.85
        sector_angle = 75
        
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        dx = x - center_x
        dy = y - center_y
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dx, dy) * 180 / np.pi
        
        half_angle = sector_angle / 2
        mask = ((distances >= inner_radius) & 
                (distances <= outer_radius) & 
                (np.abs(angles) <= half_angle))
        
        self.mask = mask.astype(np.float32)
        self.mask_tensor = torch.tensor(self.mask, dtype=torch.float32)
        self.center_x, self.center_y = center_x, center_y
        self.inner_radius, self.outer_radius = inner_radius, outer_radius
        self.sector_angle = sector_angle
        
        self.mask_info = {
            'center_x': center_x, 'center_y': center_y,
            'inner_radius': inner_radius, 'outer_radius': outer_radius,
            'sector_angle': sector_angle, 'coverage_ratio': mask.mean()
        }
    
    def ensure_float_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float type"""
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return image
    
    def resize_mask_if_needed(self, target_h, target_w):
        """Resize mask if image size doesn't match"""
        if self.mask.shape[0] != target_h or self.mask.shape[1] != target_w:
            resized_mask = cv2.resize(self.mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            resized_mask = (resized_mask > 0.5).astype(np.float32)
            scale_x = target_w / self.mask.shape[1]
            scale_y = target_h / self.mask.shape[0]
            scaled_center_x = self.center_x * scale_x
            scaled_center_y = self.center_y * scale_y
            scaled_inner_radius = self.inner_radius * scale_y
            scaled_outer_radius = self.outer_radius * scale_y
            
            return torch.tensor(resized_mask, dtype=torch.float32), (scaled_center_x, scaled_center_y, scaled_inner_radius, scaled_outer_radius)
        else:
            return self.mask_tensor, (self.center_x, self.center_y, self.inner_radius, self.outer_radius)
    
    def create_fixed_beam_pattern(self, h, w, mask_tensor, geometry, intensity=0.5):
        """Create beam pattern based on fixed mask geometry"""
        center_x, center_y, inner_radius, outer_radius = geometry
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dx = x.float() - center_x
        dy = y.float() - center_y
        distances = torch.sqrt(dx**2 + dy**2)
        
        depth_factor = (distances - inner_radius) / (outer_radius - inner_radius)
        depth_factor = torch.clamp(depth_factor, 0, 1)
        depth_attenuation = torch.exp(-depth_factor * 2.0)
        
        focal_depth = inner_radius + (outer_radius - inner_radius) * 0.4
        focal_enhancement = torch.exp(-((distances - focal_depth) / ((outer_radius - inner_radius) * 0.3))**2)
        focal_enhancement = 0.2 * focal_enhancement + 0.8
        
        beam_pattern = mask_tensor * depth_attenuation * focal_enhancement
        final_pattern = 1 - intensity * (1 - beam_pattern)
        final_pattern = torch.clamp(final_pattern, 0.2, 1.0)
        
        return final_pattern
    
    def create_fixed_deformation_field(self, h, w, mask_tensor, geometry, intensity=0.1):
        """Create deformation field based on fixed mask geometry"""
        center_x, center_y, inner_radius, outer_radius = geometry
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dx = x.float() - center_x
        dy = y.float() - center_y
        radius = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dx, dy)
        
        radial_waves = torch.sin(radius / (outer_radius - inner_radius) * 6 * math.pi)
        radial_deform = intensity * radial_waves * 0.3
        angular_deform = intensity * torch.sin(theta * 6) * torch.cos(theta * 3) * 0.2
        
        depth_factor = (radius - inner_radius) / (outer_radius - inner_radius)
        depth_factor = torch.clamp(depth_factor, 0, 1)
        depth_scaling = 0.5 + 0.5 * depth_factor
        
        new_radius = radius + radial_deform * depth_scaling * mask_tensor
        new_theta = theta + angular_deform * depth_scaling * mask_tensor
        new_x = center_x + new_radius * torch.sin(new_theta)
        new_y = center_y + new_radius * torch.cos(new_theta)
        
        grid_x = (new_x / (w - 1)) * 2 - 1
        grid_y = (new_y / (h - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        return grid
    
    def add_ultrasound_noise(self, image: torch.Tensor, mask_tensor: torch.Tensor, noise_std: float = None):
        """Add ultrasound noise (within mask region only)"""
        if noise_std is None:
            noise_std = random.uniform(*self.noise_std_range)
        
        noise = torch.randn_like(image) * noise_std
        noisy_image = image + noise * mask_tensor
        return torch.clamp(noisy_image, 0, 1)
    
    def add_speckle_noise(self, image: torch.Tensor, mask_tensor: torch.Tensor, intensity: float = None):
        """Add speckle noise (within mask region only)"""
        if intensity is None:
            intensity = random.uniform(*self.speckle_intensity_range)
        
        speckle = torch.rand_like(image) * intensity
        multiplicative_noise = 1 + (speckle - intensity/2)
        speckle_image = image * multiplicative_noise
        
        result = speckle_image * mask_tensor + image * (1 - mask_tensor)
        return torch.clamp(result, 0, 1)
    
    def adjust_brightness_contrast(self, image: torch.Tensor, brightness: float = None, contrast: float = None):
        """Adjust brightness and contrast"""
        if brightness is None:
            brightness = random.uniform(*self.brightness_range)
        if contrast is None:
            contrast = random.uniform(*self.contrast_range)
        
        image = (image - 0.5) * contrast + 0.5
        image = image * brightness
        
        return torch.clamp(image, 0, 1)
    
    def apply_translation(self, image: torch.Tensor):
        """Apply small translation"""
        h, w = image.shape[-2:]
        tx = random.randint(-self.translation_range, self.translation_range)
        ty = random.randint(-self.translation_range, self.translation_range)
        
        if tx == 0 and ty == 0:
            return image
        
        theta = torch.tensor([[1, 0, tx/w*2], [0, 1, ty/h*2]], dtype=torch.float32)
        
        if len(image.shape) == 2:
            image_batch = image.unsqueeze(0).unsqueeze(0)
        else:
            image_batch = image.unsqueeze(0)
        
        grid = F.affine_grid(theta.unsqueeze(0), image_batch.size(), align_corners=False)
        translated = F.grid_sample(image_batch, grid, align_corners=False, padding_mode='border')
        
        if len(image.shape) == 2:
            result = translated.squeeze(0).squeeze(0)
        else:
            result = translated.squeeze(0)
        
        return result
    
    def apply_all_augmentations(self, image: torch.Tensor, 
                               apply_noise=True, apply_speckle=True, apply_beam=True, 
                               apply_deformation=True, apply_brightness=True, apply_translation=True):
        """Apply all domain randomization techniques"""
        
        image = self.ensure_float_tensor(image)
        h, w = image.shape[-2:]
        
        mask_tensor, geometry = self.resize_mask_if_needed(h, w)
        augmented = image.clone()
        
        if apply_noise and random.random() < 0.8:
            augmented = self.add_ultrasound_noise(augmented, mask_tensor)
        
        if apply_speckle and random.random() < 0.7:
            augmented = self.add_speckle_noise(augmented, mask_tensor)
        
        if apply_beam and random.random() < 0.6:
            beam_intensity = random.uniform(*self.beam_pattern_range)
            beam_pattern = self.create_fixed_beam_pattern(h, w, mask_tensor, geometry, beam_intensity)
            augmented = augmented * beam_pattern
        
        if apply_deformation and random.random() < 0.85:
            deform_intensity = random.uniform(*self.deformation_intensity_range)
            deformation_grid = self.create_fixed_deformation_field(h, w, mask_tensor, geometry, deform_intensity)
            augmented_batch = augmented.unsqueeze(0).unsqueeze(0)
            grid_batch = deformation_grid.unsqueeze(0)
            deformed = F.grid_sample(augmented_batch, grid_batch, align_corners=False, padding_mode='border')
            augmented = deformed.squeeze(0).squeeze(0)
        
        if apply_brightness and random.random() < 0.7:
            augmented = self.adjust_brightness_contrast(augmented)
        
        if apply_translation and random.random() < 0.4:
            augmented = self.apply_translation(augmented)
        
        final_result = augmented * mask_tensor
        
        return final_result
    
    def get_mask_for_image(self, target_h, target_w):
        """Get mask for specific image size"""
        mask_tensor, _ = self.resize_mask_if_needed(target_h, target_w)
        return mask_tensor

def test_fixed_mask_augmentation(mask_path="default_convex_mask.npz"):
    """Test fixed-mask-based domain randomization"""
    
    print("🔧 Testing fixed-mask domain randomization")
    print("=" * 50)
    
    dr = FixedMaskConvexDomainRandomization(mask_path=mask_path)
    test_image = create_test_ultrasound_image(256, 256)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Fixed Mask Domain Randomization Results', fontsize=16)
    
    axes[0, 0].imshow(test_image.numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    mask_for_display = dr.get_mask_for_image(256, 256)
    axes[0, 1].imshow(mask_for_display.numpy(), cmap='gray')
    axes[0, 1].set_title('Used Mask')
    axes[0, 1].axis('off')
    
    masked_original = test_image * mask_for_display
    axes[0, 2].imshow(masked_original.numpy(), cmap='gray')
    axes[0, 2].set_title('Masked Original')
    axes[0, 2].axis('off')
    
    effects = [
        ('Noise Only', lambda img: dr.apply_all_augmentations(img, apply_speckle=False, apply_beam=False, apply_deformation=False, apply_brightness=False, apply_translation=False)),
        ('Speckle Only', lambda img: dr.apply_all_augmentations(img, apply_noise=False, apply_beam=False, apply_deformation=False, apply_brightness=False, apply_translation=False)),
        ('Beam Only', lambda img: dr.apply_all_augmentations(img, apply_noise=False, apply_speckle=False, apply_deformation=False, apply_brightness=False, apply_translation=False)),
        ('Deformation Only', lambda img: dr.apply_all_augmentations(img, apply_noise=False, apply_speckle=False, apply_beam=False, apply_brightness=False, apply_translation=False)),
        ('Brightness Only', lambda img: dr.apply_all_augmentations(img, apply_noise=False, apply_speckle=False, apply_beam=False, apply_deformation=False, apply_translation=False)),
        ('Translation Only', lambda img: dr.apply_all_augmentations(img, apply_noise=False, apply_speckle=False, apply_beam=False, apply_deformation=False, apply_brightness=False)),
        ('Combined Light', lambda img: dr.apply_all_augmentations(img)),
        ('Combined Heavy', lambda img: dr.apply_all_augmentations(img)),
        ('Full Random 1', lambda img: dr.apply_all_augmentations(img)),
        ('Full Random 2', lambda img: dr.apply_all_augmentations(img))
    ]
    
    positions = [(0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
    
    for i, ((title, effect_func), (row, col)) in enumerate(zip(effects[:9], positions)):
        try:
            result = effect_func(test_image.clone())
            axes[row, col].imshow(result.numpy(), cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        except Exception as e:
            print(f"Error in {title}: {e}")
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return dr

def create_test_ultrasound_image(h, w):
    """Create test ultrasound image"""
    image = torch.zeros(h, w, dtype=torch.float32)
    
    structures = [
        (w//2, h//4, 20, 0.8),
        (w//2 - 30, h//2, 15, 0.6),
        (w//2 + 30, h//2, 15, 0.6),
        (w//2, 3*h//4, 25, 0.7)
    ]
    
    image_np = image.numpy()
    for cx, cy, radius, intensity in structures:
        cv2.circle(image_np, (cx, cy), radius, intensity, -1)
    
    image = torch.tensor(image_np) + 0.1
    noise = torch.randn(h, w) * 0.05
    image = torch.clamp(image + noise, 0, 1)
    
    return image

def batch_process_with_fixed_mask(image_list, mask_path="default_convex_mask.npz", save_results=False):
    """Batch process image list"""
    
    print(f"🔄 Starting batch processing of {len(image_list)} images")
    
    dr = FixedMaskConvexDomainRandomization(mask_path=mask_path)
    results = []
    
    for i, image in enumerate(image_list):
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        
        augmented = dr.apply_all_augmentations(image)
        results.append(augmented)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed: {i + 1}/{len(image_list)}")
    
    print("✅ Batch processing complete")
    
    if save_results:
        save_path = "batch_augmented_results.npz"
        np.savez(save_path, 
                 original_images=[img.numpy() for img in image_list],
                 augmented_images=[img.numpy() for img in results])
        print(f"💾 Results saved to: {save_path}")
    
    return results

def main():
    """Main function"""
    print("🚀 Fixed Mask Convex Array Ultrasound Domain Randomization")
    print("=" * 50)
    
    mask_path = "default_convex_mask.npz"
    try:
        data = np.load(mask_path, allow_pickle=True)
        print(f"✅ Found mask file: {mask_path}")
        mask_info = data['mask_info'].item()
        print(f"   Mask size: {data['mask'].shape}")
        print(f"   Coverage: {mask_info['coverage_ratio']*100:.1f}%")
    except:
        print(f"❌ Mask file not found: {mask_path}")
        print("Please run the mask creation script first to generate the mask file")
        return
    
    print("\n🔍 Testing fixed mask domain randomization...")
    dr = test_fixed_mask_augmentation(mask_path)
    
    print("\n💡 Usage instructions:")
    print("1. Create a domain randomizer:")
    print("   dr = FixedMaskConvexDomainRandomization('your_mask.npz')")
    print("2. Process a single image:")
    print("   augmented = dr.apply_all_augmentations(your_image)")
    print("3. Batch process:")
    print("   results = batch_process_with_fixed_mask(image_list, 'your_mask.npz')")
    print("4. Get mask for specific size:")
    print("   mask = dr.get_mask_for_image(height, width)")
    
    print("\n🎯 Advantages:")
    print("- ✅ Ensures consistency using fixed mask")
    print("- ⚡ Fast, no repeated detection required")
    print("- 🎛️ Flexible augmentation control")
    print("- 📏 Automatically adapts to different image sizes")
    print("- 🔧 Highly configurable parameters")

if __name__ == "__main__":
    main()