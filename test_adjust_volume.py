import os
import sys
import numpy as np
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def log_info(message, emoji="ℹ️"):
    print(f"{emoji} {message}")

def log_success(message, emoji="✅"):
    print(f"{emoji} {message}")

def log_error(message, emoji="❌"):
    print(f"{emoji} {message}")

def test_adjust_volume():
    """Test the adjust_volume function with both scalar and array inputs."""
    try:
        # Define parameters
        music_volume = 0.3  # Updated from 0.15 to 0.3
        fade_in = 2.0
        fade_out = 3.0
        voice_duration = 10.0
        
        # Create the adjust_volume function
        def adjust_volume(t):
            # Base volume is the music_volume parameter
            base_vol = float(music_volume)  # Ensure it's a float
            
            # Handle both scalar and array inputs
            if isinstance(t, (int, float)):
                # Scalar case
                if t < fade_in and fade_in > 0:
                    return base_vol * (t / fade_in)
                elif fade_out > 0 and t > voice_duration - fade_out:
                    fade_position = (t - (voice_duration - fade_out)) / fade_out
                    return base_vol * (1 - fade_position)
                else:
                    return base_vol
            else:
                # Array case - handle element-wise
                result = np.ones_like(t) * base_vol
                
                # Apply fade in
                if fade_in > 0:
                    fade_in_mask = t < fade_in
                    if np.any(fade_in_mask):
                        result[fade_in_mask] = base_vol * (t[fade_in_mask] / fade_in)
                
                # Apply fade out
                if fade_out > 0:
                    fade_out_mask = t > (voice_duration - fade_out)
                    if np.any(fade_out_mask):
                        fade_position = (t[fade_out_mask] - (voice_duration - fade_out)) / fade_out
                        result[fade_out_mask] = base_vol * (1 - fade_position)
                
                return result
        
        # Test with scalar inputs
        log_info("Testing with scalar inputs...")
        t_values = [0.0, 1.0, 2.0, 5.0, 7.0, 8.0, 9.0, 10.0]
        for t in t_values:
            vol = adjust_volume(t)
            log_info(f"t={t}, volume={vol}")
        
        # Test with array inputs
        log_info("\nTesting with array inputs...")
        t_array = np.array(t_values)
        vol_array = adjust_volume(t_array)
        log_info(f"t_array={t_array}")
        log_info(f"vol_array={vol_array}")
        
        # Test with a more complex array
        log_info("\nTesting with a more complex array...")
        t_complex = np.linspace(0, 10, 11)
        vol_complex = adjust_volume(t_complex)
        for i, (t, vol) in enumerate(zip(t_complex, vol_complex)):
            log_info(f"t[{i}]={t:.2f}, volume={vol:.4f}")
        
        # Test with different parameters
        log_info("\nTesting with different parameters...")
        
        # No fade in
        fade_in_test = 0.0
        def adjust_volume_no_fade_in(t):
            if isinstance(t, (int, float)):
                if fade_out > 0 and t > voice_duration - fade_out:
                    fade_position = (t - (voice_duration - fade_out)) / fade_out
                    return music_volume * (1 - fade_position)
                else:
                    return music_volume
            else:
                return np.ones_like(t) * music_volume
        
        log_info("Testing with no fade in...")
        for t in [0.0, 1.0, 5.0]:
            vol = adjust_volume_no_fade_in(t)
            log_info(f"t={t}, volume={vol}")
        
        # No fade out
        fade_out_test = 0.0
        def adjust_volume_no_fade_out(t):
            if isinstance(t, (int, float)):
                if t < fade_in and fade_in > 0:
                    return music_volume * (t / fade_in)
                else:
                    return music_volume
            else:
                return np.ones_like(t) * music_volume
        
        log_info("Testing with no fade out...")
        for t in [8.0, 9.0, 10.0]:
            vol = adjust_volume_no_fade_out(t)
            log_info(f"t={t}, volume={vol}")
        
        log_success("All tests passed successfully!")
        return True
    except Exception as e:
        log_error(f"Error testing adjust_volume: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_adjust_volume() 