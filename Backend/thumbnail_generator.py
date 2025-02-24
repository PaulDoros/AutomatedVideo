from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
from termcolor import colored
import requests
from io import BytesIO

class ThumbnailGenerator:
    def __init__(self):
        self.size = (1280, 720)  # YouTube thumbnail size
        self.assets_dir = "assets"
        self.fonts_dir = f"{self.assets_dir}/fonts"
        self.backgrounds_dir = f"{self.assets_dir}/backgrounds"
        
        # Create necessary directories
        for directory in [self.fonts_dir, self.backgrounds_dir, "test_thumbnails"]:
            os.makedirs(directory, exist_ok=True)
        
        # Ensure fonts exist and are valid
        self.montserrat_bold = f"{self.fonts_dir}/Montserrat-Bold.ttf"
        self.roboto_regular = f"{self.fonts_dir}/Roboto-Regular.ttf"
        
        # Always run font setup to verify fonts
        print(colored("Verifying fonts...", "blue"))
        from setup_fonts import download_fonts
        download_fonts()

    def generate_test_thumbnails(self):
        """Generate test thumbnails for each channel"""
        templates = {
            'tech_humor': {
                'bg_color': '#1E1E1E',
                'gradient': ['#FF4D4D', '#1E1E1E'],
                'text': 'When Your Code Works',
                'subtitle': 'Mind = Blown',
                'icon': '🔥',
                'tech_element': '💻',
                'font_size': 120,
                'subtitle_size': 72,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay', 'blur']
            },
            'ai_money': {
                'bg_color': '#004D40',
                'gradient': ['#00C853', '#004D40'],
                'text': '$500/Day with AI',
                'subtitle': 'Passive Income',
                'money_icon': '💰',
                'numbers': '$500',
                'font_size': 110,
                'subtitle_size': 66,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay']
            },
            'baby_tips': {
                'bg_color': '#E1BEE7',
                'gradient': ['#CE93D8', '#E1BEE7'],
                'text': 'Sleep Tips for Baby',
                'subtitle': 'Expert Guide',
                'style': 'parenting',
                'font_size': 100,
                'subtitle_size': 60,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay']
            },
            'quick_meals': {
                'bg_color': '#FF5722',
                'gradient': ['#FF7043', '#FF5722'],
                'text': '15-Min Healthy Meal',
                'subtitle': 'Quick & Easy',
                'style': 'food',
                'font_size': 100,
                'subtitle_size': 60,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay']
            },
            'fitness_motivation': {
                'bg_color': '#1A237E',
                'gradient': ['#303F9F', '#1A237E'],
                'text': 'Full Body Workout',
                'subtitle': 'Get Fit Fast',
                'style': 'fitness',
                'font_size': 100,
                'subtitle_size': 60,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay']
            }
        }

        for channel, template in templates.items():
            try:
                # Create base image
                img = Image.new('RGB', self.size, template['bg_color'])

                # Apply gradient with stronger contrast
                if 'gradient' in template['effects']:
                    gradient = self.create_gradient(template['gradient'])
                    img = Image.blend(img, gradient, 0.8)

                # Apply stronger overlay for better contrast
                if 'overlay' in template['effects']:
                    overlay = Image.new('RGB', self.size, template['bg_color'])
                    img = Image.blend(img, overlay, 0.4)

                # Load fonts with larger sizes
                try:
                    title_font = ImageFont.truetype(self.montserrat_bold, template['font_size'])
                    subtitle_font = ImageFont.truetype(self.roboto_regular, template['subtitle_size'])
                except Exception as e:
                    print(colored(f"Error loading fonts: {str(e)}", "red"))
                    print(colored("Using default font as fallback", "yellow"))
                    title_font = ImageFont.load_default()
                    subtitle_font = ImageFont.load_default()

                draw = ImageDraw.Draw(img)

                # Add text with enhanced shadow and stroke
                self.draw_enhanced_text(
                    draw, 
                    template['text'], 
                    title_font, 
                    template['text_color'],
                    template['shadow_color'],
                    offset_y=-50
                )
                
                self.draw_enhanced_text(
                    draw, 
                    template['subtitle'], 
                    subtitle_font,
                    template['text_color'],
                    template['shadow_color'],
                    offset_y=50
                )

                # Save with high quality
                output_path = f"test_thumbnails/{channel}.jpg"
                img.save(output_path, 'JPEG', quality=95)
                print(colored(f"✓ Generated thumbnail for {channel}", "green"))

            except Exception as e:
                print(colored(f"✗ Error generating {channel} thumbnail: {str(e)}", "red"))

    def create_gradient(self, colors):
        """Create a gradient background"""
        gradient = Image.new('RGB', self.size)
        draw = ImageDraw.Draw(gradient)
        
        for i in range(self.size[1]):
            r = int((self.size[1] - i) * int(colors[0][1:3], 16) / self.size[1] + i * int(colors[1][1:3], 16) / self.size[1])
            g = int((self.size[1] - i) * int(colors[0][3:5], 16) / self.size[1] + i * int(colors[1][3:5], 16) / self.size[1])
            b = int((self.size[1] - i) * int(colors[0][5:7], 16) / self.size[1] + i * int(colors[1][5:7], 16) / self.size[1])
            draw.line([(0, i), (self.size[0], i)], fill=(r, g, b))
            
        return gradient

    def draw_enhanced_text(self, draw, text, font, text_color, shadow_color, offset_y=0):
        """Draw text with enhanced shadow and stroke effect"""
        center_x = self.size[0] // 2
        center_y = self.size[1] // 2 + offset_y
        
        # Draw multiple shadows for stronger effect
        shadow_offsets = [(4, 4), (3, 3), (2, 2)]
        for offset in shadow_offsets:
            draw.text(
                (center_x + offset[0], center_y + offset[1]),
                text,
                font=font,
                fill=shadow_color,
                anchor="mm"
            )
        
        # Draw stroke
        stroke_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for offset in stroke_offsets:
            draw.text(
                (center_x + offset[0], center_y + offset[1]),
                text,
                font=font,
                fill=shadow_color,
                anchor="mm"
            )
        
        # Draw main text
        draw.text(
            (center_x, center_y),
            text,
            font=font,
            fill=text_color,
            anchor="mm"
        )

def main():
    print(colored("\n=== Generating Enhanced Test Thumbnails ===", "blue"))
    generator = ThumbnailGenerator()
    generator.generate_test_thumbnails()

if __name__ == "__main__":
    main() 