from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
from termcolor import colored
import requests
from io import BytesIO
import random
import emoji
import numpy as np
import json
from datetime import datetime

class ThumbnailGenerator:
    def __init__(self):
        self.size = (1280, 720)  # YouTube thumbnail size
        self.vertical_size = (1080, 1920)  # Shorts thumbnail size
        self.assets_dir = "assets"
        self.fonts_dir = f"{self.assets_dir}/fonts"
        self.backgrounds_dir = f"{self.assets_dir}/backgrounds"
        
        # Create necessary directories
        for directory in [self.fonts_dir, self.backgrounds_dir, "test_thumbnails", "temp/thumbnails"]:
            os.makedirs(directory, exist_ok=True)
        
        # Ensure fonts exist and are valid
        self.montserrat_bold = f"{self.fonts_dir}/Montserrat-Bold.ttf"
        self.roboto_regular = f"{self.fonts_dir}/Roboto-Regular.ttf"
        
        # Always run font setup to verify fonts
        print(colored("Verifying fonts...", "blue"))
        try:
            # Try direct import first
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from Backend.setup_fonts import download_fonts
            download_fonts()
        except ImportError:
            try:
                # Try relative import
                import setup_fonts
                setup_fonts.download_fonts()
            except ImportError:
                print("Warning: Could not import setup_fonts module. Font verification skipped.")
                print("Fonts may not be available for thumbnail generation.")
        
        # Setup API access for image sources
        self.pexels_api_key = os.environ.get('PEXELS_API_KEY', None)
        self.pixabay_api_key = os.environ.get('PIXABAY_API_KEY', None)
        
        if not self.pexels_api_key and not self.pixabay_api_key:
            print(colored("Warning: No API keys found for Pexels or Pixabay. Using local backgrounds only.", "yellow"))
        
    def fetch_relevant_image(self, content_type, keywords=None):
        """Fetch a relevant image from Pexels/Pixabay or use local fallback"""
        # Try Pexels first if API key is available
        if self.pexels_api_key:
            try:
                # Determine search query based on content type and keywords
                search_terms = {
                    'tech_humor': ['programming', 'computer', 'coding', 'technology'],
                    'coding_tips': ['coding', 'developer', 'programming', 'software'],
                    'life_hack': ['productivity', 'organization', 'life hack', 'tips'],
                    'food_recipe': ['cooking', 'food', 'kitchen', 'ingredients'],
                    'fitness_motivation': ['fitness', 'workout', 'exercise', 'gym']
                }
                
                # Use provided keywords or fall back to content type
                if keywords:
                    search_query = "+".join(keywords.split()[:2])
                else:
                    search_options = search_terms.get(content_type, ['background'])
                    search_query = random.choice(search_options)
                
                # Make API request to Pexels
                headers = {'Authorization': self.pexels_api_key}
                params = {
                    'query': search_query,
                    'orientation': 'portrait',
                    'per_page': 10
                }
                
                response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
                data = response.json()
                
                if 'photos' in data and data['photos']:
                    # Get a random photo from results
                    photo = random.choice(data['photos'])
                    image_url = photo['src']['portrait']  # Use portrait size for shorts
                    
                    # Download the image
                    img_response = requests.get(image_url)
                    img = Image.open(BytesIO(img_response.content))
                    
                    # Resize to fit shorts format (9:16)
                    img = self.resize_and_crop_image(img, self.vertical_size)
                    
                    print(colored(f"✓ Downloaded image from Pexels: {image_url}", "green"))
                    return img
                else:
                    print(colored("No images found on Pexels, trying Pixabay...", "yellow"))
            except Exception as e:
                print(colored(f"Error fetching image from Pexels: {str(e)}", "red"))
                print(colored("Trying Pixabay as fallback...", "yellow"))
        
        # Try Pixabay if Pexels failed or no Pexels API key
        if self.pixabay_api_key:
            try:
                # Use provided keywords or fall back to content type
                if keywords:
                    search_query = "+".join(keywords.split()[:2])
                else:
                    search_options = {
                        'tech_humor': ['technology', 'computer', 'programming'],
                        'ai_money': ['artificial intelligence', 'business', 'technology'],
                        'baby_tips': ['baby', 'parenting', 'family'],
                        'quick_meals': ['food', 'cooking', 'meal'],
                        'fitness_motivation': ['fitness', 'workout', 'exercise']
                    }
                    options = search_options.get(content_type, ['background'])
                    search_query = random.choice(options)
                
                # Make API request to Pixabay
                params = {
                    'key': self.pixabay_api_key,
                    'q': search_query,
                    'image_type': 'photo',
                    'orientation': 'vertical',
                    'per_page': 10
                }
                
                response = requests.get('https://pixabay.com/api/', params=params)
                data = response.json()
                
                if 'hits' in data and data['hits']:
                    # Get a random photo from results
                    photo = random.choice(data['hits'])
                    image_url = photo['largeImageURL']
                    
                    # Download the image
                    img_response = requests.get(image_url)
                    img = Image.open(BytesIO(img_response.content))
                    
                    # Resize to fit shorts format (9:16)
                    img = self.resize_and_crop_image(img, self.vertical_size)
                    
                    print(colored(f"✓ Downloaded image from Pixabay: {image_url}", "green"))
                    return img
                else:
                    print(colored("No images found on Pixabay, using local fallback", "yellow"))
            except Exception as e:
                print(colored(f"Error fetching image from Pixabay: {str(e)}", "red"))
        
        # Fall back to local background if both APIs fail
        print(colored("Using local background as fallback", "yellow"))
        return self.get_local_background(content_type)
            
    def get_local_background(self, content_type):
        """Get a background from local assets or generate a gradient"""
        try:
            # Check for local backgrounds by content type
            content_dir = f"{self.backgrounds_dir}/{content_type}"
            if os.path.exists(content_dir):
                bg_files = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if bg_files:
                    bg_path = os.path.join(content_dir, random.choice(bg_files))
                    img = Image.open(bg_path)
                    return self.resize_and_crop_image(img, self.vertical_size)
            
            # Generic backgrounds as fallback
            if os.path.exists(self.backgrounds_dir):
                bg_files = [f for f in os.listdir(self.backgrounds_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if bg_files:
                    bg_path = os.path.join(self.backgrounds_dir, random.choice(bg_files))
                    img = Image.open(bg_path)
                    return self.resize_and_crop_image(img, self.vertical_size)
                    
            # If no images found, create a gradient based on content type
            gradients = {
                'tech_humor': ['#3a1c71', '#d76d77'],
                'coding_tips': ['#0F2027', '#2C5364'],
                'life_hack': ['#11998e', '#38ef7d'],
                'food_recipe': ['#EB3349', '#F45C43'],
                'fitness_motivation': ['#4e54c8', '#8f94fb']
            }
            
            colors = gradients.get(content_type, ['#5433FF', '#20BDFF'])
            return self.create_gradient(colors, size=self.vertical_size)
            
        except Exception as e:
            print(colored(f"Error getting local background: {str(e)}", "red"))
            # Create a simple gradient as last resort
            return self.create_gradient(['#5433FF', '#20BDFF'], size=self.vertical_size)
            
    def resize_and_crop_image(self, img, target_size):
        """Resize and crop an image to target size with center focus"""
        # Calculate target aspect ratio
        target_ratio = target_size[0] / target_size[1]
        
        # Get current size
        width, height = img.size
        current_ratio = width / height
        
        # Resize to match target height or width while maintaining aspect ratio
        if current_ratio > target_ratio:
            # Image is wider than needed, resize based on height
            new_height = target_size[1]
            new_width = int(current_ratio * new_height)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Crop center
            left = (new_width - target_size[0]) // 2
            img = img.crop((left, 0, left + target_size[0], target_size[1]))
            
        else:
            # Image is taller than needed, resize based on width
            new_width = target_size[0]
            new_height = int(new_width / current_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Crop center
            top = (new_height - target_size[1]) // 2
            img = img.crop((0, top, target_size[0], top + target_size[1]))
            
        return img

    def create_gradient(self, colors, size=(1280, 720)):
        """Create a gradient background"""
        gradient = Image.new('RGB', size)
        draw = ImageDraw.Draw(gradient)
        
        for i in range(size[1]):
            r = int((size[1] - i) * int(colors[0][1:3], 16) / size[1] + i * int(colors[1][1:3], 16) / size[1])
            g = int((size[1] - i) * int(colors[0][3:5], 16) / size[1] + i * int(colors[1][3:5], 16) / size[1])
            b = int((size[1] - i) * int(colors[0][5:7], 16) / size[1] + i * int(colors[1][5:7], 16) / size[1])
            draw.line([(0, i), (size[0], i)], fill=(r, g, b))
            
        return gradient

    def draw_enhanced_text(self, draw, text, font, text_color, shadow_color, offset_y=0, offset_x=0):
        """Draw text with enhanced shadow and stroke effect"""
        center_x = self.size[0] // 2 + offset_x
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

    def generate_thumbnail(self, content_type, output_path=None):
        """Generate a professional thumbnail for a specific content type"""
        try:
            # Get template based on content type
            template = self.get_template(content_type)
            
            # Try to get thumbnail title from script cache
            thumbnail_title = None
            script_file = f"cache/scripts/{content_type}_latest.json"
            if os.path.exists(script_file):
                try:
                    with open(script_file, 'r') as f:
                        data = json.load(f)
                        if data.get('thumbnail_title'):
                            thumbnail_title = data['thumbnail_title']
                            print(colored(f"Using thumbnail title: {thumbnail_title}", "green"))
                except Exception as e:
                    print(colored(f"Error reading script: {str(e)}", "yellow"))
            
            # If we don't have a thumbnail title, use a generic engaging phrase
            if not thumbnail_title:
                thumbnail_title = get_engagement_phrases(content_type)
            
            # Update template with the thumbnail title
            template['text'] = thumbnail_title
            
            # Set output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"temp/thumbnails/{content_type}_{timestamp}.jpg"
                os.makedirs("temp/thumbnails", exist_ok=True)
            
            # Fetch or create background image
            try:
                # Try to get a relevant image from Pexels or Pixabay
                background = self.fetch_relevant_image(content_type, thumbnail_title)
            except Exception as e:
                print(colored(f"Error getting image from API: {str(e)}", "yellow"))
                # Fall back to local background or gradient
                background = self.get_local_background(content_type)
            
            # Apply slightly darker overlay for better text contrast
            overlay = Image.new('RGBA', background.size, (0, 0, 0, 80))  # Semi-transparent black
            background = Image.alpha_composite(background.convert('RGBA'), overlay)
            
            # Convert back to RGB for drawing
            background = background.convert('RGB')
            
            draw = ImageDraw.Draw(background)
            
            # Determine font sizes based on title length
            title_len = len(thumbnail_title)
            title_font_size = 100 if title_len < 20 else (80 if title_len < 30 else 60)
            
            # Load fonts with adjusted sizes
            try:
                title_font = ImageFont.truetype(self.montserrat_bold, title_font_size)
            except Exception as e:
                print(colored(f"Error loading fonts: {str(e)}", "red"))
                print(colored("Using default font as fallback", "yellow"))
                title_font = ImageFont.load_default()
            
            # Draw title with enhanced shadow - centered in upper half
            self.draw_enhanced_text(
                draw, 
                thumbnail_title, 
                title_font, 
                '#FFFFFF',  # White
                '#000000',  # Black shadow
                offset_y=-300
            )
            
            # Add slight vignette effect for more professional look
            background = self.add_vignette(background)
            
            # Save with high quality
            background.save(output_path, 'JPEG', quality=95)
            print(colored(f"✓ Generated thumbnail: {output_path}", "green"))
            
            return output_path
        
        except Exception as e:
            print(colored(f"Error generating thumbnail: {str(e)}", "red"))
            return None
    
    def get_emoji_image(self, emoji_char, size=200):
        """Convert emoji character to transparent PNG image"""
        try:
            # Create a transparent image
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Try to find a suitable font for emoji rendering
            emoji_font_size = int(size * 0.8)
            emoji_font = None
            
            # Try different system fonts that support emoji
            font_options = [
                ('Segoe UI Emoji', 'seguiemj.ttf'),  # Windows
                ('Apple Color Emoji', 'AppleColorEmoji.ttf'),  # Mac
                ('Noto Color Emoji', 'NotoColorEmoji.ttf'),  # Linux
                ('Twemoji', 'Twemoji.ttf')  # Twitter emoji font
            ]
            
            for font_name, font_file in font_options:
                try:
                    emoji_font = ImageFont.truetype(font_name, emoji_font_size)
                    break
                except Exception:
                    continue
            
            if emoji_font is None:
                # Fallback to included fonts or system default
                try:
                    emoji_font = ImageFont.truetype(self.roboto_regular, emoji_font_size)
                except Exception:
                    emoji_font = ImageFont.load_default()
            
            # Draw the emoji centered
            draw.text((size//2, size//2), emoji_char, font=emoji_font, fill=(255, 255, 255, 255), anchor="mm")
            
            return img
        except Exception as e:
            print(colored(f"Error creating emoji image: {str(e)}", "yellow"))
            return None
    
    def add_vignette(self, img, intensity=0.3):
        """Add a subtle vignette effect to the image"""
        # Create a radial gradient for the vignette
        width, height = img.size
        radius = max(width, height) / 2
        center_x, center_y = width / 2, height / 2
        
        # Create a mask for the vignette
        mask = Image.new('L', img.size, 255)
        draw = ImageDraw.Draw(mask)
        
        # Draw the radial gradient
        for i in range(width):
            for j in range(height):
                # Calculate distance from center
                distance = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                # Calculate intensity (stronger at edges)
                intensity_factor = min(1, distance / radius)
                intensity_value = int(255 * (1 - intensity_factor * intensity))
                mask.putpixel((i, j), intensity_value)
        
        # Apply the mask
        enhanced = ImageEnhance.Brightness(img)
        vignette = enhanced.enhance(1.0 - intensity * 0.5)
        
        # Blend the original and darkened image using the mask
        result = Image.composite(vignette, img, mask)
        
        return result

    def get_template(self, content_type):
        """Get thumbnail template based on content type"""
        templates = {
            'tech_humor': {
                'text': 'Tech Humor',
                'colors': ['#3a1c71', '#d76d77'],
                'emoji': '🤣💻'
            },
            'ai_money': {
                'text': 'AI Money',
                'colors': ['#0F2027', '#2C5364'],
                'emoji': '💰🤖'
            },
            'baby_tips': {
                'text': 'Baby Tips',
                'colors': ['#11998e', '#38ef7d'],
                'emoji': '👶🍼'
            },
            'quick_meals': {
                'text': 'Quick Meals',
                'colors': ['#EB3349', '#F45C43'],
                'emoji': '🍲⏱️'
            },
            'fitness_motivation': {
                'text': 'Fitness Motivation',
                'colors': ['#4e54c8', '#8f94fb'],
                'emoji': '💪🏋️'
            }
        }
        
        # Return template for content type or default
        return templates.get(content_type, {
            'text': 'Engaging Content',
            'colors': ['#5433FF', '#20BDFF'],
            'emoji': '🔥✨'
        })

    def generate_test_thumbnails(self):
        """Generate test thumbnails for each content type"""
        try:
            # Create test thumbnails directory if it doesn't exist
            os.makedirs("test_thumbnails", exist_ok=True)
            
            # List of content types to generate thumbnails for
            content_types = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
            
            print(colored("\n=== Generating Test Thumbnails ===", "blue"))
            
            for content_type in content_types:
                try:
                    output_path = f"test_thumbnails/{content_type}.jpg"
                    result = self.generate_thumbnail(content_type, output_path)
                    
                    if result:
                        print(colored(f"✓ Generated test thumbnail for {content_type}", "green"))
                    else:
                        print(colored(f"✗ Failed to generate test thumbnail for {content_type}", "red"))
                except Exception as e:
                    print(colored(f"✗ Error generating thumbnail for {content_type}: {str(e)}", "red"))
            
            print(colored("=== Test Thumbnail Generation Complete ===", "blue"))
            
        except Exception as e:
            print(colored(f"✗ Error generating test thumbnails: {str(e)}", "red"))

def get_engagement_phrases(content_type):
    """Return engaging transition phrases based on content type"""
    phrases = {
        'tech_humor': [
            "Pro coding secret...", 
            "Dev hack unlocked...", 
            "Programming magic...",
            "Tech genius move...",
            "Code wizardry..."
        ],
        'coding_tips': [
            "Code faster with...",
            "10x developer tip...",
            "Clean code secret...",
            "Debugging lifesaver...",
            "GitHub pro move..."
        ],
        'life_hack': [
            "Life-changing trick...",
            "Mind-blowing hack...",
            "Save hours with...",
            "Genius shortcut...",
            "Game changer..."
        ],
        'food_recipe': [
            "Chef's secret...",
            "Flavor explosion...",
            "Cooking magic...",
            "Tasty twist...",
            "Kitchen genius..."
        ],
        'fitness_motivation': [
            "Workout secret...",
            "Body transformation...",
            "Fitness breakthrough...",
            "Energy boost...",
            "Muscle builder..."
        ]
    }
    
    # Get phrases for the specific content type or use general ones
    content_phrases = phrases.get(content_type, [
        "Amazing trick...",
        "Must-know secret...",
        "Game changer...",
        "You won't believe...",
        "Trending now..."
    ])
    
    return random.choice(content_phrases)

def main():
    print(colored("\n=== Generating Enhanced Test Thumbnails ===", "blue"))
    generator = ThumbnailGenerator()
    generator.generate_test_thumbnails()

if __name__ == "__main__":
    main() 