import os
from video import generate_subtitles, process_subtitles, log_info, log_success
from moviepy.editor import ColorClip

def test_emoji_subtitles():
    # Create a test script with emojis
    test_script = """Why do programmers always mix up Halloween and Christmas? 🎃🎄
Because Oct 31 == Dec 25! 🤓
(It's a binary joke, but you'll get it if you're *hex*-perienced.) 😉
Follow for more tech humor! 👍"""

    # Save the script to a file
    os.makedirs('temp/test', exist_ok=True)
    with open('temp/test/test_script.txt', 'w', encoding='utf-8') as f:
        f.write(test_script)

    log_info('Created test script with emojis')
    print(test_script)

    # Generate subtitles
    subtitles_path = generate_subtitles(test_script, None, 'tech_humor')
    log_success(f'Generated subtitles: {subtitles_path}')

    # Create a dummy video
    dummy_video = ColorClip((1080, 1920), color=(0, 0, 0), duration=15)

    # Process subtitles
    final_video = process_subtitles(subtitles_path, dummy_video)

    # Save the video
    output_path = 'temp/test/emoji_subtitle_test.mp4'
    final_video.write_videofile(output_path, fps=30)
    log_success(f'Created test video with emoji subtitles: {output_path}')
    
    return output_path

if __name__ == "__main__":
    test_emoji_subtitles() 