import os
from utils import *
from dotenv import load_dotenv
from tiktok_upload import TikTokUploader
import random

# Load environment variables
load_dotenv("../.env")
# Check if all required environment variables are set
# This must happen before importing video which uses API keys without checking
check_env_vars()

from gpt import *
from video import *
from search import *
from uuid import uuid4
from tiktokvoice import *
from flask_cors import CORS
from termcolor import colored
from youtube import upload_video, test_channel_connection
from apiclient.errors import HttpError
from flask import Flask, request, jsonify
from moviepy.config import change_settings



# Set environment variables
SESSION_ID = os.getenv("TIKTOK_SESSION_ID")
openai_api_key = os.getenv('OPENAI_API_KEY')
change_settings({"IMAGEMAGICK_BINARY": os.getenv("IMAGEMAGICK_BINARY")})

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Constants
HOST = "0.0.0.0"
PORT = 8080
AMOUNT_OF_STOCK_VIDEOS = 5
GENERATING = False

# Set environment variables
TIKTOK_SESSION_ID = os.getenv("TIKTOK_SESSION_ID")  # This is used for TTS and default uploads

# After the existing environment variables
TIKTOK_SESSIONS = {
    'main': os.getenv("TIKTOK_SESSION_ID"),  # Default session
    'business': os.getenv("TIKTOK_SESSION_ID_BUSINESS", ""),
    'gaming': os.getenv("TIKTOK_SESSION_ID_GAMING", ""),
    'tech': os.getenv("TIKTOK_SESSION_ID_TECH", "")
}

# Generation Endpoint
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        # Set global variable
        global GENERATING
        GENERATING = True

        # Clean
        clean_dir("../temp/")
        clean_dir("../subtitles/")


        # Parse JSON
        data = request.get_json()
        paragraph_number = int(data.get('paragraphNumber', 1))  # Default to 1 if not provided
        ai_model = data.get('aiModel')  # Get the AI model selected by the user
        n_threads = data.get('threads')  # Amount of threads to use for video generation
        subtitles_position = data.get('subtitlesPosition')  # Position of the subtitles in the video
        text_color = data.get('color') # Color of subtitle text

        # Get 'useMusic' from the request data and default to False if not provided
        use_music = data.get('useMusic', False)

        # Get 'automateYoutubeUpload' from the request data and default to False if not provided
        automate_youtube_upload = data.get('automateYoutubeUpload', False)

        # Get the ZIP Url of the songs
        songs_zip_url = data.get('zipUrl')

        # Download songs
        if use_music:
            # Downloads a ZIP file containing popular TikTok Songs
            if songs_zip_url:
                fetch_songs(songs_zip_url)
            else:
                # Default to a ZIP file containing popular TikTok Songs
                fetch_songs("https://filebin.net/2avx134kdibc4c3q/drive-download-20240209T180019Z-001.zip")

        # Print little information about the video which is to be generated
        print(colored("[Video to be generated]", "blue"))
        print(colored("   Subject: " + data["videoSubject"], "blue"))
        print(colored("   AI Model: " + ai_model, "blue"))  # Print the AI model being used
        print(colored("   Custom Prompt: " + data["customPrompt"], "blue"))  # Print the AI model being used



        if not GENERATING:
            return jsonify(
                {
                    "status": "error",
                    "message": "Video generation was cancelled.",
                    "data": [],
                }
            )
        
        voice = data["voice"]
        voice_prefix = voice[:2]


        if not voice:
            print(colored("[!] No voice was selected. Defaulting to \"en_us_001\"", "yellow"))
            voice = "en_us_001"
            voice_prefix = voice[:2]


        # Generate a script
        script = generate_script(data["videoSubject"], paragraph_number, ai_model, voice, data["customPrompt"])  # Pass the AI model to the script generation

        # Generate search terms - one per script sentence
        sentences = script.split(". ")
        search_terms = get_search_terms(
            data["videoSubject"], 
            len(sentences),  # Get one term per sentence
            script, 
            ai_model
        )

        # Search for videos - one per search term
        video_urls = []
        min_dur = 5  # Reduced minimum duration since we'll clip them

        for search_term in search_terms:
            if not GENERATING:
                return jsonify({
                    "status": "error",
                    "message": "Video generation was cancelled.",
                    "data": [],
                })
            found_urls = search_for_stock_videos(
                search_term, 
                os.getenv("PEXELS_API_KEY"), 
                10,  # Reduced number of results to check
                min_dur
            )
            if found_urls:
                video_urls.append(found_urls[0])  # Add the first (best) matching video

        # Ensure we have enough videos
        if len(video_urls) < len(sentences):
            print(colored(f"[-] Not enough videos found. Need {len(sentences)}, got {len(video_urls)}", "yellow"))
            # Fill missing videos with duplicates if needed
            while len(video_urls) < len(sentences):
                video_urls.append(random.choice(video_urls))

        # Download videos
        video_paths = []
        for i, video_url in enumerate(video_urls):
            try:
                saved_video_path = save_video(video_url)
                video_paths.append({
                    'path': saved_video_path,
                    'sentence_index': i  # Track which sentence this video matches
                })
            except Exception as e:
                print(colored(f"[-] Could not download video {i+1}: {str(e)}", "red"))

        # Let user know
        print(colored("[+] Videos downloaded!", "green"))

        # Let user know
        print(colored("[+] Script generated!\n", "green"))

        if not GENERATING:
            return jsonify(
                {
                    "status": "error",
                    "message": "Video generation was cancelled.",
                    "data": [],
                }
            )

        # Split script into sentences
        sentences = script.split(". ")

        # Remove empty strings
        sentences = list(filter(lambda x: x != "", sentences))
        paths = []

        # Generate TTS for every sentence
        for sentence in sentences:
            if not GENERATING:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Video generation was cancelled.",
                        "data": [],
                    }
                )
            current_tts_path = f"../temp/{uuid4()}.mp3"
            tts(sentence, voice, filename=current_tts_path)
            audio_clip = AudioFileClip(current_tts_path)
            paths.append(audio_clip)

        # Combine all TTS files using moviepy
        final_audio = concatenate_audioclips(paths)
        tts_path = f"../temp/{uuid4()}.mp3"
        final_audio.write_audiofile(tts_path)

        try:
            subtitles_path = generate_subtitles(audio_path=tts_path, sentences=sentences, audio_clips=paths, voice=voice_prefix)
        except Exception as e:
            print(colored(f"[-] Error generating subtitles: {e}", "red"))
            subtitles_path = None

        # Concatenate videos
        temp_audio = AudioFileClip(tts_path)
        combined_video_path = combine_videos(video_paths, temp_audio.duration, 5, n_threads or 2)

        # Put everything together
        try:
            final_video_path = generate_video(combined_video_path, tts_path, subtitles_path, n_threads or 2, subtitles_position, text_color or "#FFFF00")
        except Exception as e:
            print(colored(f"[-] Error generating final video: {e}", "red"))
            final_video_path = None

        # Define metadata for the video
        title, description, keywords = generate_metadata(data["videoSubject"], script, ai_model)

        print(colored("[-] Metadata for YouTube upload:", "blue"))
        print(colored("   Title: ", "blue"))
        print(colored(f"   {title}", "blue"))
        print(colored("   Description: ", "blue"))
        print(colored(f"   {description}", "blue"))
        print(colored("   Keywords: ", "blue"))
        print(colored(f"  {', '.join(keywords)}", "blue"))

        if automate_youtube_upload:
            try:
                # Get selected YouTube channel
                youtube_channel = data.get('youtubeAccount', 'main')
                
                video_metadata = {
                    'video_path': os.path.abspath(f"../temp/{final_video_path}"),
                    'title': title,
                    'description': description,
                    'category': "28",  # Science & Technology
                    'keywords': ",".join(keywords),
                    'privacy_status': "public",
                }

                # Upload to selected YouTube channel
                video_response = upload_video(
                    **video_metadata,
                    channel=youtube_channel  # Add channel parameter
                )
                
                if video_response:
                    print(colored(f"[+] Video uploaded to YouTube channel: {youtube_channel}", "green"))
            except Exception as e:
                print(colored(f"[-] YouTube upload error: {str(e)}", "red"))

        # TikTok Upload
        if data.get('automateTikTokUpload', False):
            try:
                # Get selected account type
                account_type = data.get('tiktokAccount', 'main')
                
                # Get the corresponding session ID
                session_map = {
                    'main': os.getenv('TIKTOK_SESSION_ID'),
                    'business': os.getenv('TIKTOK_SESSION_ID_BUSINESS'),
                    'gaming': os.getenv('TIKTOK_SESSION_ID_GAMING'),
                    'tech': os.getenv('TIKTOK_SESSION_ID_TECH')
                }
                
                tiktok_session_id = session_map.get(account_type)
                
                if not tiktok_session_id:
                    print(colored(f"[-] TikTok session ID not found for: {account_type}", "red"))
                else:
                    print(colored(f"[+] Uploading to TikTok channel: {account_type}", "blue"))
                    uploader = TikTokUploader(tiktok_session_id)
                    
                    # Generate TikTok-appropriate tags
                    tiktok_tags = uploader.generate_tiktok_tags(title, ",".join(keywords))
                    
                    # Upload to TikTok
                    tiktok_response = uploader.upload_video(
                        video_path=f"../temp/{final_video_path}",
                        title=title[:150],  # TikTok title length limit
                        tags=tiktok_tags
                    )
                    
                    if tiktok_response:
                        print(colored("[+] Video successfully uploaded to TikTok!", "green"))
                    else:
                        print(colored("[-] Failed to upload to TikTok", "red"))
                        
            except Exception as e:
                print(colored(f"[-] TikTok upload error: {str(e)}", "red"))

        video_clip = VideoFileClip(f"../temp/{final_video_path}")
        if use_music:
            # Select a random song
            song_path = choose_random_song()

            # Add song to video at 30% volume using moviepy
            original_duration = video_clip.duration
            original_audio = video_clip.audio
            song_clip = AudioFileClip(song_path).set_fps(44100)

            # Set the volume of the song to 10% of the original volume
            song_clip = song_clip.volumex(0.1).set_fps(44100)

            # Add the song to the video
            comp_audio = CompositeAudioClip([original_audio, song_clip])
            video_clip = video_clip.set_audio(comp_audio)
            video_clip = video_clip.set_fps(30)
            video_clip = video_clip.set_duration(original_duration)
            video_clip.write_videofile(f"../{final_video_path}", threads=n_threads or 1)
        else:
            video_clip.write_videofile(f"../{final_video_path}", threads=n_threads or 1)


        # Let user know
        print(colored(f"[+] Video generated: {final_video_path}!", "green"))

        # Stop FFMPEG processes
        if os.name == "nt":
            # Windows
            os.system("taskkill /f /im ffmpeg.exe")
        else:
            # Other OS
            os.system("pkill -f ffmpeg")

        GENERATING = False

        # Return JSON
        return jsonify(
            {
                "status": "success",
                "message": "Video generated! See MoneyPrinter/output.mp4 for result.",
                "data": final_video_path,
            }
        )
    except Exception as err:
        print(colored(f"[-] Error: {str(err)}", "red"))
        return jsonify(
            {
                "status": "error",
                "message": f"Could not retrieve stock videos: {str(err)}",
                "data": [],
            }
        )


@app.route("/api/cancel", methods=["POST"])
def cancel():
    print(colored("[!] Received cancellation request...", "yellow"))

    global GENERATING
    GENERATING = False

    return jsonify({"status": "success", "message": "Cancelled video generation."})


@app.route("/api/tiktok-accounts", methods=["GET"])
def get_tiktok_accounts():
    # Only return accounts that have session IDs configured
    available_accounts = [
        {
            'id': account,
            'name': account.replace('_', ' ').title(),
            'active': bool(session)
        }
        for account, session in TIKTOK_SESSIONS.items() 
        if session  # Only include accounts with valid session IDs
    ]
    return jsonify({
        "status": "success",
        "accounts": available_accounts
    })


@app.route("/api/test-youtube", methods=["GET"])
def test_youtube():
    """Test YouTube channel connections."""
    try:
        # Get specific channel to test
        channel = request.args.get('channel', None)
        
        if channel:
            # Test specific channel
            success = test_channel_connection(channel)
            return jsonify({
                "status": "success" if success else "error",
                "message": f"YouTube {channel} channel test {'successful' if success else 'failed'}",
                "channel": channel
            })
        else:
            # Test all channels
            results = {}
            channels = ['main', 'business', 'gaming', 'tech']
            
            for ch in channels:
                results[ch] = test_channel_connection(ch)
            
            return jsonify({
                "status": "success",
                "results": results,
                "all_success": all(results.values())
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error testing YouTube channels: {str(e)}"
        })


if __name__ == "__main__":

    # Run Flask App
    app.run(debug=True, host=HOST, port=PORT)
