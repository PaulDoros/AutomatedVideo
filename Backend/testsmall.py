from moviepy.editor import TextClip
clip = TextClip("Hello, world!", fontsize=70, color="white", font="Arial-Bold", method="label")
clip = clip.set_duration(3)
clip.write_videofile("test.mp4", fps=24)
