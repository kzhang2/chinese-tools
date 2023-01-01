import os
import re
import numpy as np
import subprocess
import whisper
import time 


base_dir = os.path.expanduser("~/Videos/chibi-maruko-chinese")
vid_file_names = os.listdir(base_dir)
vid_file_names = sorted(vid_file_names, key=lambda x: int(re.search("#\d+", x).group(0)[1:]))

model = whisper.load_model("small")
for vfn in vid_file_names:
    vfp = f"{base_dir}/{vfn}"
    name = vfn.split(".")[0]
    afp = f"audio/{name}.wav"
    atp = f"audio-transcriptions/{name}.txt"
    print(afp)
    if not os.path.isfile(afp):
        subprocess.run(["ffmpeg", "-i", vfp, "-ac", "1", "-f", "wav", afp, "-y", "-hide_banner", "-loglevel", "panic"])
    if not os.path.isfile(atp):
        start = time.time()
        result = model.transcribe(afp)
        print(result["text"])
        with open(atp, "w") as f:
            f.write(result["text"])
        print(time.time() - start)
    break
