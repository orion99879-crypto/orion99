# Prototype: Motion-Comic MVP — Full-stack Example

This document contains a **minimal, runnable prototype** that demonstrates the full pipeline from a short chapter text → scene breakdown → character-aware image generation → TTS → lip-sync → motion-comic assembly (FFmpeg) → served via a frontend player.

It is intentionally *opinionated* and small: use local or lightweight cloud APIs, and start with a motion-comic approach (pan/zoom + still character renders + lip-synced closeups). The code below is ready-to-copy. Replace API keys and tweak prompts to your taste.

---

## Project structure

```
motion-comic-mvp/
├─ backend/
│  ├─ app.py                 # FastAPI app (API + job enqueue)
│  ├─ worker.py              # Job worker: orchestration pipeline
│  ├─ generate_scene.py      # LLM prompt -> scene JSON helper
│  ├─ sd_client.py           # Simple SD client wrapper (AUTOMATIC1111 or Replicate)
│  ├─ tts_client.py          # TTS wrapper (ElevenLabs or Coqui)
│  ├─ wav2lip_wrapper.py     # Lip-sync helper (calls Wav2Lip)
│  ├─ assemble.py            # FFmpeg assembly helper
│  └─ requirements.txt
├─ frontend/
│  ├─ pages/
│  │  ├─ index.js            # Upload story + generate UI
│  │  └─ player.js           # Simple HLS/MP4 player
│  ├─ package.json
│  └─ next.config.js
├─ demo_assets/
│  └─ sample_character/     # reference images for demo character
└─ README.md
```

---

## 1) Backend: FastAPI + Worker

### `backend/requirements.txt`

```
fastapi
uvicorn[standard]
httpx
pydantic
python-multipart
redis
rq
moviepy
python-dotenv
```

(You may need additional packages for model hosting: `torch`, `transformers`, etc., if self-hosting SD/LLM.)


### `backend/app.py` (FastAPI API)

```python
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import uuid
import os
import rq
import redis
from pathlib import Path
from pydantic import BaseModel

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
rq_conn = redis.from_url(REDIS_URL)
queue = rq.Queue('jobs', connection=rq_conn)

app = FastAPI()

STORAGE = Path(__file__).parent.parent / 'backend_storage'
STORAGE.mkdir(exist_ok=True)

class GenerateRequest(BaseModel):
    title: str
    chapter_text: str
    character_name: str
    character_images: list = []

@app.post('/generate')
async def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    job_folder = STORAGE / job_id
    job_folder.mkdir()
    # store minimal metadata
    (job_folder / 'meta.json').write_text(req.json())
    # enqueue worker job
    queue.enqueue('worker.process_job', job_id)
    return JSONResponse({'job_id': job_id})

@app.get('/status/{job_id}')
async def status(job_id: str):
    job_folder = STORAGE / job_id
    if not job_folder.exists():
        return JSONResponse({'status': 'not_found'})
    if (job_folder / 'done').exists():
        return JSONResponse({'status': 'done', 'result': (job_folder / 'output.mp4').as_posix()})
    if (job_folder / 'progress').exists():
        return JSONResponse({'status': 'processing', 'progress': (job_folder / 'progress').read_text()})
    return JSONResponse({'status': 'queued'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

Notes: This API accepts a JSON body describing the chapter and character (for demo we skip multipart upload of images; you can extend it). It enqueues a job on Redis RQ which calls `worker.process_job`.


### `backend/worker.py` (orchestration worker)

```python
import json
from pathlib import Path
from generate_scene import chapter_to_scenes
from sd_client import generate_character_views
from tts_client import synthesize_dialogue
from wav2lip_wrapper import lip_sync_image_audio
from assemble import assemble_scenes_to_video

STORAGE = Path(__file__).parent / 'backend_storage'

def process_job(job_id: str):
    job_folder = STORAGE / job_id
    meta = json.loads((job_folder / 'meta.json').read_text())

    # 1) LLM: chapter -> scene array
    scenes = chapter_to_scenes(meta['chapter_text'])
    (job_folder / 'scenes.json').write_text(json.dumps(scenes, indent=2))

    # 2) For each character, ensure we have a character embedding / token (DreamBooth flow)
    # For demo: we skip heavy training and use reference images directly.

    # 3) Generate images for scenes
    images = []
    for i, scene in enumerate(scenes):
        img_path = job_folder / f'scene_{i:02d}.png'
        prompt = scene['prompt']
        generate_character_views(prompt, str(img_path))
        images.append({'img': str(img_path), 'dialogue': scene.get('dialogue', [])})

    # 4) TTS: generate audio for each line
    audio_segments = []
    for i, sc in enumerate(images):
        segs = []
        for turn in sc['dialogue']:
            actor = turn['char']
            text = turn['line']
            audio_path = job_folder / f'scene_{i:02d}_{actor}.wav'
            synthesize_dialogue(actor, text, str(audio_path))
            segs.append(str(audio_path))
        audio_segments.append(segs)

    # 5) Lip-sync closeups for the first character appearance in each scene using Wav2Lip
    lip_videos = []
    for i, sc in enumerate(images):
        if audio_segments[i]:
            # pick first audio & use same image
            lip_out = job_folder / f'lip_{i:02d}.mp4'
            lip_sync_image_audio(sc['img'], audio_segments[i][0], str(lip_out))
            lip_videos.append(str(lip_out))
        else:
            lip_videos.append(None)

    # 6) Assemble with FFmpeg
    output = job_folder / 'output.mp4'
    assemble_scenes_to_video(job_folder, images, audio_segments, lip_videos, str(output))

    # 7) mark done
    (job_folder / 'done').write_text('true')

# Expose for RQ import
```


## 2) Helpers: LLM scene breakdown

### `backend/generate_scene.py`

```python
# lightweight LLM wrapper using OpenAI or local LLM
import os
import json

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Example: simplified version that splits text into "scenes" by paragraphs

def chapter_to_scenes(chapter_text: str):
    paras = [p.strip() for p in chapter_text.split('\n\n') if p.strip()]
    scenes = []
    for p in paras[:6]:  # limit to first 6 paras for demo
        scene = {
            'prompt': f"A cinematic anime-style shot: {p[:200]} - close-up, expressive, high detail",
            'dialogue': []
        }
        # naive dialogue detection: lines with - or :
        for line in p.split('\n'):
            if ':' in line:
                char, text = line.split(':', 1)
                scene['dialogue'].append({'char': char.strip().lower().replace(' ', '_'), 'line': text.strip()})
        scenes.append(scene)
    return scenes
```

This is intentionally simple so you can replace it with a richer LLM prompt later. For production, call GPT-4o with a prompt that returns structured JSON.


## 3) SD client wrapper (AUTOMATIC1111 or Replicate)

### `backend/sd_client.py`

```python
# This wrapper targets two common modes:
# 1) Local AUTOMATIC1111 WebUI running at http://127.0.0.1:7860
# 2) Replicate API (if you prefer cloud)

import os
import httpx

AUTOMATIC_URL = os.getenv('AUTOMATIC1111_URL', 'http://127.0.0.1:7860')
REPLICATE_TOKEN = os.getenv('REPLICATE_TOKEN')


def generate_character_views(prompt: str, out_path: str):
    # Prefer local AUTOMATIC1111 if available
    try:
        r = httpx.post(f"{AUTOMATIC_URL}/sdapi/v1/txt2img", json={
            "prompt": prompt,
            "sampler_index": "Euler a",
            "width": 1024,
            "height": 1024,
            "steps": 20
        }, timeout=120.0)
        j = r.json()
        b64 = j['images'][0]
        import base64
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(b64))
        return out_path
    except Exception as e:
        print('AUTOMATIC1111 generation failed, falling back (ensure you have a model available)')
        raise
```

This function uses the local AUTOMATIC1111 WebUI API. If you prefer Replicate or another cloud API, swap the implementation.


## 4) TTS wrapper

### `backend/tts_client.py`

```python
# Basic abstraction for voice cloning + synthesis. This example calls ElevenLabs REST API.
import os
import httpx

ELEVEN_KEY = os.getenv('ELEVENLABS_API_KEY')
BASE = 'https://api.elevenlabs.io/v1'

headers = {
    'xi-api-key': ELEVEN_KEY,
    'Content-Type': 'application/json'
}


def synthesize_dialogue(character_name: str, text: str, out_path: str):
    # For demo: use a shared voice id or map character_name -> voice_id
    voice_id = os.getenv('ELEVEN_VOICE_ID')
    if not voice_id:
        # fallback to default TTS via gTTS (very basic) to avoid needing a paid key
        from gtts import gTTS
        tts = gTTS(text)
        tts.save(out_path)
        return out_path

    payload = {
        'text': text,
        'voice': voice_id
    }
    resp = httpx.post(f"{BASE}/text-to-speech/{voice_id}", json=payload, headers=headers, timeout=60.0)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    return out_path
```

If you can't use ElevenLabs, the code falls back to `gTTS` (Google TTS), which is free but lower quality. For voice cloning, use ElevenLabs voice creation or Coqui fine-tune.


## 5) Lip-sync wrapper (Wav2Lip)

### `backend/wav2lip_wrapper.py`

```python
# Assumes you have Wav2Lip repo available and a script to run it.
import subprocess

def lip_sync_image_audio(image_path: str, audio_path: str, out_video: str):
    # Wav2Lip usage (simplified):
    # python inference.py --checkpoint_path <Wav2Lip_checkpoint> --face <image> --audio <audio> --outfile <out>
    cmd = [
        'python', 'Wav2Lip/inference.py',
        '--checkpoint_path', 'Wav2Lip/checkpoints/wav2lip_gan.pth',
        '--face', image_path,
        '--audio', audio_path,
        '--outfile', out_video
    ]
    subprocess.run(cmd, check=True)
    return out_video
```

You must clone the Wav2Lip repo and have the checkpoint downloaded. For a quick demo, you can skip lip-sync and just overlay audio.


## 6) Assembly with FFmpeg

### `backend/assemble.py`

```python
import subprocess
from pathlib import Path

def assemble_scenes_to_video(job_folder: Path, images: list, audio_segments: list, lip_videos: list, output_path: str):
    # Create a simple slideshow with pan/zoom (using ffmpeg zoompan or concat)
    # For demo: generate short clips from each image, add audio, stitch.
    tmp = job_folder / 'tmp'
    tmp.mkdir(exist_ok=True)
    clips = []
    for i, item in enumerate(images):
        img = item['img']
        scene_audio_files = audio_segments[i]
        # create a 4s video from image with simple zoom using ffmpeg
        clip_path = tmp / f'clip_{i:02d}.mp4'
        cmd_img = [
            'ffmpeg', '-y', '-loop', '1', '-i', img,
            '-vf', "scale=1280:720,zoompan=z='if(lte(on,25),1.0,1.0+0.0005*on)':d=25*4:s=1280x720",
            '-c:v', 'libx264', '-t', '4', '-pix_fmt', 'yuv420p', str(clip_path)
        ]
        subprocess.run(cmd_img, check=True)

        # if there's lip video for this scene, overlay it on top (simple overlay)
        if lip_videos[i]:
            overlay_out = tmp / f'clip_{i:02d}_ov.mp4'
            cmd_ov = [
                'ffmpeg', '-y', '-i', str(clip_path), '-i', lip_videos[i],
                '-filter_complex', "[0:v][1:v] overlay=main_w-320:main_h-240",
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(overlay_out)
            ]
            subprocess.run(cmd_ov, check=True)
            clips.append(str(overlay_out))
        else:
            clips.append(str(clip_path))

    # concatenate all clips
    concat_list = tmp / 'concat.txt'
    concat_list.write_text('\n'.join([f"file '{c}'" for c in clips]))
    out_tmp = job_folder / 'out_tmp.mp4'
    cmd_concat = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(concat_list), '-c', 'copy', str(out_tmp)]
    subprocess.run(cmd_concat, check=True)

    # If there are scene audios, mix them sequentially (simple approach: append silence clips and overlay)
    # For demo: just output out_tmp as final result
    Path(output_path).write_bytes(out_tmp.read_bytes())
    return output_path
```

This is a very simple assembly pipeline; replace with more advanced timeline editor as needed.


## 7) Frontend (Next.js) — minimal

### `frontend/package.json`

```json
{
  "name": "motion-comic-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start -p 3000"
  },
  "dependencies": {
    "next": "13.x",
    "react": "18.x",
    "react-dom": "18.x",
    "axios": "^1.0.0"
  }
}
```


### `frontend/pages/index.js`

```jsx
import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [text, setText] = useState('')
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)

  async function submit() {
    const payload = {
      title: 'Demo',
      chapter_text: text,
      character_name: 'akira',
      character_images: []
    }
    const r = await axios.post('http://localhost:8000/generate', payload)
    setJobId(r.data.job_id)
    setStatus('queued')
  }

  async function poll() {
    if (!jobId) return
    const r = await axios.get(`http://localhost:8000/status/${jobId}`)
    setStatus(r.data.status)
    if (r.data.status === 'done') {
      // show download or player
      window.open(`http://localhost:8000/backend_storage/${jobId}/output.mp4`, '_blank')
    }
  }

  return (
    <div style={{ padding: 40 }}>
      <h1>Motion-Comic MVP</h1>
      <textarea rows={12} cols={80} value={text} onChange={e => setText(e.target.value)} />
      <br />
      <button onClick={submit}>Generate Episode</button>
      <button onClick={poll}>Check Status</button>
      <div>Status: {status}</div>
    </div>
  )
}
```


### `frontend/pages/player.js`

```jsx
export default function Player({ url }) {
  return (
    <div>
      <h2>Player</h2>
      <video controls width="800">
        <source src={url} type="video/mp4" />
      </video>
    </div>
  )
}
```


---

## Final notes & how to run (demo path)

1. **Install dependencies**

```
# backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
```

2. **Run a Redis instance** (default port 6379). Locally: `docker run -p 6379:6379 redis`.

3. **Run FastAPI**

```
cd backend
uvicorn app:app --reload --port 8000
```

4. **Run RQ worker** (so `worker.process_job` is callable by RQ)

```
# from backend directory
rq worker --url redis://localhost:6379/0 jobs
```

(Alternatively, adapt to celery or a simple Python background process.)

5. **Optional but recommended**: Run AUTOMATIC1111 WebUI for local SD generation: https://github.com/AUTOMATIC1111/stable-diffusion-webui

6. **Open frontend**

```
cd frontend
npm run dev
# open http://localhost:3000
```

7. **Paste a short demo text** and hit *Generate Episode*. Poll status until done and open the output MP4.

---

## What this prototype intentionally *doesn't* do (and how to evolve it)

- No DreamBooth training step: for real continuity, add a DreamBooth training flow that produces a small embedding token per user character and use that token in prompts.
- No advanced LLM prompt: integrate GPT-4 (or local Llama 2) to return well-structured JSON scenes.
- No persistent CDN or streaming HLS—this demo outputs MP4 files. Production should transcode to HLS and serve via CDN.
- No worker autoscaling or GPU orchestration—use Kubernetes + KNative or GPU node groups when scaling.

---

## Next steps I can implement for you right now (pick one)
1. Convert the simple LLM scene-splitter to a robust GPT-4 prompt that returns validated JSON scene objects.
2. Add a DreamBooth training script and attach it to `worker.process_job` for automated character embedding creation.
3. Provide a Docker Compose that launches Redis, FastAPI, RQ worker, and a stub SD service for local testing.


---

Good luck — this prototype is designed so you can go from idea → working demo in a few hours (assuming you have the local prerequisites like ffmpeg and, optionally, AUTOMATIC1111 / Wav2Lip cloned). Tweak prompts and pipeline components as you iterate.

