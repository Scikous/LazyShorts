import os
import shutil
import subprocess
import gc
import numpy as np
import torch
import torchaudio.transforms as T
import soundfile as sf
import pysubs2
from pysubs2 import SSAEvent, SSAStyle, Color
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from enum import Enum

# --- CONFIGURATION ---
class VerticalMode(Enum):
    CROP_CENTER = "crop_center"
    BLUR_BG = "blur_bg"
    SPLIT_SCREEN = "split_screen"

class Config:
    TEMP_DIR = Path("temp_process")
    OUTPUT_DIR = Path("output")
    
    WHISPER_MODEL = "large-v3-turbo"
    DEMUCS_MODEL = "htdemucs_ft"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
    
    MAX_WORKERS = 3 
    
    SAMPLE_RATE_DEMUCS = 44100
    SAMPLE_RATE_WHISPER = 16000

    # (x, y, width, height)
    FACECAM_COORDS = (1500, 600, 420, 480) 

Config.TEMP_DIR.mkdir(exist_ok=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True)

# --- UTILS ---
def cleanup_gpu():
    if Config.DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

def get_video_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    return float(subprocess.check_output(cmd).strip())

def run_ffmpeg_with_progress(cmd: List[str], total_duration: float, desc: str = "Processing"):
    """Runs FFmpeg with a live tqdm progress bar."""
    cmd_with_progress = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    process = subprocess.Popen(
        cmd_with_progress, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True,
        startupinfo=startupinfo
    )
    
    with tqdm(total=total_duration, desc=desc, unit="s", bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}s") as pbar:
        for line in process.stdout:
            if "out_time_us=" in line:
                try:
                    us = int(line.split("=")[1].strip())
                    sec = us / 1_000_000
                    pbar.n = min(sec, total_duration)
                    pbar.refresh()
                except ValueError:
                    pass
                    
    process.wait()
    if process.returncode != 0:
        err = process.stderr.read()
        raise RuntimeError(f"FFmpeg Error: {err}")

# --- STEP 1: SILENCE REMOVAL (OPTIMIZED) ---
class SilenceRemover:
    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
        process = subprocess.run(cmd, capture_output=True, check=True)
        # Added .copy() to fix PyTorch warning
        audio_np = np.frombuffer(process.stdout, np.float32).copy()
        return torch.from_numpy(audio_np)

    @staticmethod
    def get_timestamps(audio_tensor: torch.Tensor, min_silence_ms: int = 500) -> List[dict]:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
        (get_speech_ts, _, _, _, _) = utils
        return get_speech_ts(audio_tensor, model, threshold=0.5, min_speech_duration_ms=100, min_silence_duration_ms=min_silence_ms, return_seconds=True)

    @staticmethod
    def render_batch(args):
        idx, batch, input_video, temp_dir = args
        filter_parts = []
        concat_inputs = []
        
        # --- FAST SEEK OPTIMIZATION ---
        # We find the start time of the first segment in this batch.
        # We subtract a small buffer (5s) to ensure we hit a keyframe before the segment starts.
        # Then we tell FFmpeg to seek (-ss) to that point BEFORE reading the file.
        # This prevents FFmpeg from decoding the entire video up to this point.
        
        global_start = batch[0][0]
        seek_offset = max(0, global_start - 5.0) 
        
        for j, (start, end) in enumerate(batch):
            # Adjust timestamps relative to the seek offset
            rel_start = start - seek_offset
            rel_end = end - seek_offset
            
            # Sanity check to prevent negative timestamps due to float precision
            rel_start = max(0, rel_start)
            
            filter_parts.append(f"[0:v]trim=start={rel_start}:end={rel_end},setpts=PTS-STARTPTS[v{j}];[0:a]atrim=start={rel_start}:end={rel_end},asetpts=PTS-STARTPTS[a{j}];")
            concat_inputs.append(f"[v{j}][a{j}]")
        
        full_filter = "".join(filter_parts) + f"{''.join(concat_inputs)}concat=n={len(batch)}:v=1:a=1[outv][outa]"
        filter_file = temp_dir / f"filter_{idx}.txt"
        with open(filter_file, "w") as f: f.write(full_filter)
            
        output_chunk = temp_dir / f"chunk_{idx:04d}.mp4"
        
        cmd = [
            "ffmpeg", "-y", "-v", "error", 
            "-hwaccel", "cuda",
            "-ss", str(seek_offset),  # <--- FAST SEEK HERE
            "-i", str(input_video),
            "-filter_complex_script", str(filter_file), 
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "h264_nvenc", 
            "-preset", "p1",      
            "-cq", "26",          
            "-c:a", "aac", "-b:a", "192k", 
            str(output_chunk)
        ]
        subprocess.run(cmd, check=True)
        return output_chunk

    @classmethod
    def process(cls, input_video: str, output_video: str, padding_ms: int = 200):
        print(f"--- Step 1: Removing Silence ---")
        wav = cls.load_audio(input_video)
        timestamps = cls.get_timestamps(wav)
        
        total_duration = get_video_duration(input_video)
        segments = []
        for ts in timestamps:
            start = max(0, ts['start'] - (padding_ms / 1000))
            end = min(total_duration, ts['end'] + (padding_ms / 1000))
            if segments and start < segments[-1][1]: segments[-1][1] = max(segments[-1][1], end)
            else: segments.append([start, end])

        batch_size = 60 
        batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
        
        temp_dir = Config.TEMP_DIR / "silence_chunks"
        if temp_dir.exists(): shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        worker_args = [(i, b, input_video, temp_dir) for i, b in enumerate(batches)]
        chunk_files = [None] * len(batches)

        print(f"   > Rendering {len(batches)} batches...")
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {executor.submit(cls.render_batch, arg): arg[0] for arg in worker_args}
            with tqdm(total=len(batches), desc="Cutting Chunks") as pbar:
                for future in futures:
                    idx = futures[future]
                    chunk_files[idx] = future.result()
                    pbar.update(1)

        concat_list = temp_dir / "merge_list.txt"
        valid_chunks = [c for c in chunk_files if c]
        with open(concat_list, "w") as f:
            for chunk in valid_chunks: f.write(f"file '{chunk.resolve()}'\n")

        est_duration = sum([seg[1] - seg[0] for seg in segments])
        merge_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output_video)]
        run_ffmpeg_with_progress(merge_cmd, est_duration, desc="Merging")
        
        shutil.rmtree(temp_dir)
        cleanup_gpu()

# --- STEP 2: VOCAL EXTRACTION ---
class VocalExtractor:
    @staticmethod
    def load_audio_tensor(path: str, sr: int = 44100) -> torch.Tensor:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        # Added .copy() to fix PyTorch warning
        audio_np = np.frombuffer(out, np.float32).copy()
        return torch.from_numpy(audio_np).reshape([-1, 2]).T.to(Config.DEVICE)

    @classmethod
    def process(cls, input_video: str, output_audio: str):
        print(f"--- Step 2: Extracting Vocals ---")
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = get_model(Config.DEMUCS_MODEL)
        model.to(Config.DEVICE)

        wav = cls.load_audio_tensor(input_video)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(model, wav[None], device=Config.DEVICE, shifts=0, split=True, overlap=0.25, progress=True)
        
        vocals_stereo = sources[0, 3] * ref.std() + ref.mean()
        vocals_mono = vocals_stereo.mean(dim=0, keepdim=True)
        
        resampler = T.Resample(Config.SAMPLE_RATE_DEMUCS, Config.SAMPLE_RATE_WHISPER).to(Config.DEVICE)
        vocals_16k = resampler(vocals_mono)

        audio_np = vocals_16k.cpu().numpy().T
        sf.write(output_audio, audio_np, Config.SAMPLE_RATE_WHISPER)

        del model, sources, wav, vocals_stereo, vocals_mono, vocals_16k
        cleanup_gpu()

# --- STEP 3 & 4: TRANSCRIPTION & SUBTITLES ---
class SubtitleGenerator:
    @staticmethod
    def transcribe(audio_path: str):
        print(f"--- Step 3: Transcribing ---")
        from faster_whisper import WhisperModel
        
        model = WhisperModel(Config.WHISPER_MODEL, device=Config.DEVICE, compute_type=Config.COMPUTE_TYPE)
        
        segments, _ = model.transcribe(
            str(audio_path),
            beam_size=10, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True
        )
        
        result_segments = list(segments)
        cleanup_gpu()
        return result_segments

    @staticmethod
    def create_hormozi_ass(segments, output_file: str, is_vertical: bool = False):
        print(f"--- Step 4: Generating ASS ---")
        subs = pysubs2.SSAFile()
        
        if is_vertical:
            subs.info["PlayResX"] = "1080"
            subs.info["PlayResY"] = "1920"
            margin_v = 400
            font_size = 90
        else:
            subs.info["PlayResX"] = "1920"
            subs.info["PlayResY"] = "1080"
            margin_v = 100
            font_size = 80

        style = SSAStyle(
            fontname="Impact",
            fontsize=font_size,
            primarycolor=Color(255, 255, 0),
            secondarycolor=Color(0, 0, 0),
            outlinecolor=Color(0, 0, 0),
            backcolor=Color(0, 0, 0, 0),
            bold=True,
            alignment=5,
            outline=3,
            shadow=0,
            marginv=margin_v 
        )
        subs.styles["Hormozi"] = style

        all_words = []
        for s in segments:
            if s.words: all_words.extend(s.words)

        buffer = []
        for word in all_words:
            buffer.append(word)
            txt = word.word.strip()
            is_end = txt[-1] in ".?!," if txt else False
            
            if len(buffer) >= 2 or is_end:
                start_ms = int(buffer[0].start * 1000)
                end_ms = int(buffer[-1].end * 1000)
                text_content = " ".join([w.word.strip() for w in buffer]).upper()
                subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))
                buffer = []
        
        if buffer:
            start_ms = int(buffer[0].start * 1000)
            end_ms = int(buffer[-1].end * 1000)
            text_content = " ".join([w.word.strip() for w in buffer]).upper()
            subs.events.append(SSAEvent(start=start_ms, end=end_ms, text=text_content, style="Hormozi"))

        subs.save(output_file)

# --- STEP 5: BURN IN (FIXED) ---
def burn_subtitles(
    video_path: str, 
    ass_path: str, 
    output_path: str, 
    mode: VerticalMode = VerticalMode.CROP_CENTER,
    facecam_coords: Tuple[int, int, int, int] = None
):
    print(f"--- Step 5: Burning Subtitles (Mode: {mode.name}) ---")
    
    total_duration = get_video_duration(video_path)
    
    if mode == VerticalMode.CROP_CENTER:
        vf_chain = "crop=ih*(9/16):ih,scale=1080:1920"
        
    elif mode == VerticalMode.BLUR_BG:
        # FIX: Added 'format=yuv420p' to fix green overlay bug.
        # FIX: Changed scale to 'iw/4:ih/4' for safer math.
        vf_chain = (
            "split=2[bg][fg];"
            "[bg]format=yuv420p,scale=iw/4:ih/4,boxblur=10,scale=1080:1920:flags=bilinear[bg_blurred];"
            "[fg]scale=1080:-1[fg_scaled];"
            "[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2"
        )
        
    elif mode == VerticalMode.SPLIT_SCREEN:
        if not facecam_coords:
            raise ValueError("Facecam coordinates required.")
        fx, fy, fw, fh = facecam_coords
        
        vf_chain = (
            f"split=2[main][face];"
            f"[main]scale=1080:960:force_original_aspect_ratio=decrease[main_v];"
            f"[face]crop={fw}:{fh}:{fx}:{fy},scale=1080:960:force_original_aspect_ratio=decrease[face_v];"
            f"color=s=1080x1920:c=black[canvas];"
            f"[canvas][main_v]overlay=(W-w)/2:(960-h)/2[p1];"
            f"[p1][face_v]overlay=(W-w)/2:960+(960-h)/2"
        )
    else:
        vf_chain = "null"

    if vf_chain == "null":
        full_vf = f"ass={ass_path}, setsar=1"
    else:
        full_vf = f"{vf_chain}[v_final];[v_final]ass={ass_path}, setsar=1"

    cmd = [
        "ffmpeg", "-y", 
        "-hwaccel", "cuda",
        "-i", str(video_path),
        "-filter_complex", full_vf,
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "23",
        "-c:a", "copy",
        str(output_path)
    ]
    
    run_ffmpeg_with_progress(cmd, total_duration, desc=f"Burning ({mode.name})")

# --- MAIN CONTROLLER ---
def main(input_file: str, mode_9_16: VerticalMode = VerticalMode.BLUR_BG):
    input_path = Path(input_file)

    cut_video = Config.OUTPUT_DIR / "1_silence_removed.mp4"
    vocals_audio = Config.TEMP_DIR / "2_vocals_16k_mono.wav"
    
    # # 1. Cut Silence
    # SilenceRemover.process(input_path, cut_video)
    
    # # 2. Extract Vocals
    # VocalExtractor.process(cut_video, vocals_audio)
    
    # 3. Transcribe
    segments = SubtitleGenerator.transcribe(vocals_audio)
    
    # 4 & 5. Generate Short-Form
    print(f"\n--- Creating Short-Form (9:16) Version [{mode_9_16.name}] ---")
    subs_9_16 = Config.TEMP_DIR / "subs_9_16.ass"
    final_9_16 = Config.OUTPUT_DIR / f"{input_path.stem}_final_9_16.mp4"
    
    SubtitleGenerator.create_hormozi_ass(segments, subs_9_16, is_vertical=True)
    
    burn_subtitles(
        cut_video, 
        subs_9_16, 
        final_9_16, 
        mode=mode_9_16,
        facecam_coords=Config.FACECAM_COORDS
    )

if __name__ == "__main__":
    main("sloppyshorts-1.mp4", mode_9_16=VerticalMode.BLUR_BG)
 