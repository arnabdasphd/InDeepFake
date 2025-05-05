import locale
locale.getpreferredencoding = lambda: "UTF-8"
import os
import urllib.request
import urllib.error
from tqdm import tqdm
import shutil

def conditional_download(url, download_file_path, redownload=False):
    base_path = os.path.dirname(download_file_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if os.path.exists(download_file_path) and not redownload:
        return
    if os.path.exists(download_file_path) and redownload:
        os.remove(download_file_path)
    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError:
        return
    with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))
        except urllib.error.URLError:
            return

def download_models(base_path, redownload=False):
    model_urls = [
        ("https://huggingface.co/SWivid/E2-TTS/resolve/main/E2TTS_Base/model_1200000.pt", f"{base_path}/ckpts/E2TTS_Base/model_1200000.pt"),
        ("https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt", f"{base_path}/ckpts/F5TTS_Base/model_1200000.pt"),
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin", f"{base_path}/ckpts/vocos-mel-24khz/pytorch_model.bin"),
        ("https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/config.yaml", f"{base_path}/ckpts/vocos-mel-24khz/config.yaml")
    ]
    for url, path in model_urls:
        conditional_download(url, path, redownload=redownload)

def download_whisper_model(base_path, redownload=False):
    model_urls = [
        ("https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/resolve/main/config.json", f"{base_path}/faster-whisper-large-v3-turbo-ct2/config.json"),
        ("https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/resolve/main/model.bin", f"{base_path}/faster-whisper-large-v3-turbo-ct2/model.bin"),
        ("https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/resolve/main/preprocessor_config.json", f"{base_path}/faster-whisper-large-v3-turbo-ct2/preprocessor_config.json"),
        ("https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/resolve/main/tokenizer.json", f"{base_path}/faster-whisper-large-v3-turbo-ct2/tokenizer.json"),
        ("https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2/resolve/main/vocabulary.json", f"{base_path}/faster-whisper-large-v3-turbo-ct2/vocabulary.json"),
    ]
    for url, path in model_urls:
        conditional_download(url, path, redownload=redownload)

base_path = "/content"
install_path = f"{base_path}/F5-TTS"
if os.path.exists(install_path):
    shutil.rmtree(install_path)

!git clone https://github.com/NeuralFalconYT/F5-TTS.git
download_models(install_path, redownload=False)
download_whisper_model(base_path, redownload=True)

!pip install -r requirements.txt
!pip install pydub==0.25.1
!pip install pysrt==1.1.2
!pip install faster-whisper
!pip install ctranslate2==4.4.0

from IPython.display import clear_output
clear_output()

import time
import os
os.kill(os.getpid(), 9)






# base_path = "."
base_path = "/content"







import os
import re
import gc
import uuid
import time
import torch
import shutil
import librosa
import numpy as np
import torchaudio
import subprocess
from tqdm.notebook import tqdm
from pydub import AudioSegment
from einops import rearrange
from vocos import Vocos
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from model import CFM, UNetT, DiT
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
)
import nltk
from nltk.tokenize import sent_tokenize
from IPython.display import clear_output

nltk.download('punkt')
install_path = f"{base_path}/F5-TTS"
os.chdir(install_path)

def is_gpu_memory_over_limit(limit_gb=14.5):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, text=True)
    memory_used_mb_list = result.stdout.strip().splitlines()
    for memory_used_mb in memory_used_mb_list:
        memory_used_gb = int(memory_used_mb) / 1024.0
        if memory_used_gb > limit_gb:
            return True
    return False

def load_whisper():
    global whisper_pipe, whisper_model
    try:
        if whisper_pipe is not None:
            del whisper_pipe
            whisper_pipe = None
        if whisper_model is not None:
            del whisper_model
            whisper_model = None
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
    except:
        pass
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper_pipe, whisper_model

def initialize_vocoder_and_model(exp_name="F5TTS_Base", ckpt_step=1200000, device="cuda", target_sample_rate=24000,
                                  n_mel_channels=100, hop_length=256, dataset_name="Emilia_ZH_EN", tokenizer="pinyin",
                                  ode_method='euler', use_ema=True):
    global vocos, model
    try:
        if vocos is not None:
            del vocos
            vocos = None
        if model is not None:
            del model
            model = None
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
    except:
        pass
    if exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    elif exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

    vocos_local_path = "./ckpts/vocos-mel-24khz"
    vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
    vocos.load_state_dict(state_dict)
    vocos.eval()

    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)
    ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"
    model = load_checkpoint(model, ckpt_path, device, use_ema=use_ema)
    return vocos, model

def merge_audio(audio_list, save_path):
    merged_audio = AudioSegment.empty()
    for audio_file in audio_list:
        audio_segment = AudioSegment.from_wav(audio_file)
        merged_audio += audio_segment
    merged_audio.export(save_path, format="wav")

def chunks_sentences(paragraph, join_limit=2):
    sentences = sent_tokenize(paragraph)
    return [' '.join(sentences[i:i + join_limit]) for i in range(0, len(sentences), join_limit)]

def clean_file_name(file_path):
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    cleaned = re.sub(r'[^a-zA-Z\d]+', '_', file_name)
    clean_file_name = re.sub(r'_+', '_', cleaned).strip('_')
    random_uuid = uuid.uuid4().hex[:6]
    return os.path.join(os.path.dirname(file_path), clean_file_name + f"_{random_uuid}" + file_extension)

def tts_file_name(text):
    if text.endswith("."):
        text = text[:-1]
    text = text.lower().strip().replace(" ", "_")
    truncated_text = text[:25] if len(text) > 25 else text if len(text) > 0 else "empty"
    random_string = uuid.uuid4().hex[:8].upper()
    file_name = f"{base_path}/f5_Voice/{truncated_text}_{random_string}.wav"
    return clean_file_name(file_name)

def is_audio_duration_greater_than_30s(audio_path, max_duration=30):
    try:
        audio = AudioSegment.from_file(audio_path)
    except:
        return False
    return len(audio) / 1000 > max_duration

def trim_audio(input_audio_path, max_duration=30):
    output_folder = f"{base_path}/trim_audio"
    os.makedirs(output_folder, exist_ok=True)
    audio = AudioSegment.from_file(input_audio_path)
    trimmed_audio = audio[:max_duration * 1000] if len(audio) / 1000 > max_duration else audio
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    output_file = f"{output_folder}/{base_name}_trimmed.wav"
    trimmed_audio.export(output_file, format="wav")
    return output_file

def process_audio(reference_audio, max_duration=15):
    global old_trim_audio
    if is_audio_duration_greater_than_30s(reference_audio, max_duration):
        f_base_name = os.path.basename(reference_audio)
        f_name, _ = os.path.splitext(f_base_name)
        trimmed_audio_path = f"{base_path}/trim_audio/{f_name}_trimmed.wav"
        if old_trim_audio != trimmed_audio_path:
            reference_audio = trim_audio(reference_audio, max_duration)
            old_trim_audio = reference_audio
        else:
            reference_audio = trimmed_audio_path
    return reference_audio

def voice_clone(reference_audio, text, output_dir="", target_sample_rate=24000, remove_silence=False,
                fix_duration=None, chunks=0, exp_name="F5TTS_Base", progress_bar=True):
    global device, old_audio_path, old_ref_text, old_exp_name
    global whisper_pipe, whisper_model, vocos, model, seed

    reference_audio = process_audio(reference_audio, max_duration=15)

    if old_exp_name != exp_name:
        vocos, model = initialize_vocoder_and_model(device=device, exp_name=exp_name)
        old_exp_name = exp_name

    if is_gpu_memory_over_limit(limit_gb=14.5):
        whisper_pipe, whisper_model = load_whisper()
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        vocos, model = initialize_vocoder_and_model(device=device, exp_name=exp_name)

    final_audio_path = tts_file_name(text)
    output_dir = f"{base_path}/f5_Voice/temp"
    nfe_step, cfg_strength = 32, 2.0
    ode_method, speed, target_rms = 'euler', 1.0, 0.1
    hop_length, tokenizer = 256, "pinyin"
    sway_sampling_coef = -1.

    if old_audio_path == reference_audio:
        ref_text = old_ref_text
    else:
        ref_text = whisper_pipe(reference_audio)['text'].strip()
        old_audio_path, old_ref_text = reference_audio, ref_text

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    audio, sr = torchaudio.load(reference_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)
    audio = audio.to(device)

    prompts = [text] if chunks == 0 else chunks_sentences(text, join_limit=chunks)
    audio_list = []
    number_of_prompts = len(prompts)
    iterable = tqdm(enumerate(prompts), total=number_of_prompts, desc="Processing Prompts") if progress_bar else enumerate(prompts)

    for i, text in iterable:
        gen_text = text.strip()
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list) if tokenizer == "pinyin" else [text_list]
        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None and number_of_prompts == 1:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            zh_pause_punc = r"。，、；：？！"
            ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
            gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave *= rms / target_rms

        generated_wave_tensor = torch.tensor(generated_wave).unsqueeze(0) if isinstance(generated_wave, np.ndarray) else generated_wave
        save_audio_path = f"{output_dir}/{i}.wav"
        torchaudio.save(save_audio_path, generated_wave_tensor, target_sample_rate)
        audio_list.append(save_audio_path)

    if len(audio_list) == 1:
        shutil.copy(audio_list[-1], final_audio_path)
    elif len(audio_list) > 1:
        merge_audio(audio_list, final_audio_path)
    else:
        final_audio_path = None

    return final_audio_path

whisper_pipe = whisper_model = vocos = model = seed = None
whisper_pipe, whisper_model = load_whisper()
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
vocos, model = initialize_vocoder_and_model(device=device, exp_name="F5TTS_Base")
old_audio_path = old_ref_text = old_trim_audio = ""
old_exp_name = "F5TTS_Base"
os.makedirs(f"{base_path}/f5_Voice", exist_ok=True)
clear_output()
print("Model Import Complete")





import os
from google.colab import files
from IPython.display import clear_output

def upload_audio():
    upload_folder = f"{base_path}/user_upload"
    os.makedirs(upload_folder, exist_ok=True)
    os.chdir(upload_folder)
    uploaded = files.upload()
    os.chdir(install_path)

    audio_name_list = []
    for fn in uploaded.keys():
        file_path = f"{upload_folder}/{fn}"
        save_file_path = clean_file_name(file_path)
        os.rename(file_path, save_file_path)
        audio_name_list.append(save_file_path)

    audio_extensions = ('.wav', '.mp3')
    valid_audio_files = [f for f in audio_name_list if f.lower().endswith(audio_extensions)]

    clear_output()
    return valid_audio_files[0] if valid_audio_files else None

upload_audio()




Reference_Audio_Path = '/content/nnn.wav'
TTS_Text = 'Hello World!'
Remove_Silence = True
Split_Sentences = 3 if len(TTS_Text) > 300 else 0
Choose_Model = "F5TTS_Base"
seed = None

cloned_voice_path = voice_clone(
    Reference_Audio_Path,
    TTS_Text,
    remove_silence=Remove_Silence,
    chunks=Split_Sentences,
    exp_name=Choose_Model
)

clear_output()
print(f"TTS saved at {cloned_voice_path}")
Audio(cloned_voice_path)
