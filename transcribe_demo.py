#!/usr/bin/env python

import argparse
import asyncio
from datetime import datetime, timedelta
import numpy as np
import os
import speech_recognition as sr
import sys
import torch
import whisper

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(whisper.tokenizer.LANGUAGES.keys()) + sorted(
        [k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in sys.platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    return parser


def load_model(args):
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english and (args.language in set([None, 'en', 'English'])):
        model = model + ".en"
    audio_model = whisper.load_model(
        model, download_root=os.path.join(os.getcwd(), "models"))
    return audio_model


def setup_audio_src(args):
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in sys.platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source)

    return source, recorder


async def input_stream_generator(async_state):
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def record_callback(_, audio: sr.AudioData) -> None:
        wav_data = audio.get_wav_data()
        loop.call_soon_threadsafe(data_queue.put_nowait,
                                  np.frombuffer(wav_data, dtype=async_state.raw_datatype, count=len(wav_data)//2, offset=0).astype(np.float32, order='C') / 32768.0)

    async_state.recorder.listen_in_background(
        async_state.source, record_callback, phrase_time_limit=async_state.record_timeout)
    while True:
        sample = await data_queue.get()
        yield sample


class AsyncState(object):
    def __init__(self, phrase_timeout, record_timeout, task="transcribe", language=None, raw_datatype=np.int16):
        # Setup an object to track our state in
        self.phrase_time = datetime.utcnow()
        self.phrase_timeout = phrase_timeout
        self.record_timeout = record_timeout
        self.global_buffer = np.array([])
        self.fp16 = torch.cuda.is_available()
        self.raw_datatype = raw_datatype
        self.transcription = ['']
        self.task = task
        self.language = language
        self.audio_model = None
        self.source = None
        self.recorder = None


async def process_audio_buffer(async_state):
    async for last_sample in input_stream_generator(async_state):
        now = datetime.utcnow()
        phrase_complete = False
        if now - async_state.phrase_time > timedelta(seconds=async_state.phrase_timeout):
            async_state.global_buffer = np.array([], dtype=np.int16)
            phrase_complete = True
        async_state.phrase_time = now
        async_state.global_buffer = np.concatenate(
            (async_state.global_buffer, last_sample))
        # Read the transcription.
        result = async_state.audio_model.transcribe(
            async_state.global_buffer, fp16=async_state.fp16, task=async_state.task)
        text = result['text'].strip()
        # If we detected a pause between recordings, add a new item to our transcripion.
        # Otherwise edit the existing one.
        if phrase_complete:
            async_state.transcription.append(text)
        else:
            async_state.transcription[-1] = text
        # Clear the console to reprint the updated transcription.
        os.system('cls' if os.name == 'nt' else 'clear')
        for line in async_state.transcription:
            print(line)
        # Flush stdout.
        print('', end='', flush=True)


async def async_main(async_state):
    audio_task = asyncio.create_task(process_audio_buffer(async_state))
    while True:
        await asyncio.sleep(1)

def main():
    args = get_arg_parser().parse_args()
    async_state = AsyncState(
        args.phrase_timeout, args.record_timeout, args.task, args.language)
    async_state.audio_model = load_model(args)
    async_state.source, async_state.recorder = setup_audio_src(args)

    # Cue the user that we're ready to go.
    print("\nModel loaded.\n")

    try:
        asyncio.run(async_main(async_state))
    except KeyboardInterrupt:
        print("\n\nTranscription:")
        for line in async_state.transcription:
            print(line)
        sys.exit('\nInterrupted by user')


if __name__ == "__main__":
    main()
