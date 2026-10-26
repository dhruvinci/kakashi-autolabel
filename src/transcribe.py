"""Local CPU ASR. Outputs speech hypotheses, never visual claims or certified text."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from faster_whisper import WhisperModel
from catalog import now

ROOT=Path(__file__).resolve().parents[1]


def decode_audio(path):
    # PyAV 19 removed an argument used by faster-whisper 1.2.1. Decode via the
    # existing ffmpeg binary into memory, avoiding both that API and another file.
    data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),
                                  '-map','0:a:0','-ac','1','-ar','16000','-f','f32le','pipe:1'],timeout=180)
    return np.frombuffer(data,dtype=np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',default='small.en')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve original ASR output; choose a new version')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    model=WhisperModel(args.model,device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,
                       download_root=str(ROOT/'models'/'whisper'))
    segments,info=model.transcribe(decode_audio(args.video),language='en',beam_size=5,
                                  word_timestamps=True,vad_filter=True,condition_on_previous_text=False)
    rows=[]
    temp=args.output.with_suffix('.partial.jsonl')
    with temp.open('w') as out:
        for seg in segments:
            row=asdict(seg)
            row['evidence_kind']='spoken_asr_unverified'
            rows.append(row)
            out.write(json.dumps(row)+'\n')
            out.flush()
            print(f'{seg.start:.2f}-{seg.end:.2f}: {seg.text}',flush=True)
    with args.video.open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    doc={'model':args.model,'runtime':'faster-whisper','device':'cpu','compute_type':'int8',
         'created_at':now(),'source_sha256':sha,'duration_s':info.duration,'language':info.language,
         'elapsed_s':time.monotonic()-start,'status':'unverified_machine_transcript',
         'warning':'Names, technique terms, speaker identity and time alignment may be wrong.',
         'segments':rows}
    args.output.write_text(json.dumps(doc,indent=2))


if __name__=='__main__':
    main()
