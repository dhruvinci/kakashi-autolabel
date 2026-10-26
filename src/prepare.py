"""Reproducible timestamped visual evidence from one master, without subclip copies."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def extract_frames(video, dest, interval_s=0.5):
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError('Use a new evidence version; do not overwrite frozen frames')
    if interval_s <= 0:
        raise ValueError('Positive frame interval required')
    dest.mkdir(parents=True)
    # showinfo reports the actual decoded PTS; never infer timestamps from filenames.
    vf = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_s})',showinfo"
    command = ['ffmpeg','-hide_banner','-nostdin','-threads','2','-i',str(video),
               '-map','0:v:0','-vf',vf,'-fps_mode','vfr','-q:v','3','-threads','2',str(dest/'%06d.jpg')]
    log_path = dest.parent/(dest.name+'-decode.log')
    with log_path.open('w') as log:
        subprocess.run(command, stdout=subprocess.DEVNULL,stderr=log,check=True,timeout=1800)
    times = [float(m.group(1)) for m in re.finditer(r'\bn:\s*\d+.*?\bpts_time:([^\s]+)',log_path.read_text())]
    images = sorted(dest.glob('*.jpg'))
    if not times or len(times) != len(images):
        raise ValueError('Decoded image/timestamp count mismatch')
    rows = []
    for i,(path,t) in enumerate(zip(images,times),1):
        with Image.open(path) as im:
            width,height = im.size
        rows.append({'frame_id':f'f{i:06d}','time_s':t,'path':str(path.resolve()),
                     'width':width,'height':height,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (dest.parent/(dest.name+'.json')).write_text(json.dumps(rows,indent=2))
    return rows


def make_sheets(frames, dest, *, per_sheet=12, columns=3, width=426):
    dest = Path(dest)
    dest.mkdir(parents=True,exist_ok=True)
    font = ImageFont.load_default(size=18)
    height = 240
    label = 28
    panels = []
    for index,start in enumerate(range(0,len(frames),per_sheet),1):
        batch = frames[start:start+per_sheet]
        sheet = Image.new('RGB',(columns*width,((len(batch)+columns-1)//columns)*(height+label)),(20,22,25))
        draw = ImageDraw.Draw(sheet)
        for j,frame in enumerate(batch):
            x,y = (j%columns)*width,(j//columns)*(height+label)
            with Image.open(frame['path']) as original:
                thumb = original.copy()
                thumb.thumbnail((width,height))
                sheet.paste(thumb,(x+(width-thumb.width)//2,y+(height-thumb.height)//2))
            t=frame['time_s']
            draw.text((x+6,y+height+4),f"{frame['frame_id']}  {int(t//60):02d}:{t%60:06.3f}",font=font,fill='white')
        path=dest/f'sheet-{index:03d}.jpg'
        sheet.save(path,quality=91)
        panels.append({'path':str(path.resolve()),'frame_ids':[r['frame_id'] for r in batch],
                       'start_s':batch[0]['time_s'],'end_s':batch[-1]['time_s'],
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (dest/'index.json').write_text(json.dumps(panels,indent=2))
    return panels


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=0.5)
    args=parser.parse_args()
    if shutil.disk_usage(args.video.parent).free < 31_000_000_000:
        raise RuntimeError('Evidence preparation would encroach on free-space reserve')
    rows=extract_frames(args.video,args.output/'frames',args.interval)
    sheets=make_sheets(rows,args.output/'sheets')
    overview=[]
    for target in range(0,int(rows[-1]['time_s'])+1,10):
        overview.append(min(rows,key=lambda r:abs(r['time_s']-target)))
    make_sheets(overview,args.output/'overview',per_sheet=12)
    print(json.dumps({'frames':len(rows),'sheets':len(sheets),'first_time_s':rows[0]['time_s'],
                      'last_time_s':rows[-1]['time_s']}))


if __name__=='__main__':
    main()
