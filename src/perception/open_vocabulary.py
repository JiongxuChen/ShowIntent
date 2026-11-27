"""Text-conditioned detector adapter; A's original implementation stays intact."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
from .run import PerceptionModels


def query_groups(config):
    queries=config['queries'];count=config['classes_per_prompt']
    if not queries or type(count) is not int or count<1:raise ValueError('Invalid prompt groups')
    labels=[q['label'] for q in queries]
    if len(labels)!=len(set(labels)):raise ValueError('Duplicate canonical labels')
    for q in queries:
        if not q['label'].strip() or q['text']!=q['text'].strip().lower() or not q['text'] or '.' in q['text']:
            raise ValueError('Queries must be explicit lowercase phrases without periods')
    result=[]
    for begin in range(0,len(queries),count):
        caption='';spans=[]
        for q in queries[begin:begin+count]:
            start=len(caption);caption+=q['text'];spans.append((q['label'],start,len(caption)));caption+='. '
        result.append((caption,spans))
    return result


def token_spans(offsets,spans):
    groups=[]
    for label,start,end in spans:
        indexes=[i for i,(a,b) in enumerate(offsets) if b>a and max(a,start)<min(b,end)]
        if not indexes:raise ValueError('Phrase has no tokens: '+label)
        groups.append((label,indexes))
    return groups


def normalized_to_pixels(box,width,height):
    cx,cy,bw,bh=[float(v) for v in box]
    if not all(math.isfinite(v) for v in [cx,cy,bw,bh]) or bw<=0 or bh<=0:return None
    x1=max(0.,min(1.,cx-bw/2))*width;y1=max(0.,min(1.,cy-bh/2))*height
    x2=max(0.,min(1.,cx+bw/2))*width;y2=max(0.,min(1.,cy+bh/2))*height
    return (x1,y1,x2,y2) if x1<x2 and y1<y2 else None


class GroundingDinoModels(PerceptionModels):
    def __init__(self,model_path,confidence,use_hands,*,device='0',prompt_config):
        import torch
        from transformers import AutoConfig,AutoProcessor,AutoModelForZeroShotObjectDetection
        self.prompt_config=json.loads(Path(prompt_config).read_text())
        if confidence!=self.prompt_config['box_threshold']:raise ValueError('CLI/config box thresholds differ')
        self.confidence=confidence;self.device=torch.device('cuda:'+device if str(device).isdigit() else device)
        self.processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
        config=AutoConfig.from_pretrained(str(model_path),local_files_only=True)
        # Use the supported PyTorch fallback; no JIT CUDA extension compilation.
        config.disable_custom_kernels=True
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path),config=config,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32).to(self.device).eval()
        self.groups=[]
        self.forward_calls=0
        for caption,spans in query_groups(self.prompt_config):
            tokens=self.processor.tokenizer(caption,return_offsets_mapping=True,truncation=False)
            if len(tokens['input_ids'])>config.max_text_len:raise ValueError('Prompt exceeds model token limit; reduce classes_per_prompt')
            self.groups.append((caption,token_spans(tokens['offset_mapping'],spans),tokens['input_ids']))
        self.yolo=SimpleNamespace(names={i:q['label'] for i,q in enumerate(self.prompt_config['queries'])})
        self.hands=None
        if use_hands:
            import mediapipe as mp
            self.hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=.5,min_tracking_confidence=.5)

    def detect_objects(self,frame,imgsz):
        import cv2
        import torch
        from PIL import Image
        h,w=frame.shape[:2];scale=imgsz/max(h,w)
        resized=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale)))) if scale!=1 else frame
        image=Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB));rows=[]
        for caption,groups,ids in self.groups:
            inputs=self.processor(images=image,text=caption,return_tensors='pt',do_resize=False)
            if inputs['input_ids'][0].tolist()!=ids:raise ValueError('Processor changed prompt tokenization')
            inputs=inputs.to(self.device)
            with torch.inference_mode():
                outputs=self.model(**inputs)
                self.forward_calls+=1
                probabilities=outputs.logits[0].sigmoid()
                # Fix one canonical name per query using only phrase tokens;
                # punctuation and other prompts cannot become object labels.
                scores=torch.stack([probabilities[:,indexes].max(dim=-1).values for _,indexes in groups],dim=-1)
                best,labels=scores.max(dim=-1)
                selected=torch.nonzero(best>=max(self.confidence,self.prompt_config['text_threshold'])).flatten()
                boxes=outputs.pred_boxes[0,selected].cpu().tolist();values=best[selected].cpu().tolist();names=labels[selected].cpu().tolist()
            for box,score,index in zip(boxes,values,names):
                pixels=normalized_to_pixels(box,w,h)
                if pixels is not None:rows.append((groups[index][0],pixels,float(score)))
        return rows
