from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

PROJECT_ID = "0f5a1142-14ad-4f90-a7a2-e2812fb91c4a"
ENGINE_VERSION = "AetherScale-0.4.2"


@dataclass(slots=True)
class MotionPacket:
    flow: torch.Tensor
    scene_cuts: torch.Tensor
    confidence: torch.Tensor
    width: int
    height: int
    engine: str
    metadata: Dict[str, Any]


def _device(index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("AetherScale Motion Analysis requires CUDA.")
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device {index}.")
    return torch.device(f"cuda:{index}")


def _analysis_size(h: int, w: int, long_edge: int) -> Tuple[int, int]:
    scale = min(1.0, max(64, int(long_edge)) / max(h, w))
    ah = max(16, int(round(h * scale / 8.0)) * 8)
    aw = max(16, int(round(w * scale / 8.0)) * 8)
    return ah, aw


def _frame_luma(frame_hwc: torch.Tensor, device: torch.device, h: int, w: int) -> torch.Tensor:
    # One frame only. Never move the complete video batch to CUDA.
    x = frame_hwc[..., :3].to(device=device, dtype=torch.float32, non_blocking=False)
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    if tuple(x.shape[-2:]) != (h, w):
        x = F.interpolate(x, size=(h, w), mode="area")
    return 0.2126*x[:,0:1] + 0.7152*x[:,1:2] + 0.0722*x[:,2:3]


def _scene_score(a: torch.Tensor, b: torch.Tensor) -> float:
    delta = (b-a).abs().mean()
    structural = (a.mean()-b.mean()).abs()*0.25 + (a.flatten().std()-b.flatten().std()).abs()*0.15
    return float((delta + structural).clamp(0,1).item())


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=torch.float32),
        torch.arange(w, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = xx[None] + flow[:,0]
    gy = yy[None] + flow[:,1]
    gx = 2*gx/max(w-1,1)-1
    gy = 2*gy/max(h-1,1)-1
    return F.grid_sample(
        img, torch.stack((gx,gy),-1),
        mode="bilinear", padding_mode="border", align_corners=True
    )


def _lk_level(a, b, flow, iterations: int, window: int, eps: float = 1e-4):
    kx = torch.tensor(
        [[-1,0,1],[-2,0,2],[-1,0,1]],
        device=a.device, dtype=torch.float32
    ).view(1,1,3,3)/8
    ky = kx.transpose(-1,-2).contiguous()
    ones = torch.ones((1,1,window,window), device=a.device)
    pad = window//2
    conf = torch.ones_like(a)

    for _ in range(max(1,iterations)):
        bw = _warp(b, flow)
        avg = .5*(a+bw)
        ix = F.conv2d(avg,kx,padding=1)
        iy = F.conv2d(avg,ky,padding=1)
        it = bw-a
        box = lambda x: F.conv2d(x,ones,padding=pad)
        sxx, syy, sxy = box(ix*ix), box(iy*iy), box(ix*iy)
        sxt, syt = box(ix*it), box(iy*it)
        det = sxx*syy-sxy*sxy
        inv = 1/(det+eps)
        du = (-syy*sxt+sxy*syt)*inv
        dv = (sxy*sxt-sxx*syt)*inv
        trace = sxx+syy
        valid = (det>eps)&(trace>eps)
        du = torch.where(valid,du,torch.zeros_like(du)).clamp(-2.5,2.5)
        dv = torch.where(valid,dv,torch.zeros_like(dv)).clamp(-2.5,2.5)
        flow = flow + torch.cat((du,dv),1)
        eig = .5*(trace-torch.sqrt((sxx-syy)**2+4*sxy**2+eps))
        conf = (eig/(eig.mean((2,3),keepdim=True)+eps)).clamp(0,1)
    return flow, conf


def dense_lk_flow(a, b, levels: int, iterations: int, window: int):
    pa,pb=[a],[b]
    for _ in range(1,max(1,levels)):
        if min(pa[-1].shape[-2:])<=32: break
        pa.append(F.avg_pool2d(pa[-1],2,2))
        pb.append(F.avg_pool2d(pb[-1],2,2))
    flow=conf=None
    for level in range(len(pa)-1,-1,-1):
        aa,bb=pa[level],pb[level]
        h,w=aa.shape[-2:]
        if flow is None:
            flow=torch.zeros((1,2,h,w),device=aa.device)
        else:
            oh,ow=flow.shape[-2:]
            flow=F.interpolate(flow,size=(h,w),mode="bilinear",align_corners=True)
            flow[:,0]*=w/max(ow,1); flow[:,1]*=h/max(oh,1)
        flow,conf=_lk_level(aa,bb,flow,iterations,window)
    return flow,conf


def analyze_motion(
    images_bhwc: torch.Tensor,
    *,
    cuda_device: int,
    engine: str,
    quality: str,
    scene_cut_threshold: float,
    reset_on_scene_cut: bool,
    output_device: str,
    motion_mode: str = "scene_cuts_only",
    analysis_long_edge: int = 512,
    storage_precision: str = "float16",
) -> MotionPacket:
    if images_bhwc.ndim != 4:
        raise ValueError(f"Expected IMAGE [T,H,W,C], got {tuple(images_bhwc.shape)}")
    if engine not in ("auto","torch_lk"):
        raise RuntimeError("nvidia_optical_flow is reserved; use auto/torch_lk in this build.")

    n,h,w = int(images_bhwc.shape[0]),int(images_bhwc.shape[1]),int(images_bhwc.shape[2])
    if n<2:
        return MotionPacket(
            torch.empty((0,0,0,2),dtype=torch.float16),
            torch.empty((0,),dtype=torch.bool),
            torch.empty((0,0,0,1),dtype=torch.float16),
            w,h,"none",{"frames":n,"motion_mode":motion_mode}
        )

    if motion_mode=="full_flow":
        ah,aw=h,w
    else:
        ah,aw=_analysis_size(h,w,analysis_long_edge)

    keep_flow = motion_mode != "scene_cuts_only"
    dtype = torch.float16 if storage_precision=="float16" else torch.float32
    flows = torch.empty((n-1,ah,aw,2),dtype=dtype) if keep_flow else torch.empty((0,0,0,2),dtype=dtype)
    confs = torch.empty((n-1,ah,aw,1),dtype=dtype) if keep_flow else torch.empty((0,0,0,1),dtype=dtype)
    cuts = torch.empty((n-1,),dtype=torch.bool)
    scores = torch.empty((n-1,),dtype=torch.float32)

    presets={
        "fast":dict(levels=3,iterations=2,window=5),
        "balanced":dict(levels=4,iterations=3,window=7),
        "quality":dict(levels=5,iterations=5,window=9),
    }
    dev=_device(cuda_device)
    prev=_frame_luma(images_bhwc[0],dev,ah,aw)

    for i in range(n-1):
        curr=_frame_luma(images_bhwc[i+1],dev,ah,aw)
        score=_scene_score(prev,curr)
        cut=score>=float(scene_cut_threshold)
        scores[i]=score; cuts[i]=cut
        if keep_flow:
            if cut and reset_on_scene_cut:
                f=torch.zeros((1,2,ah,aw),device=dev)
                c=torch.zeros((1,1,ah,aw),device=dev)
            else:
                f,c=dense_lk_flow(curr,prev,**presets[quality])
            flows[i].copy_(f[0].permute(1,2,0).to("cpu",dtype=dtype))
            confs[i].copy_(c[0].permute(1,2,0).to("cpu",dtype=dtype))
            del f,c
        del prev
        prev=curr
    del prev

    target=images_bhwc.device if output_device=="same_as_input" else torch.device("cpu")
    if target.type!="cpu":
        flows,confs,cuts=flows.to(target),confs.to(target),cuts.to(target)

    return MotionPacket(
        flows,cuts,confs,w,h,"torch_lk",
        {
            "frames":n,"pairs":n-1,"quality":quality,"motion_mode":motion_mode,
            "scene_cut_threshold":float(scene_cut_threshold),
            "scene_cut_scores":[round(float(x),6) for x in scores],
            "scene_cuts":[bool(x) for x in cuts.cpu()],
            "flow_width":aw,"flow_height":ah,
            "scale_x_to_source":w/aw,"scale_y_to_source":h/ah,
            "flow_storage_dtype":str(dtype).replace("torch.",""),
            "flow_storage_mb":round(flows.numel()*flows.element_size()/1024**2,1),
            "confidence_storage_mb":round(confs.numel()*confs.element_size()/1024**2,1),
            "streaming_pairwise":True,
            "cuda_full_batch_copy":False,
            "direction":"current_to_previous",
        }
    )


def _approx_p95(x: torch.Tensor, max_samples: int = 262144) -> torch.Tensor:
    flat=x.reshape(-1)
    if flat.numel()==0: return torch.tensor(1.0)
    stride=max(1,int(flat.numel())//max_samples)
    sample=flat[::stride][:max_samples].float()
    k=max(1,min(int(sample.numel()),int(round(sample.numel()*.95))))
    return sample.kthvalue(k).values.clamp_min(1e-6)


def flow_visualization(packet: MotionPacket, *, max_preview_frames: int = 8) -> torch.Tensor:
    flow=packet.flow
    if flow.numel()==0:
        # Compact scene-cut timeline instead of a huge fake flow frame.
        width=512
        img=torch.zeros((1,64,width,3),dtype=torch.float32)
        cuts=packet.scene_cuts.detach().cpu()
        if cuts.numel():
            for i,c in enumerate(cuts):
                if bool(c):
                    x=min(width-1,int(round(i/max(1,cuts.numel()-1)*(width-1))))
                    img[0,:,max(0,x-1):min(width,x+2),:]=1.0
        return img

    total=int(flow.shape[0])
    count=min(max_preview_frames,total)
    idx=torch.linspace(0,total-1,count).round().long()
    f=flow.detach().cpu()[idx].float()
    fx,fy=f[...,0],f[...,1]
    mag=torch.sqrt(fx*fx+fy*fy)
    angle=torch.atan2(fy,fx)
    scale=_approx_p95(mag)
    v=(mag/scale).clamp(0,1)
    r=.5+.5*torch.cos(angle)
    g=.5+.5*torch.cos(angle-2.09439510239)
    b=.5+.5*torch.cos(angle+2.09439510239)
    return (torch.stack((r,g,b),-1)*v[...,None]).clamp(0,1)


def neural_vram_plan(width:int,height:int,*,history_frames:int,safety_margin_mb:int,measured_context_mb:int=0):
    px=max(1,int(width))*max(1,int(height))
    visible=(px*8)*2 + px*8 + (px*4)*2 + (px*8)*max(0,int(history_frames))
    visible_mb=visible/1024**2
    return {
        "resolution":[int(width),int(height)],
        "visible_surfaces_mb":round(visible_mb,1),
        "measured_context_mb":int(measured_context_mb),
        "safety_margin_mb":int(safety_margin_mb),
        "estimated_required_free_mb":round(visible_mb+measured_context_mb+safety_margin_mb,1),
    }
