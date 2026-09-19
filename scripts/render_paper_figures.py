"""Render source-backed manuscript figures as editable SVG and print-ready PNG.

Requires Pillow. Inputs are the reviewed aggregate data in docs/figures;
no model, external service, credentials or private training rows are loaded.
"""
from html import escape
import argparse
import json
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
FONTS={'latin':'C:/Windows/Fonts/times.ttf','latin_bold':'C:/Windows/Fonts/timesbd.ttf',
       'cjk':'C:/Windows/Fonts/simsun.ttc','cjk_bold':'C:/Windows/Fonts/simhei.ttf'}

class Canvas:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.image=Image.new('RGB',(w*2,h*2),'white')
        self.draw=ImageDraw.Draw(self.image)
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    def line(self,points,color='#29343e',width=2,dash=False):
        pairs=[(x*2,y*2) for x,y in points]
        if dash:
            for a,b in zip(pairs,pairs[1:]):
                steps=max(1,int(((a[0]-b[0])**2+(a[1]-b[1])**2)**.5/12))
                for i in range(0,steps,2):
                    self.draw.line([(a[0]+(b[0]-a[0])*i/steps,a[1]+(b[1]-a[1])*i/steps),(a[0]+(b[0]-a[0])*min(i+1,steps)/steps,a[1]+(b[1]-a[1])*min(i+1,steps)/steps)],fill=color,width=width*2)
        else:self.draw.line(pairs,fill=color,width=width*2)
        extra=' stroke-dasharray="8 5"' if dash else ''
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')
    def box(self,x,y,w,h,fill='#f5f8fb',border='#3e5b76'):
        self.draw.rectangle((x*2,y*2,(x+w)*2,(y+h)*2),fill=fill,outline=border,width=3)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{border}" stroke-width="1.5"/>')
    def text(self,x,y,text,size=25,anchor='middle',color='#17222b',bold=False):
        chunks=re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',text)
        fonts=[ImageFont.truetype(FONTS[('latin' if chunk.isascii() else 'cjk')+('_bold' if bold else '')],size*2) for chunk in chunks]
        widths=[self.draw.textlength(t,font=f) for t,f in zip(chunks,fonts)]
        left=x*2-(sum(widths)/2 if anchor=='middle' else sum(widths) if anchor=='end' else 0)
        for t,f,w in zip(chunks,fonts,widths):
            self.draw.text((left,y*2),t,font=f,fill=color,anchor='lt');left+=w
        weight='bold' if bold else 'normal'
        self.svg.append(f'<text x="{x}" y="{y+size*.85}" text-anchor="{anchor}" font-family="Times New Roman, SimSun, serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>')
    def arrow(self,points,color='#3e5b76'):
        self.line(points,color,3)
        x,y=points[-1];px,py=points[-2]
        if x>px:self.line([(x-10,y-6),(x,y),(x-10,y+6)],color,3)
        elif x<px:self.line([(x+10,y-6),(x,y),(x+10,y+6)],color,3)
        elif y>py:self.line([(x-6,y-10),(x,y),(x+6,y-10)],color,3)
        else:self.line([(x-6,y+10),(x,y),(x+6,y+10)],color,3)
    def save(self,name):
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f'{name}.svg').write_text(''.join(self.svg)+ '</svg>',encoding='utf-8')
        self.image.save(OUT/f'{name}.png',dpi=(300,300))

def architecture():
    c=Canvas(1260,420)
    c.box(15,68,195,125);c.text(112,86,'操作者与原文件',25,bold=True)
    c.text(112,126,'论文 / 图片',24);c.text(112,159,'数值能带 / 结构',23)
    c.box(260,68,260,125);c.text(390,86,'宿主 AI 模型',26,bold=True)
    c.text(390,124,'视觉识别、查文献',24);c.text(390,158,'选择工具、解释结果',24)
    c.box(573,90,190,82);c.text(668,103,'MCP 客户端',25,bold=True);c.text(668,138,'调用与返回',23)
    c.box(825,68,417,125);c.text(1033,84,'本地 MCP 服务：17 个工具',26,bold=True)
    c.text(1033,124,'证据校验 → 数值分析 → 导出',24);c.text(1033,158,'不调用历史模型，不自行训练',23)
    c.arrow([(210,131),(255,131)]);c.arrow([(520,131),(568,131)]);c.arrow([(763,131),(820,131)])
    c.arrow([(1033,67),(1033,26),(390,26),(390,63)])
    c.text(708,0,'返回数据或缺失证据提示',23)
    c.box(15,237,350,66,'#fff8e9','#8d7439');c.text(190,248,'本地适配器：文件 → 附件编号',23);c.text(190,278,'记录文件摘要，不让 AI 编造字节',21)
    c.arrow([(112,193),(112,233)],'#8d7439')
    c.arrow([(365,269),(799,269),(799,187),(821,187)],'#8d7439');c.text(566,239,'编号与摘要',22)
    c.box(825,237,417,66);c.text(1033,248,'可选 PDF / OCR 后备解析',23);c.text(1033,278,'子进程内存限制、超时与大小检查',21)
    c.arrow([(1033,236),(1033,198)])
    c.line([(15,336),(1242,336)],'#999999',1,dash=True)
    c.text(630,348,'历史离线分支：AFLOW 数值数据 → 遮蔽预训练 → 分类微调 → 留出测试',24)
    c.text(630,384,'历史权重与当前 MCP 主链路断开；未知内容不补成科学事实',23,bold=True)
    c.save('mcp_architecture')

def training():
    d=json.loads((OUT/'paper_metrics.json').read_bytes());c=Canvas(1260,395)
    c.text(320,8,'(a) 历史训练过程（54 轮）',26,bold=True)
    c.text(961,8,'(b) 历史留出测试（11,987 条）',26,bold=True)
    x0,x1,y0,y1=84,608,68,299
    maximum=max(max(p['training_loss'],p['validation_loss']) for p in d['history'])*1.04
    for i in range(5):
        y=y1-i*(y1-y0)/4;c.line([(x0,y),(x1,y)],'#dddddd',1)
        c.text(x0-12,y-12,f'{maximum*i/4:.1f}',22,'end')
    for field,color,dash in [('training_loss','#215c8b',False),('validation_loss','#b46b18',True)]:
        c.line([(x0+(p['epoch']-1)/53*(x1-x0),y1-p[field]/maximum*(y1-y0)) for p in d['history']],color,2,dash)
    c.line([(x0,y0),(x0,y1),(x1,y1)])
    for tick in [1,10,20,30,40,54]:c.text(x0+(tick-1)/53*(x1-x0),310,str(tick),22)
    c.text(45,37,'损失（无量纲）',22,'start');c.text(353,343,'训练轮数',23)
    c.line([(326,55),(361,55)],'#215c8b',2);c.text(365,43,'训练',22,'start')
    c.line([(452,55),(487,55)],'#b46b18',2,True);c.text(493,43,'验证',22,'start')
    lx,rx,top,bottom=738,1208,68,299
    for val in range(0,101,25):
        y=bottom-val/100*(bottom-top);c.line([(lx,y),(rx,y)],'#dddddd',1);c.text(lx-12,y-12,str(val),22,'end')
    for x,key,label,fill in [(785,'accuracy','准确率','#215c8b'),(1029,'macro_f1','宏平均 F1','#aebdca')]:
        value=d['classification'][key]*100;h=value/100*(bottom-top)
        c.box(x,bottom-h,114,h,fill,'#244c6b');c.text(x+57,bottom-h-30,f'{value:.2f}%',24,bold=True);c.text(x+57,312,label,24)
    c.line([(lx,top),(lx,bottom),(rx,bottom)]);c.text(717,37,'指标（%）',22,'start')
    c.text(630,373,'历史数值模型的记录；不是当前 MCP、图片识别或 AI 对照实验的准确率',23)
    c.save('historical_training_evaluation')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key,default in FONTS.items():parser.add_argument('--'+key.replace('_','-')+'-font',default=default)
    args=parser.parse_args()
    FONTS={key:getattr(args,key+'_font') for key in FONTS}
    if any(not Path(path).is_file() for path in FONTS.values()):
        parser.error('Provide installed fonts using --cjk-font, --cjk-bold-font, --latin-font and --latin-bold-font')
    architecture();training();print('Rendered architecture and historical training figures.')
