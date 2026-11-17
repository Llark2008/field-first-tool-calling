"""A separate loopback monitor for the isolated training-mechanism pilot."""
from . import live_progress
from .ablation_progress import HTML as BASE_HTML

HTML=BASE_HTML.replace('学习率 × 配对数据','损失归一化与 Adam 偏差修正').replace(
    '开发筛选实验 · R0 原数据/原学习率 · R1 降学习率 · R2 改配对数据 · R3 两者结合',
    '机制开发实验 · M0 原配方 · M1 监督 token 等权 · M2 Adam 偏差修正').replace(
    "training_R1:'R1 训练',training_R2:'R2 训练',training_R3:'R3 训练'",
    "training_M1:'M1 训练：token 等权',training_M2:'M2 训练：偏差修正'")

if __name__=='__main__':
    live_progress.HTML=HTML
    live_progress.main()
