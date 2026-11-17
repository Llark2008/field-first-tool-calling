"""Loopback progress view for the matched fields SFT experiment."""
from pathlib import Path
import sys
from toolcall_lab import live_progress

ROOT=Path.home()/'.local/share/local-tool-calling/runs/fields-sft-20261115'

def main():
    live_progress.HTML=live_progress.HTML.replace('本地工具调用实验','字段先行微调 · J / F 对照').replace('A：显式缺参训练 · B：自然缺参训练 · 数据 / 训练 / 评测全流程','600 条相同配对样本 · J 直接动作 / F 字段后动作 · 各 75 次更新 · 开发集与已见训练回放').replace('完整任务能力保留条件未通过时，最终测试保持封存。','完整任务、缺参识别、误追问联合判断；封存测试本轮不使用。')
    live_progress.HTML=live_progress.HTML.replace('training_A','training_J').replace('training_B','training_F').replace('A 组训练','J 直接动作训练').replace('B 组训练','F 字段先行训练').replace('final_eval','replay').replace('最终测试评测','已见训练回放').replace('selection','development').replace('检查点选择','开发集评测')
    live_progress.HTML=live_progress.HTML.replace('损失下降不等于实际任务成功率提高。','各组目标长度不同，原始 loss 不作跨组排名；实际生成准确率为主。')
    sys.argv=[sys.argv[0],'--root',str(ROOT),'--port','49413'];live_progress.main()

if __name__=='__main__':main()
