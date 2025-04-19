from flask import Flask, render_template, jsonify
import re
from datetime import datetime
import os
import json

app = Flask(__name__, template_folder='.')

def parse_log_file(log_file_path):
    # 原有解析训练日志的代码保持不变
    data = []
    best_rewards = []
    current_best = -float('inf')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            eval_match = re.search(r'Step (\d+) \| Eval Reward: ([\d\.]+) \| Best: ([\d\.]+)', line)
            if eval_match:
                step = int(eval_match.group(1))
                reward = float(eval_match.group(2))
                best = float(eval_match.group(3))
                
                if best > current_best:
                    current_best = best
                    best_rewards.append({'step': step, 'reward': best})
                
                data.append({
                    'step': step,
                    'reward': reward,
                    'best': best,
                    'timestamp': datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f').timestamp()
                })
    
    return {
        'all_rewards': data,
        'best_rewards': best_rewards
    }

# 新添加的函数：读取评估结果
def get_eval_results():
    # 这里模拟您的eval_ppo.py输出，实际使用时可以从文件或数据库读取
    return {
        'eval_r': 0.5766333077887111,
        'eval_afg': 0.5488297390760813,
        'eval_for': 0.6228632629802577,
        'eval_robs': 0.5397805348308757,
        'eval_agv': 0.5704118661089694,
        'eval_arops': 0.8703773704445578
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_training_data():
    log_file = os.path.join('logs', 'ppo', 'training.log')
    if not os.path.exists(log_file):
        return jsonify({'error': 'Log file not found at {}'.format(log_file)}), 404
    
    data = parse_log_file(log_file)
    return jsonify(data)

# 新添加的路由：获取评估结果
@app.route('/eval_data')
def get_eval_data():
    eval_results = get_eval_results()
    return jsonify(eval_results)

if __name__ == '__main__':
    os.makedirs(os.path.join('logs', 'ppo'), exist_ok=True)
    app.run(debug=True)