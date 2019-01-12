import numpy as np
from flask import Flask, jsonify, request
import memcache
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

bc = BertClient(port=8100, port_out=8101)
mc = memcache.Client(['127.0.0.1:11211'], debug=False)
with open('data/train.tsv', encoding='utf-8') as f:
    questions = [v.split('\t')[0].strip() for v in f]


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/cache/upload/')
def set_cache():
    if not mc.get('vecs'):
        vecs = bc.encode(questions[:100])
        mc.set('vecs', vecs, time=86400)

    return jsonify({'status': 1, 'msg': 'upload success'})


@app.route('/similarity/', methods=['GET'])
def get_similarity():
    q = request.args.get('q', '其他')

    vecs = mc.get('vecs')

    v = bc.encode([q])
    sim = cosine_similarity(v[0].reshape(-1, 768), vecs)
    sim_index = np.argsort(sim)
    idx = sim_index[0][::-1][0]

    pro = sim[0][idx]
    answer = questions[idx]

    return jsonify({'status': 1, 'msg': {'answer': answer, 'value': float(pro)}})


if __name__ == '__main__':
    app.run()
