import numpy as np
from flask import Flask, jsonify, request
import memcache
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

with open('data/train.tsv', encoding='utf-8') as f:
    questions = [v.split('\t')[0].strip() for v in f]


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/cache/upload/')
def set_cache():
    bc = BertClient(port=5555, port_out=5556)
    mc = memcache.Client(['127.0.0.1:11211'], debug=True)
    if not mc.get('vecs'):
        vecs = bc.encode(questions)
        mc.set('vecs', vecs)
    bc.close()

    return jsonify({'status': 1, 'msg': 'upload success'})


@app.route('/similarity/', methods=['GET'])
def get_similarity():
    q = request.args.get('q', '其他')
    mc = memcache.Client(['127.0.0.1:11211'], debug=False)
    vecs = mc.get('vecs')

    bc = BertClient(port=5555, port_out=5556)
    v = bc.encode([q])
    bc.close()
    sim = cosine_similarity(v[0].reshape(-1, 768), vecs)
    sim_index = np.argsort(sim)
    idx = sim_index[0][::-1][0]

    pro = sim[0][idx]
    answer = questions[idx]

    return jsonify({'status': 1, 'msg': {'answer': answer, 'value': float(pro)}})


if __name__ == '__main__':
    app.run()
