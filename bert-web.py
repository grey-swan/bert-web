import numpy as np
from flask import Flask, jsonify, request, render_template
import memcache
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

bc = BertClient(port=5555, port_out=5556)
mc = memcache.Client(['127.0.0.1:11211'], debug=False)
with open('data/train.tsv', encoding='utf-8') as f:
    questions = [v.split('\t')[0].strip() for v in f]


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/cache/upload/')
def set_cache():
    if not mc.get('vecs'):
        vecs = bc.encode(questions)
        mc.set('vecs', vecs, time=86400)

    return jsonify({'status': 1, 'msg': 'upload success'})


@app.route('/similarity/', methods=['GET', 'POST'])
def get_similarity():
    answer = ''
    pro = None
    if request.method.upper() == 'POST':
        q = request.form['q']
        if q:
            vecs = mc.get('vecs')

            v = bc.encode([q])
            sim = cosine_similarity(v[0].reshape(-1, 768), vecs)
            sim_index = np.argsort(sim)
            idx = sim_index[0][::-1][0]

            pro = sim[0][idx]
            pro = '%.1f%%' % (float(pro) * 100)
            answer = questions[idx]

    return render_template('index.html', answer=answer, pro=pro)


if __name__ == '__main__':
    app.run()
