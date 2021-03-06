# _*_ coding: utf-8 _*_
from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np



app = Flask(__name__)

X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4]) #4개변인 들어갈수있게만듬
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) #얜 1개
W = tf.Variable(tf.random.normal([4, 1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")


hypothesis = tf.matmul(X, W) + b


saver = tf.compat.v1.train.Saver() #저장된 모델을 가져오는 객체
model = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(model)

save_path = "./model/cabbage/saved.cpkt"
saver.restore(sess, save_path)


@app.route("/", methods=['GET', 'POST'])
def indext():##indext?
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if str(request.form['vegit_category']) in ['cabbage', '배추']:
            save_path = './model/cabbage/saved.cpkt'
        elif str(request.form['vegit_category']) in ['lettuce', '상추']:
            save_path = './model/lettuce/lettuce_saved.cpkt'
        elif str(request.form['vegit_category']) in ['radish', '무']:
            save_path = './model/radish/radish_saved.cpkt'
        elif str(request.form['vegit_category']) in ['rice', '쌀']:
            save_path = './model/rice/rice_saved.cpkt'
        elif str(request.form['vegit_category']) in ['leek','springonion','welshonion','파']:
            save_path = './model/springonion/springonion_saved.cpkt'
        elif str(request.form['vegit_category']) in ['onion','Onion', '양파']:
            save_path = './model/onion/onion_saved.cpkt'
        elif str(request.form['vegit_category']) in ['pear', '배']:
            save_path = './model/pear/pear_saved.cpkt'
            
        saver.restore(sess, save_path)
        
        avg_temp = float(request.form['avg_temp'])
        max_temp = float(request.form['max_temp'])
        min_temp = float(request.form['min_temp'])      
        rain_fall = float(request.form['rain_fall'])

    price = 0

    data = ((avg_temp, min_temp, max_temp, rain_fall),)
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data})

    price = dict[0]
    return render_template('index.html', price=price)
    

        
if __name__ == '__main__':
    app.run(debug = True)
