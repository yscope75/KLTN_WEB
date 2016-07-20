import sys
import os
import datetime
import flask
import caffe
import traceback 
import cv2
import urllib
import cPickle
import fetutil
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import argparse as ap
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from werkzeug.utils import secure_filename
from optparse import OptionParser
from flask import send_from_directory
from sklearn.cluster import MiniBatchKMeans, KMeans

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads') 
ALLOWED_IMAGE_EXTENSTIONS = set(['jpg', 'bmp', 'png', 'jpeg', 'jpe'])
REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/..' + '/caffe')

# create flask app object
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dic = joblib.load('dics.pkl')
labels = joblib.load('labels.pkl')
# root url
@app.route('/')
def home():
    # get history images 
    history = fetutil.get_history()
    for h in history:
        h[0] = str(flask.url_for('uploaded_file', filename=h[0]))
    return flask.render_template('home.html', has_result = False, history=history)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    """
        Claasify an image with imageurl
    """
    imageurl = flask.request.args.get('imageurl', '')
    if app.classifier == app.SVMClf:
        try:
            # read image from url use below codes
            resp = urllib.urlopen(imageurl)
            image = np.array(bytearray(resp.read()),dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            return flask.render_template(
                'home.html', has_result=True,
                result=(False, 'Cannot read image from URL.')
            )
    else :
        try:
            string_buffer = StringIO.StringIO(
                urllib.urlopen(imageurl).read())
            image = caffe.io.load_image(string_buffer)
        except Exception as e:
            # not continue if have any exeption
            return flask.render_template(
                'home.html', has_result=True,
                result=(False, 'Cannot read image from URL.')
            )
    result = app.classifier.classify(image)
    return flask.render_template('home.html', has_result=True, result=result, imagesrc=imageurl)

@app.route('/classify_upload', methods= ['POST'])
def classify_upload():
    """
        Prerocess uploaded file then call classify 
    """
    imageupload = flask.request.files['imagefile']
    try:
        if imageupload.filename == '':
            flash('No slected file')
            return redirect(request.url)
        if imageupload and allowed_image(imageupload.filename):
            imagename_ = secure_filename(imageupload.filename)
            imagename_ = str(datetime.datetime.now()).replace(' ', '_') + imagename_
            app.imName = imagename_
            imagename = os.path.join(UPLOAD_FOLDER, imagename_) 
            imageupload.save(imagename) 
        if app.classifier == app.SVMClf:
            image = cv2.imread(imagename, cv2.CV_LOAD_IMAGE_COLOR) 
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
        else:
            im = Image.open(imagename)
            image = np.asarray(im).astype(np.float32) / 255.
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
                image = np.tile(image, (1, 1, 3))
            elif image.shape[2] == 4:
                image = image[:, :, :3]
    except Exception as e:
        return flask.render_template('home.html', has_result=True, result=(False, str(traceback.format_exc()) ))

    result = app.classifier.classify(image)
    if result[0]:
        fetutil.save_features(app, result[2])
    return flask.render_template('home.html', has_result=True, result=result, imagesrc= str(flask.url_for('uploaded_file', filename=app.imName)))

@app.route('/radio_change', methods=['GET', 'POST'])
def apply_change():
    checked = int(flask.request.form['selected'])
    if checked == 1:
        app.classifier = app.LeNetClf
    elif checked == 2:
        app.classifier = app.AlexClf
    elif checked == 3:
        app.classifier = app.SVMClf
    return ('', 204)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
def allowed_image(imagename):
    return '.' in imagename and \
        imagename.rsplit('.',1)[1] in ALLOWED_IMAGE_EXTENSTIONS

def start_from_terminal(app):
    """
        parse command line options
    """
    parser = OptionParser()
    parser.add_option(
        '-g', '--gpu',
        help='use gpu mode',
        action='store_true', default=False
    )
    parser.add_option(
        '-d', '--debug',
        help='use debug mode',
        action='store_true', default=True
    )
    opts, args = parser.parse_args()
    DeepClassifier.alex_args.update({'gpu_mode': opts.gpu})
    DeepClassifier.googlenet_args.update({'gpu_mode': opts.gpu})
    app.imName = ''
    # Init classifier
    app.AlexClf = DeepClassifier(**DeepClassifier.alex_args)
    app.AlexClf.net.forward()
    app.LeNetClf = DeepClassifier(**DeepClassifier.googlenet_args)
    app.LeNetClf.net.forward()
    app.SVMClf = SVMClassifier()
    app.classifier = app.LeNetClf
    app.run(debug=True)

def check_model_args(model_args):
    for key, val in model_args.iteritems():
        if not os.path.exists(val):
            return False
    return True

class DeepClassifier(object):
    """docstring for Classifier
        Setup pretrained model with parameters from flask object
    """
    #args for alex_nets model
    alex_args = {
        'model_def_file': (
            '{}/models/bvlc_alexnet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_alexnet/bvlc_alexnet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
        'svm_trained': 'FC8_classifier_Scaler.pkl',
        'name': 'Alex',
        'image_dim': 256,
        'raw_scale': 255.,
    }

    # args for googlenet
    googlenet_args = {
        'model_def_file': (
            '{}/models/bvlc_googlenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_googlenet/bvlc_googlenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
        'svm_trained': 'CNN_GoogLeNet_classifier_Scaler.pkl',
        'name': 'Google',
        'image_dim': 256,
        'raw_scale': 255.,
    }

    
    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                class_labels_file, bet_file, raw_scale, image_dim, gpu_mode, svm_trained, name):
        self.svm = svm_trained
        self.name = name 
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim),
            raw_scale= raw_scale,
            mean=np.load(mean_file).mean(1).mean(1),
            channel_swap=(2, 1, 0)
        )

    def classify(self, image):
        try:
            scores = self.net.predict([image], oversample=True)
            if self.name ==                                                                                                                                                                                                      'Alex':
                feature = self.net.blobs['fc8'].data[0]
            else :
                feature = self.net.blobs['loss3/classifier'].data[0]
            SVM, stdSlr = joblib.load(self.svm)
            im_feature = stdSlr.transform(feature)
            index =  SVM.predict(im_feature)
            predictions = labels[index[0]]
            return (True, [(1,1)], dic[predictions])
        except Exception as e:

            return (False, 'Some thing went wrong when classifying !')
class SVMClassifier(object):
    """docstring for SVMClassifier"""
    def __init__(self):
        self.kmean = joblib.load('k_means_8000.pkl')
        self.SVM = joblib.load('SVM_8000_notTransform.pkl')
        self.fea_det = cv2.FeatureDetector_create("SIFT")
        self.des_ext = cv2.DescriptorExtractor_create("SIFT")

    def classify(self, image):
        try:
            kpts = self.fea_det.detect(image)
            kpts, des = self.des_ext.compute(image, kpts)
            input_feature = np.zeros((1,self.kmean.n_clusters), "float32")
            words = self.kmean.predict(des)
            for w in words:
                input_feature[0][w] += 1

            self.features = des
            # perform the predictions
            predictions = self.SVM.predict(input_feature)
            return (True, [(1,1)], dic[labels[predictions]])
        except Exception as e:
            return (False, 'Some thing went wrong when classifying with SVM !')


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(fetutil.FEATURES_FOLDER):
        os.makedirs(fetutil.FEATURES_FOLDER)
    start_from_terminal(app)
