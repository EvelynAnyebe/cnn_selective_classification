import json
import numpy as np
import os

# from glob import glob
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.metrics import log_loss
# from sklearn.model_selection import train_test_split


#receives a list of strings
def write_csv(filename,recs):
    f=open(filename,"a+")
#     print(recs)
    recs=recs.astype('str')
    for rec in recs:
        f.write("%s\n" % ','.join(list(rec)))
    f.close() 
    
def my_evaluation(y_true,y_pred):
    acc=np.sum(np.equal(y_true,y_pred))/y_true.shape[0] #accuracy
    return(acc)

def plot_cm(y_true, y_pred,label):
    cm = confusion_matrix(y_true, y_pred,labels=label)
    return cm

def covered_idx(c,g):
    q =100*(1-c)
    if(q<0):
        q = 0
    elif(q>100):
        q =100
    threshold = np.percentile(g,q)
    covered_idx = g > threshold
    return covered_idx

def sr_covered_idx(c,y_true,y_pred):
    sr = np.max(y_pred, axis=1)
    sr_sorted = np.sort(sr)
    threshold = sr_sorted[sr.shape[0] - int(c * sr.shape[0])]
    covered_idx = sr > threshold
    return covered_idx

def selective_risk(c,y_true,y_pred,g):
    covered_indx = covered_idx(c,g)
    coverage = np.mean(covered_indx)
    acc = np.sum( np.equal( np.argmax(y_pred[covered_indx],axis=1), 
                             np.argmax(y_true[covered_indx], axis=1)))/np.sum(covered_indx)
    risk = 1-acc
    return coverage, risk,acc

def sr_selective_risk(c, y_true, y_pred, mc=False):
    if(mc):
        sr = np.max(y_pred, axis=1)
    else:
        sr = mc_dropout(100)
    sr_sorted = np.sort(sr) 
    threshold = sr_sorted[sr.shape[0] - int(c * sr.shape[0])]
    covered_idx = sr > threshold
    coverage = np.mean(covered_idx)
    selective_acc = np.sum(np.equal(np.argmax(y_pred[covered_idx], 1),
                                    np.argmax(y_true[covered_idx], 1)))/np.sum(covered_idx)
    risk = 1-selective_acc
    return coverage, risk, selective_acc

# def to_train(filename):
#     checkpoints = os.listdir("checkpoints/")
#     if filename in checkpoints:
#         return False
#     else:
#         return True


# def save_dict(filename, dict):

#     with open(filename, 'w') as fp:
#         json.dump(dict, fp)


# def calc_selective_risk(model, regression, calibrated_coverage=None):
#     prediction, pred = model.predict()
#     if calibrated_coverage is None:
#         threshold = 0.5
#     else:
#         threshold = np.percentile(prediction[:, -1], 100 - 100 * calibrated_coverage)
#     covered_idx = prediction[:, -1] > threshold

#     coverage = np.mean(covered_idx)
#     y_hat = np.argmax(prediction[:, :-1], 1)
#     if regression:
#         loss = np.sum(np.mean((prediction[covered_idx, :-1] - model.y_test[covered_idx, :-1]) ** 2, -1)) / np.sum(
#             covered_idx)
#     else:
#         loss = np.sum(y_hat[covered_idx] != np.argmax(model.y_test[covered_idx, :], 1)) / np.sum(covered_idx)
#     return loss, coverage


# def train_profile(model_name, model_cls, coverages, model_baseline=None, regression=False, alpha=0.5):
#     results = {}
#     for coverage_rate in coverages:
#         print("running {}_{}.h5".format(model_name, coverage_rate))
#         model = model_cls(train=to_train("{}_{}.h5".format(model_name, coverage_rate)),
#                           filename="{}_{}.h5".format(model_name, coverage_rate),
#                           coverage=coverage_rate,
#                           alpha=alpha)

#         loss, coverage = calc_selective_risk(model, regression)

#         results[coverage] = {"lambda": coverage_rate, "selective_risk": loss}
#         if model_baseline is not None:
#             if regression:
#                 results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

#             else:

#                 results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
#             results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

#         save_dict("results/{}.json".format(model_name), results)


# def post_calibration(model_name, model_cls, lamda, calibrated_coverage=None, model_baseline=None, regression=False):
#     results = {}
#     print("calibrating {}_{}.h5".format(model_name, lamda))
#     model = model_cls(train=to_train("{}_{}.h5".format(model_name, lamda)),
#                       filename="{}_{}.h5".format(model_name, lamda), coverage=lamda)
#     loss, coverage = calc_selective_risk(model, regression, calibrated_coverage)

#     results[coverage]={"lambda":lamda, "selective_risk":loss}
#     if model_baseline is not None:
#         if regression:
#             results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

#         else:
#             results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
#             results[coverage]["mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))

#         results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

#     return results


# def my_generator(func, x_train, y_train, batch_size, k=10):
#     while True:
#         res = func(x_train, y_train, batch_size
#                    ).next()
#         yield [res[0], [res[1], res[1][:, :-1]]]


# def create_cats_vs_dogs_npz(cats_vs_dogs_path='datasets'):
#     labels = ['cat', 'dog']
#     label_to_y_dict = {l: i for i, l in enumerate(labels)}

#     def _load_from_dir(dir_name):
#         glob_path = os.path.join(cats_vs_dogs_path, dir_name, '*.*.jpg')
#         imgs_paths = glob(glob_path)
#         images = [resize_and_crop_image(p, 64) for p in imgs_paths]
#         x = np.stack(images)
#         y = [label_to_y_dict[os.path.split(p)[-1][:3]] for p in imgs_paths]
#         y = np.array(y)
#         return x, y

#     x_train, y_train = _load_from_dir('train')
#     x_test, y_test = _load_from_dir('test')

#     np.savez_compressed(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'),
#                         x_train=x_train, y_train=y_train,
#                         x_test=x_test, y_test=y_test)


# def load_cats_vs_dogs(cats_vs_dogs_path='datasets/'):
#     npz_file = np.load(os.path.join(cats_vs_dogs_path, 'cats_vs_dogs.npz'))
#     x_train = npz_file['x_train']
#     y_train = npz_file['y_train']
#     x_test = npz_file['x_test']
#     y_test = npz_file['y_test']

#     return (x_train, y_train), (x_test, y_test)
