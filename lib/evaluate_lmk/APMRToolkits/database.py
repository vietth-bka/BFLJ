import os
import pickle
import json
import numpy as np
from .image import *
from data.CrowdHuman_json import COCO
from tqdm import tqdm
import PIL.Image as P
import matplotlib.pyplot as plt

# PERSON_CLASSES = ['background', 'person']
PERSON_CLASSES = ['background', 'person', 'face']
# DBBase
class Database(object):
    def __init__(self, gtpath=None, dtpath=None, body_key=None, head_key=None, mode=0, if_face=False):
        """
        mode=0: only body; mode=1: only head
        """
        self.images = dict()
        self.eval_mode = mode
        print('Eval mode:', self.eval_mode)
        self.loadData(gtpath, body_key, head_key, True, if_face)
        self.loadData(dtpath, body_key, head_key, False, if_face)
        
        tot_gt = [i for key, img in self.images.items() for i in img.gtboxes]
        tot_dt = [i for key, img in self.images.items() for i in img.dtboxes]
        print(f'tot gt: {len(tot_gt)}, tot_dt: {len(tot_dt)}')

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self.scorelist = None
        self.if_face = if_face

    def loadData(self, fpath, body_key=None, head_key=None, if_gt=True, if_face=False):
        assert os.path.isfile(fpath), fpath + " does not exist!"
        if 'odgt' in fpath or 'dump' in fpath:
            with open(fpath, "r") as f:
                lines = f.readlines()
            records = [json.loads(line.strip('\n')) for line in lines]
            print('Loading detections..')
        else:
            # coco = COCO(fpath)
            # records = coco.process_gt()
            records = pickle.load(open('../lib/data/CrHu_val4370.pkl','rb'))
            src = '../../data/CrowdHuman/CrowdHuman_val/Images/'
            for r in records:
                img = P.open(src+r['ID'])
                width, height = img.size
                r['width'] = width
                r['height'] = height

            # print(records[0])
            print('Loading gt..')
        if if_gt:
            for record in records:
                self.images[record["ID"]] = Image(self.eval_mode)   #record["ID"]: image's name
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True, if_face=if_face)
        else:
            for record in records:
                self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, False, if_face=if_face)
                self.images[record["ID"]].clip_all_boader()

    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        print('---Start comparing---')
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for idx, ID in enumerate(self.images):
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech_lmk(thres, self.if_face, idx)
                # result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
            # if self.if_face and idx==3:
            #     print(idx, len(result))
        
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][4], reverse=True)
        self.scorelist = scorelist
        print('***Length scorelist:', len(scorelist), 'face:', self.if_face)
        # if self.if_face:
        #     exit(-1)

    def eval_MR(self, ref="CALTECH_-2"):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst)-1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self.scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        fppiX, fppiY = list(), list() 
        for i, item in enumerate(self.scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0

            fn = (self._gtNum - self._ignNum) - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            missrate = 1.0 - recall
            fppi = fp / self._imageNum
            fppiX.append(fppi)
            fppiY.append(missrate)
        # print('***fppi:', fppiX[-10:])
        # print('***missrate:',fppiY[-10:])
        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY = list(), list() 
        total_det = len(self.scorelist)
        print('total_det:', total_det)
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        for i, item in tqdm(enumerate(self.scorelist)):
            assert len(item) == 3, 'wrong score list'
            if len(item) == 0:
                continue
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp/total_images)

        # APs = [i[0]*i[1] for i in zip(rpX, rpY)]
        # print('max AP:', max(APs), 'th:', thr[APs.index(max(APs))])
        AP = _calculate_map(rpX, rpY)
        return AP, (rpX, rpY, thr, fpn, recalln, fppi)

    def eval_AP_AL(self):
        """
        :meth: evaluate by average lmk mse
        """
        # calculate general ap score        
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h
            return area
        
        def _calculate_mapl(recall, precision, lmk):
            assert len(recall) == len(precision) == len(lmk)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i-1] + precision[i]) / 2
                delta_w = recall[i] - recall[i-1]
                area += delta_w * delta_h * (lmk[i] + lmk[i-1])/2
            return area

        tp, fp = 0.0, 0.0
        rpX, rpY, rpZ = list(), list(), list()
        total_det = len(self.scorelist)
        print('total_det:', total_det)
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        recalln = []
        thr = []
        fppi = []
        lmk_ls = []
        cnt, cnt0 = 0,0
        for i, item in tqdm(enumerate(self.scorelist)):
            assert len(item) == 4, f'wrong length of item {len(item)}'
            if len(item) == 0:
                print('Skip')
                continue
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            if item[3] != None:
                lmk_ls.append(item[3])
                lmk_mean = sum(lmk_ls)/len(lmk_ls)    #np.exp(-sum(lmk_ls)/len(lmk_ls))                
            elif len(lmk_ls) > 0:
                lmk_mean = rpZ[-1]                
            else:                
                lmk_mean = -1
            
            rpX.append(recall)
            rpY.append(precision)
            rpZ.append(lmk_mean)
            fpn.append(fp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp/total_images)
        
        if rpZ[0] == -1: rpZ[0] = rpZ[1]
        
        
        # APs = [i[0]*i[1] for i in zip(rpX, rpY)]
        # print('max AP:', max(APs), 'th:', thr[APs.index(max(APs))])
        AP = _calculate_map(rpX, rpY)
        # AL = _calculate_mapl(rpX, rpY, rpZ)
        AL = _calculate_map(rpX, rpZ)
        

        plt.plot(rpX, rpZ, linewidth=1.0, label=f'AL={AL:.4f}')
        plt.plot(rpX, rpY, linewidth=2.0, label=f'AP={AP:.4f}')
        plt.legend()
        plt.savefig("last_results.png")
        # plt.savefig("rpZ.png")
        # plt.show()
    
        return (AP, AL), (rpX, rpY, thr, fpn, recalln, fppi)
