import numpy as np
import PIL.Image as P

class Image(object):
    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None

    def load(self, record, body_key, head_key, class_names, gtflag, if_face=False):
        """
        :meth: read the object from a dict
        body_key = 'box'
        head_key = None
        class_name = PERSON_CLASSES
        if_face = False for body, True for face
        """
        if "ID" in record and self.ID is None:
            self.ID = record['ID']
        if "width" in record and self._width is None:
            self._width = record["width"]
        if "height" in record and self._height is None:
            self._height = record["height"]

        if gtflag:
            # self._gtNum = len(record["gtboxes"])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes', class_names, if_face)
            # print('body_bbox:', body_bbox)
            # print('record:', record)
            self._gtNum = len(body_bbox)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array([body_bbox[i,-1]!=-1 and head_bbox[i,-1]!=-1 for i in range(len(body_bbox))])
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack((body_bbox[:, :-1], head_bbox[:, :-1], gt_tag.reshape(-1, 1)))
            elif self.eval_mode == 3:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, 4] == -1).sum()
            else:
                raise Exception('Unknown evaluation mode!')
        if not gtflag:
            self._dtNum = len(record["dtboxes"])
            if self.eval_mode == 0 or self.eval_mode == 3:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key, 'score', if_face=if_face)
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))            
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()

        # dt_matched = np.zeros(dtboxes.shape[0])
        # gt_matched = np.zeros(gtboxes.shape[0])

        # dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        # gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if isinstance(dtboxes, list) or dtboxes.shape[-1] < 4:
            return []
        if isinstance(gtboxes, list) or gtboxes.shape[-1] < 4:
            scorelist = list()
            for i, dt in enumerate(dtboxes):
                scorelist.append((dt, 0, self.ID))
            return scorelist
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))  #Dx5
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))  #Gx5
        # print('s-dtboxes:', dtboxes)
        # print('s-gtboxes:', gtboxes)
        
        if isinstance(dtboxes, list) or dtboxes.shape[-1] < 4:
            return []
        if isinstance(gtboxes, list) or gtboxes.shape[-1] < 4:
            scorelist = list()
            for i, dt in enumerate(dtboxes):
                scorelist.append((dt, 0, self.ID))
            return scorelist

        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes, True)  #DxG
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes, False) #DxG
        else:
            return list()

        scorelist = list()        
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    # gtboxes are sorted as the descent of tag. 
                    # Consider gt with positive tags first then compare to the negative ones later.
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j
                            # maxpos = -1 # should be -1 because tag -1 is for ignored box            
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                #this branch is for maxpos=-1
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))        
        return scorelist
    
    def compare_caltech_lmk(self, thres, if_face, idx):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        
        if isinstance(dtboxes, list) or dtboxes.shape[-1] < 4 or len(dtboxes)==0:
            return []
        if isinstance(gtboxes, list) or gtboxes.shape[-1] < 4 or len(gtboxes)==0:
            scorelist = list()
            for i, dt in enumerate(dtboxes):
                if self.eval_mode == 3 and if_face:
                    scorelist.append((dt, 0, self.ID, 0.0))
                else:
                    scorelist.append((dt, 0, self.ID))
            return scorelist

        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[4], reverse=True))  #Dx16
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[4], reverse=True))  #Gx16
        
        if self.eval_mode == 3 and if_face:
            assert dtboxes.shape[-1] == gtboxes.shape[-1] == 15, f'wrong {dtboxes.shape[-1]} != {gtboxes.shape[-1]}, {self.ID}'
        else:
            assert dtboxes.shape[-1] == gtboxes.shape[-1] == 5

        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes, True)  #DxG
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes, False) #DxG
        else:
            return list()

        scorelist = list()        
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[4] > 0:
                    overlap = overlap_iou[i][j]
                    if overlap > maxiou:
                        maxiou = overlap
                        maxpos = j
                else:
                    # gtboxes are sorted as the descent of tag. 
                    # Consider gt with positive tags first then compare to the negative ones later.
                    if maxpos >= 0:
                        break
                    else:
                        overlap = overlap_ioa[i][j]
                        if overlap > thres:
                            maxiou = overlap
                            maxpos = j
                            # maxpos = -1 # should be -1 because tag -1 is for ignored box
            if maxpos >= 0:
                if gtboxes[maxpos, 4] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    if self.eval_mode == 3 and if_face:
                        assert gtboxes[maxpos].shape[-1] == 15
                        if gtboxes[maxpos][-10:].sum() == 0:
                            scorelist.append((dt, 1, self.ID, None))
                        else:
                            scorelist.append((dt, 1, self.ID, lmk_cost(dt, gtboxes[maxpos])))
                    else:
                        scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                #this branch is for maxpos=-1
                dt_matched[i] = 0                
                if self.eval_mode == 3 and if_face:                     
                    scorelist.append((dt, 0, self.ID, 0.0))                    
                else:
                    scorelist.append((dt, 0, self.ID))
        
        # print('len scorelist', len(scorelist))
        # with open('checkzero.txt', 'a') as f:
        #     for s in scorelist:
        #         f.write(str(s) +'\n')
        # f.close()
        return scorelist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:,None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:,None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def box_overlap_opr(self, dboxes:np.ndarray, gboxes:np.ndarray, if_iou):
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4, \
                                f'wrong shape of dboxes {dboxes.shape[1]} or gboxes {gboxes.shape[1]}'
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes[:,:4], axis = 1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes[:,:4], axis = 0), (N, 1, 1))

        iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
        ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])        
        if if_iou:
            gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1]) 
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):
        def _clip_boundary(boxes,height,width):
            assert boxes.shape[-1]>=4
            boxes[:,0] = np.minimum(np.maximum(boxes[:,0],0), width - 1)
            boxes[:,1] = np.minimum(np.maximum(boxes[:,1],0), height - 1)
            boxes[:,2] = np.maximum(np.minimum(boxes[:,2],width), 0)
            boxes[:,3] = np.maximum(np.minimum(boxes[:,3],height), 0)
            return boxes

        assert self.dtboxes.shape[-1]>=4
        assert self.gtboxes.shape[-1]>=4
        assert self._width is not None and self._height is not None
        if self.eval_mode == 2:
            self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4], self._height, self._width)
            self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4], self._height, self._width)
            self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8], self._height, self._width)
            self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8], self._height, self._width)
        else:
            self.dtboxes = _clip_boundary(self.dtboxes, self._height, self._width)
            self.gtboxes = _clip_boundary(self.gtboxes, self._height, self._width)

    def load_gt_boxes(self, dict_input, key_name, class_names, if_face=False):
        """
        This function is for loading both bodies and faces gt.
        """
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = 1
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            if 'hbox' in rb and 'fbox' in rb:
                head_bbox.append(np.hstack((rb['hbox'], head_tag)))
                body_bbox.append(np.hstack((rb['fbox'], body_tag)))
            else:
                head_bbox.append(np.hstack((rb['bbox'], head_tag)))
                if not if_face:
                    body_bbox.append(np.hstack((rb['bbox'], body_tag)))
                else:
                    if self.eval_mode == 3:                        
                        if rb['fbox'][2] > 1:
                            if body_tag == -1:
                                tag_f = -1
                                body_bbox.append(np.hstack((rb['bbox'], tag_f, self.lmk_ratio(rb['lmk']))))                                
                            else:
                                tag_f = 2
                                body_bbox.append(np.hstack((rb['fbox'], tag_f, self.lmk_ratio(rb['lmk']))))
                        else:
                            if body_tag == -1:
                                body_bbox.append(np.hstack((rb['bbox'], -1, self.lmk_ratio(rb['lmk']))))
            
                    else:
                        if rb['fbox'][2] > 1:
                            if body_tag == -1:
                                tag_f = -1
                                body_bbox.append(np.hstack((rb['bbox'], tag_f)))
                                # body_bbox.append(np.hstack((rb['fbox'], tag_f)))
                            else:
                                tag_f = 2
                                body_bbox.append(np.hstack((rb['fbox'], tag_f)))
                        else:
                            if body_tag == -1:
                                body_bbox.append(np.hstack((rb['bbox'], -1)))
                    
        head_bbox = np.array(head_bbox)
        head_bbox[:, 2:4] += head_bbox[:, :2]
        if len(body_bbox) < 1:
            return np.empty([0, 5]), head_bbox
        else:
            body_bbox = np.array(body_bbox)
            body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    def load_det_boxes(self, dict_input, key_name, key_box, key_score=None, key_tag=None, if_face=False):
        """ 
        dict_input=record
        key_name='dtboxes'
        key_box='box'
        key_score='score'
        key_tag=None
        """
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score], rb[key_tag])) for rb in dict_input[key_name]])
            else:                
                if not if_face:
                    bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score])) for rb in dict_input[key_name] if rb['tag'] == 1])
                else:
                    if self.eval_mode == 3:
                        try:    #should try 'cause the images may have no faces, should avoid vstack nothing.
                            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score], self.lmk_ratio(rb['lmk'])))
                                                                                        for rb in dict_input[key_name] if rb['tag'] == 2])
                        except:                            
                            bboxes = np.empty([0, 15])
                    else:
                        try:
                            bboxes = np.vstack([np.hstack((rb[key_box], rb[key_score])) for rb in dict_input[key_name] if rb['tag'] == 2])
                        except:
                            bboxes = np.empty([0, 5])
        else:
            if key_tag:
                bboxes = np.vstack([np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]])
            else:
                bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist

    def lmk_ratio(self, lmk):
        if len(lmk) > 10: lmk = lmk[:-1]
        assert len(lmk)==10, len(lmk)
        lmk = np.array(lmk)
        # print('lmk:', lmk)
        lmk[:10:2] /= self._width
        lmk[1:11:2] /= self._height
        # lmk = np.clip(lmk, a_min=0.0, a_max=1.0)
        return list(lmk)

def lmk_cost(dtbox, gtbox):
    dt = dtbox[-10:]
    gt = gtbox[-10:]
    assert dt.shape[-1]==gt.shape[-1]==10, 'Wrong shape of lmks'
    # cost = np.sqrt(np.power(gt-dt, 2).sum()/dt.shape[-1]) #euclid distance
    cost = np.abs(gt-dt).sum()/dt.shape[-1]   #l1-smooth
    return np.exp(-cost)    