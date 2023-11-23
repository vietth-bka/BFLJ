import argparse
from .APMRToolkits import *

dbName = 'human'
def compute_APMR(dt_path, gt_path, target_key=None, mode=3, if_face=False):
    print('Mode:', mode)
    database = Database(gt_path, dt_path, target_key, None, mode, if_face)
    database.compare()    
    if mode==0:
        mAP,_ = database.eval_AP()
        mMR,_ = database.eval_MR()
        # line = 'AP:{:.4f}, MR:{:.4f}.'.format(mAP, mMR)
        # return mAP, mMR
        if not if_face:
            line = 'BODY-->AP:{:.4f}, MR:{:.4f}.'.format(mAP, mMR)            
        else:
            line = 'FACE-->AP:{:.4f}, MR:{:.4f}.'.format(mAP, mMR)            
    elif mode==3:
        if if_face:
            (mAP, mAL),_ = database.eval_AP_AL()
            mMR,_ = database.eval_MR()
            line = 'FACE-->AP:{:.4f}, AL:{:.4f} MR:{:.4f}.'.format(mAP, mAL, mMR)
        else:
            mAP,_ = database.eval_AP()
            mMR,_ = database.eval_MR()
            line = 'BODY-->AP:{:.4f}, MR:{:.4f}.'.format(mAP, mMR)
    
    print(line)
    return mAP, mMR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze a json result file with iou match')
    parser.add_argument('--detfile', required=True, help='path of json result file to load')
    parser.add_argument('--target_key', default=None, required=True)
    args = parser.parse_args()
    compute_APMR(args.detfile, args.target_key, 0)
